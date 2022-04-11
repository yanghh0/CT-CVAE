#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Transformer encoder implementations.
"""

from __future__ import annotations
from typing import Tuple, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import math

from parlai.agents.ctcvae.modules import (
    create_position_codes,
    get_n_positions_from_options,
    LAYER_NORM_EPS,
    MultiHeadAttention,
    TransformerFFN,
)
from parlai.agents.ctcvae.modules.modular import swappable
from parlai.agents.ctcvae.modules.ctr_dict import Length_dict, Symbol_dict, Specific_dict
from parlai.core.opt import Opt
from parlai.utils.misc import warn_once
from parlai.utils.torch import PipelineHelper
import torch.nn.functional as F


def sample_gaussian(mu, logvar):
    epsilon = logvar.new_empty(logvar.size()).normal_()
    std = torch.exp(0.5 * logvar)
    z = mu + std * epsilon
    return z


def gaussian_kld(recog_mu, recog_logvar, prior_mu, prior_logvar):
    kld = -0.5 * torch.sum(1 - (prior_logvar - recog_logvar)
                             - torch.div(torch.pow(prior_mu - recog_mu, 2), torch.exp(prior_logvar))
                             - torch.div(torch.exp(recog_logvar), torch.exp(prior_logvar)), 1)
    return kld


class RecognitionNetwork(nn.Module):
    def __init__(
        self,
        recog_input_size,
        latent_size
    ):
        super().__init__()
        self.recogNet = nn.Linear(recog_input_size, latent_size * 2)
        nn.init.xavier_normal_(self.recogNet.weight)

    def forward(self, inputs):
        recog_mulogvar = self.recogNet(inputs)
        recog_mu, recog_logvar = torch.chunk(recog_mulogvar, 2, 1)
        return recog_mu, recog_logvar


class PriorNetwork(nn.Module):
    def __init__(
        self,
        prior_input_size,
        latent_size
    ):
        super().__init__()
        self.priorNet = nn.Sequential(
            nn.Linear(prior_input_size, 256),
            nn.Tanh(),
            nn.Linear(256, latent_size * 2)
        )
        nn.init.xavier_normal_(self.priorNet[0].weight)
        nn.init.xavier_normal_(self.priorNet[2].weight)

    def forward(self, inputs):
        prior_mulogvar = self.priorNet(inputs)
        prior_mu, prior_logvar = torch.chunk(prior_mulogvar, 2, 1)
        return prior_mu, prior_logvar


class BowProject(nn.Module):
    def __init__(
        self,
        bow_input_size,
        num_features
    ):
        super().__init__()
        self.bow_project = nn.Sequential(
            nn.Linear(bow_input_size, 512),
            nn.Tanh(),
            nn.Linear(512, num_features)
        )
        nn.init.xavier_normal_(self.bow_project[0].weight)
        nn.init.xavier_normal_(self.bow_project[2].weight)

    def forward(self, inputs):
        bow_output = self.bow_project(inputs)
        bow_logits = F.log_softmax(bow_output, dim=1)
        return bow_logits


@swappable(self_attention=MultiHeadAttention, feedforward=TransformerFFN)
class TransformerEncoderLayer(nn.Module):
    """
    Implements a single Transformer encoder layer.
    """

    def __init__(
        self,
        n_heads: int,
        d_size: int,
        ffn_size: int,
        attention_dropout: float = 0.0,
        relu_dropout: float = 0.0,
        dropout: float = 0.0,
        activation: str = 'relu',
        variant: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.dim = d_size
        self.ffn_dim = ffn_size
        self.activation = activation
        self.variant = variant
        self.attention = self.swappables.self_attention(  # type: ignore
            n_heads, d_size, dropout=attention_dropout  # --attention-dropout
        )
        self.norm1 = torch.nn.LayerNorm(d_size, eps=LAYER_NORM_EPS)
        self.ffn = self.swappables.feedforward(  # type: ignore
            d_size,
            ffn_size,
            relu_dropout=relu_dropout,
            activation=self.activation,
        )
        self.norm2 = torch.nn.LayerNorm(d_size, eps=LAYER_NORM_EPS)
        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self, tensor: torch.Tensor, mask: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        """
        Forward pass.
        """
        residual = tensor
        if self.variant == 'prelayernorm':
            tensor = self.norm1(tensor)
        attended_tensor = self.attention(tensor, mask=mask)[0]
        tensor = residual + self.dropout(attended_tensor)
        if self.variant == 'aiayn' or self.variant == 'xlm' or self.variant == 'bart':
            tensor = self.norm1(tensor)
        residual = tensor
        if self.variant == 'prelayernorm':
            tensor = self.norm2(tensor)
        tensor = residual + self.dropout(self.ffn(tensor))
        if self.variant == 'aiayn' or self.variant == 'xlm' or self.variant == 'bart':
            tensor = self.norm2(tensor)
        tensor *= mask.unsqueeze(-1).type_as(tensor)
        return tensor

attention_head_num = 2

@swappable(
    layer=TransformerEncoderLayer,
    encoder_attention=MultiHeadAttention,
)
class TransformerEncoder(nn.Module):
    """
    Transformer encoder module.

    For documentation on parameters that are take directly from opt,
    see parlai/agents/transformer/transformer.py

    :param opt: ParlAI-parsed options.
    :param vocabulary_size: Count of tokens/words in the dictionary.
    :param embedding: an embedding matrix for the bottom layer of the transformer.
        If none, one is created for this encoder.
    :param int padding_idx: Reserved padding index in the embeddings matrix.
    :param str reduction_type: Type of reduction at the end of the encoder.
    :param int n_positions: Size of the position embeddings matrix.
    :param int n_segments: Number of segments/lang/sentence embeddings.
    :param bool embeddings_scale: Scale embeddings relative to their dimensionality.
        Found useful in fairseq.
    """

    def __init__(
        self,
        opt: Opt,
        vocabulary_size: int,
        embedding: Optional[nn.Embedding] = None,
        turn_embedding: Optional[nn.Embedding] = None,
        role_embedding: Optional[nn.Embedding] = None,
        padding_idx: int = 0,
        reduction_type: str = 'mean',
        n_positions: Optional[int] = None,
        n_segments: Optional[int] = None,
        embeddings_scale: Optional[bool] = None,
        dropout: Optional[float] = None,
        activation: Optional[str] = None,
        variant: Optional[str] = None,
        output_scaling: Optional[float] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        def _default(val, default):
            return val if val is not None else default

        self.opt = opt
        self.embedding_size = opt['embedding_size']
        self.d_model = opt['d_model']
        self.n_heads = opt['n_heads']

        if self.opt["element"] == "D":
            self.control_embeddings = nn.Embedding(
                # len(Length_dict),
                # len(Symbol_dict),
                len(Specific_dict),
                opt['control_embedding'], 
                sparse=False
            )
            self.priorNet_mulogvar = PriorNetwork(self.d_model + opt['control_embedding'], opt['latentsize'])
        elif self.opt["element"] == "D+":
            self.control_embeddings_symbol = nn.Embedding(
                len(Symbol_dict),
                opt['control_embedding'], 
                sparse=False
            )
            self.control_embeddings_specific = nn.Embedding(
                len(Specific_dict),
                opt['control_embedding'], 
                sparse=False
            )
            self.priorNet_mulogvar = PriorNetwork(self.d_model + opt['control_embedding'] * 2, opt['latentsize'])
        else:
            self.priorNet_mulogvar = PriorNetwork(self.d_model, opt['latentsize'])

        self.recogNet_mulogvar = RecognitionNetwork(self.d_model, opt['latentsize'])    
        self.bow_project = BowProject(self.d_model + opt['latentsize'], vocabulary_size)

        codes = torch.empty(attention_head_num, self.d_model).cuda()
        codes = torch.nn.init.uniform_(codes)
        self.codes = torch.nn.Parameter(codes)
        self.w_encoder_attention = self.swappables.encoder_attention(
            self.n_heads, self.d_model, dropout=0.1
        )
        self._linear_layer = nn.Linear(self.d_model * attention_head_num, self.d_model)
        nn.init.xavier_normal_(self._linear_layer.weight)

        self.ffn_size = opt['ffn_size']
        self.n_layers = (
            opt['n_encoder_layers']
            if opt.get('n_encoder_layers', -1) > 0
            else opt['n_layers']
        )
        self.dim = self.embedding_size
        self.embeddings_scale = _default(
            embeddings_scale, opt.get('embeddings_scale', False)
        )
        self.reduction_type = reduction_type
        self.padding_idx = padding_idx
        # this is --dropout, not --relu-dropout or --attention-dropout
        self.dropout_frac = _default(dropout, opt.get('dropout', 0.0))
        self.dropout = nn.Dropout(p=self.dropout_frac)
        self.variant = _default(variant, opt.get('variant', 'aiayn'))
        self.n_segments = _default(n_segments, opt.get('n_segments', 0))

        self.n_positions = _default(n_positions, get_n_positions_from_options(opt))
        self.out_dim = self.embedding_size
        assert (
            self.d_model % self.n_heads == 0
        ), 'Transformer embedding size must be a multiple of n_heads'

        # check input formats:
        if embedding is not None:
            assert (
                self.embedding_size is None
                or self.embedding_size == embedding.weight.shape[1]
            ), "Embedding dim must match the embedding size."

        if embedding is not None:
            self.embeddings = embedding
        else:
            raise AssertionError(
                "This code should not execute. Left here in case we want to enable it."
            )
            assert self.padding_idx is not None
            self.embeddings = nn.Embedding(
                vocabulary_size, self.embedding_size, padding_idx=padding_idx
            )
            nn.init.normal_(self.embeddings.weight, 0, self.embedding_size ** -0.5)

        # create the positional embeddings
        self.position_embeddings = nn.Embedding(self.n_positions, self.d_model)
        if not opt.get('learn_positional_embeddings', False):
            create_position_codes(
                self.n_positions,
                self.d_model,
                out=self.position_embeddings.weight,
            )
        else:
            nn.init.normal_(
                self.position_embeddings.weight, 0, self.embedding_size ** -0.5
            )

        # embedding normalization
        if (
            self.variant == 'xlm'
            or self.variant == 'prelayernorm'
            or self.variant == 'bart'
        ):
            self.norm_embeddings = torch.nn.LayerNorm(self.dim, eps=LAYER_NORM_EPS)
        elif self.variant == 'aiayn':
            pass
        else:
            raise ValueError("Can't handle --variant {}".format(self.variant))

        if self.n_segments >= 1:
            self.segment_embeddings = nn.Embedding(self.n_segments, self.dim)

        self.turn_embeddings = turn_embedding
        self.role_embeddings = role_embedding

        # build the model
        self.layers = nn.ModuleList()
        for _ in range(self.n_layers):
            self.layers.append(
                self.swappables.layer(  # type: ignore
                    self.n_heads,
                    self.d_model,
                    self.ffn_size,
                    attention_dropout=opt.get('attention_dropout', 0.0),
                    relu_dropout=opt.get('relu_dropout', 0.0),
                    dropout=self.dropout_frac,
                    variant=self.variant,
                    activation=_default(activation, opt.get('activation', 'relu')),
                )
            )
        self.output_scaling = _default(output_scaling, opt.get('output_scaling', 1.0))

        self.input_layer = nn.Linear(self.embedding_size, self.d_model, bias=False)
        nn.init.xavier_normal_(self.input_layer.weight)

    def forward_embedding(
        self,
        input: torch.LongTensor,
        positions: Optional[torch.LongTensor] = None,
        segments: Optional[torch.LongTensor] = None,
        roles: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, torch.BoolTensor]:
        """
        Embed tokens prior to feeding into transformer.

        :param LongTensor[batch,seqlen] input:
            The input IDs
        :param LongTensor[batch,seqlen] positions:
            Positions for input IDs
        :param LongTensor[batch,seqlen]:
            If provided, additionally adds ``segments`` as extra embedding features.

        :return (tensor, mask):
            return embedded input and mask
        """
        mask = input != self.padding_idx
        if positions is None:
            positions = (mask.cumsum(dim=1, dtype=torch.int64) - 1).clamp_(min=0)
        tensor = self.embeddings(input)
        tensor = self.input_layer(tensor)
        if self.embeddings_scale:
            tensor = tensor * np.sqrt(self.dim)

        if positions.max().item() > self.n_positions:
            warn_once(
                'You are inputting a sequence of {x} length, but only have '
                '--n-positions {y}. Set --truncate or increase --n-positions'.format(
                    x=positions.max().item(), y=self.n_positions
                )
            )
        position_embs = self.position_embeddings(positions).expand_as(tensor)
        tensor = tensor + position_embs
        if segments is not None:
            tensor = tensor + self.turn_embeddings(segments)
        if roles is not None:
            tensor = tensor + self.role_embeddings(roles)

        return tensor, mask

    def forward_layers(
        self, tensor: torch.Tensor, mask: torch.BoolTensor
    ) -> torch.Tensor:
        """
        Apply transformer layers to input.

        :param tensor:
            embedded input
        :param mask:
            mask of input

        :return tensor:
            return embedding after applying transformer layers
        """
        if getattr(self.layers, 'is_model_parallel', False):
            # factored out for readability. It is equivalent to the other
            # condition
            tensor = self._apply_model_parallel(tensor, mask)
        else:
            for i in range(self.n_layers):
                tensor = self.layers[i](tensor, mask)

        return tensor

    def reduce_output(
        self, tensor: torch.Tensor, mask: torch.BoolTensor
    ) -> Tuple[torch.Tensor, Optional[torch.BoolTensor]]:
        """
        Reduce transformer output at end of forward pass.

        :param tensor:
            encoded input
        :param mask:
            mask for encoded input

        :return (tensor, mask):
            returns the reduced tensor, and mask if appropriate
        """
        tensor *= self.output_scaling
        if self.reduction_type == 'first':
            return tensor[:, 0, :], None
        elif self.reduction_type == 'max':
            return tensor.max(dim=1)[0], None
        elif self.reduction_type == 'mean':
            divisor = mask.float().sum(dim=1).unsqueeze(-1).clamp(min=1).type_as(tensor)
            output = tensor.sum(dim=1) / divisor
            return output, None
        elif self.reduction_type is None or 'none' in self.reduction_type:
            return tensor, mask
        else:
            raise ValueError(
                "Can't handle --reduction-type {}".format(self.reduction_type)
            )

    def forward(  # type: ignore
        self,
        input: torch.LongTensor,
        post_input: torch.LongTensor,
        reverse_post_input: torch.LongTensor,
        class_labels,
        positions: Optional[torch.LongTensor] = None,
        segments: Optional[torch.LongTensor] = None,
        roles: Optional[torch.LongTensor] = None,
        post_positions: Optional[torch.LongTensor] = None,
        post_segments: Optional[torch.LongTensor] = None,
        post_roles: Optional[torch.LongTensor] = None,
        reverse_post_positions: Optional[torch.LongTensor] = None,
        reverse_post_segments: Optional[torch.LongTensor] = None,
        reverse_post_roles: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.BoolTensor]]:
        """
        Forward pass.

        :param LongTensor[batch,seqlen] input:
            The input IDs
        :param LongTensor[batch,seqlen] positions:
            Positions for input IDs
        :param LongTensor[batch,seqlen] segments:
            If provided, additionally adds ``segments`` as extra embedding features.
        """
        # embed input
        tensor, mask = self.forward_embedding(input, positions, segments, roles)
        tensor = self.dropout(tensor)
        tensor *= mask.unsqueeze(-1).type_as(tensor)
        tensor = self.forward_layers(tensor, mask)

        prior_encode = self.w_encoder_attention(
            query=self.codes.unsqueeze(0).repeat(tensor.size(0), 1, 1),
            key=tensor, 
            value=tensor,
            mask=mask.unsqueeze(1)
        )[0]

        temp_list = []
        for i in range(attention_head_num):
            temp_list.append(prior_encode[:,i])
        prior_encode = torch.cat(temp_list, dim=1)
        prior_encode = self._linear_layer(prior_encode)

        ctr_info = None
        if self.opt["element"] == "D":
            ctr_info = self.control_embeddings(class_labels)
            prior_mu, prior_logvar = self.priorNet_mulogvar(torch.cat([prior_encode, ctr_info], dim=1))
        elif self.opt["element"] == "D+":
            ctr_info_specific = self.control_embeddings_specific(class_labels[0])
            ctr_info_symbol = self.control_embeddings_symbol(class_labels[1])
            ctr_info = [ctr_info_specific, ctr_info_symbol]
            prior_mu, prior_logvar = self.priorNet_mulogvar(torch.cat([prior_encode, ctr_info_specific, ctr_info_symbol], dim=1))
        else:
            prior_mu, prior_logvar = self.priorNet_mulogvar(prior_encode)

        mu = prior_mu.clone()
        logvar = prior_logvar.clone()

        if self.opt["datatype"] == "train":
            if post_input.size(1) > self.opt['text_truncate']:
                post_input = post_input[:, post_input.size(1) - self.opt['text_truncate']:]
                if post_segments is not None:
                    post_segments = post_segments[:, post_segments.size(1) - self.opt['text_truncate']:]
                if post_roles is not None:
                    post_roles = post_roles[:, post_roles.size(1) - self.opt['text_truncate']:]

            post_tensor, post_mask = self.forward_embedding(post_input, None, post_segments, post_roles)
            post_tensor = self.dropout(post_tensor)
            post_tensor *= post_mask.unsqueeze(-1).type_as(post_tensor)
            post_tensor = self.forward_layers(post_tensor, post_mask)

            post_encode = self.w_encoder_attention(
                query=self.codes.unsqueeze(0).repeat(post_tensor.size(0), 1, 1),
                key=post_tensor,
                value=post_tensor,
                mask=post_mask.unsqueeze(1)
            )[0]

            temp_list = []
            for i in range(attention_head_num):
                temp_list.append(post_encode[:,i])
            post_encode = torch.cat(temp_list, dim=1)
            post_encode = self._linear_layer(post_encode)

            recog_mu, recog_logvar = self.recogNet_mulogvar(post_encode)
            latent_sample = sample_gaussian(recog_mu, recog_logvar)
            kld = gaussian_kld(recog_mu, recog_logvar, prior_mu, prior_logvar)

            regular_item = None
            if self.opt["regular"] == True:
                ctr_info_random_length = self.control_embeddings_length(class_labels[2])
                b_prior_mu, b_prior_logvar = self.priorNet_mulogvar(torch.cat([prior_encode, ctr_info, ctr_info_random_length], dim=1))

                b_kld = gaussian_kld(recog_mu, recog_logvar, b_prior_mu, b_prior_logvar)

                gap = kld - b_kld
                bound = -0.25 * (torch.abs(class_labels[2] - class_labels[1]) // 2)
                regular_item = torch.where(
                    gap < bound,
                    # torch.FloatTensor(gap.size(0)).fill_(-0.5).cuda(),
                    bound,
                    gap
                ).float()

            mu = recog_mu.clone()
            logvar = recog_logvar.clone()
        else:
            latent_sample = sample_gaussian(prior_mu, prior_logvar)
            kld = None
            regular_item = None

        enc_out = torch.cat([prior_encode, latent_sample], 1)
        bow_logits = self.bow_project(enc_out)
        mmi = self.calc_mmi(mu, logvar, latent_sample)

        return tensor, mask, latent_sample, kld, regular_item, bow_logits, ctr_info, mmi 

    def log_sum_exp(self, value, dim=None, keepdim=False):
        """Numerically stable implementation of the operation
           value.exp().sum(dim, keepdim).log()
        """
        if dim is not None:
            m, _ = torch.max(value, dim=dim, keepdim=True)
            value0 = value - m
            if keepdim is False:
                m = m.squeeze(dim)
            return m + torch.log(torch.sum(torch.exp(value0), dim=dim, keepdim=keepdim))
        else:
            m = torch.max(value)
            sum_exp = torch.sum(torch.exp(value - m))
            return m + torch.log(sum_exp)

    def calc_mmi(self, mu, logvar, latent_sample):
        """Approximate the mutual information between x and z
        """
        batch_size, latent_size = mu.size()

        a0 = -0.5 * latent_size * math.log(2 * math.pi)
        a1 = -0.5 * (1 + logvar).sum(-1)
        neg_entropy = (a0 + a1).mean()

        latent_sample = latent_sample.unsqueeze(1)   # (batch_size, 1, latent_size)
        mu = mu.unsqueeze(0)          # (1, batch_size, latent_size)
        logvar = logvar.unsqueeze(0)  # (1, batch_size, latent_size)
        var = logvar.exp()

        dev = latent_sample - mu

        b0 = -0.5 * latent_size * math.log(2 * math.pi)
        b1 = -0.5 * logvar.sum(-1)
        b2 = -0.5 * ((dev ** 2) / var).sum(dim=-1)
        log_density = b0 + b1 + b2
        log_qz = self.log_sum_exp(log_density, dim=1) - math.log(batch_size)

        return (neg_entropy - log_qz.mean(-1)).item()

    def _apply_model_parallel(self, tensor, mask):
        """
        Pipeline application of model parallelism.
        """
        chunks = PipelineHelper.split((tensor, mask))
        work_items = PipelineHelper.schedule_work_items(self.layers, chunks)

        for chunk_idx, layer_nos, next_device in work_items:
            s_tensor, s_mask = chunks[chunk_idx]
            for layer_no in layer_nos:
                s_tensor = self.layers[layer_no](s_tensor, s_mask)
            chunks[chunk_idx] = PipelineHelper.chunk_to((s_tensor, s_mask), next_device)

        tensor_out, mask_out = PipelineHelper.join(chunks)
        return tensor_out
