#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from parlai.utils.torch import neginf
from functools import lru_cache
from parlai.agents.seq2seq.modules import (
    OutputLayer,
    RNNEncoder,
    _transpose_hidden_state,
)
from parlai.core.torch_generator_agent import TorchGeneratorModel
import math


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
            nn.Linear(prior_input_size, np.maximum(latent_size * 2, 100)),
            nn.Tanh(),
            nn.Linear(np.maximum(latent_size * 2, 100), latent_size * 2)
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


class CvaeEncoder(RNNEncoder):
    """
    RNN Encoder.

    Modified to encode history vector in context lstm.
    """

    def __init__(
        self,
        num_features,
        embeddingsize,
        hiddensize,
        latentsize, 
        device,
        padding_idx=0,
        rnn_class="lstm",
        numlayers=2,
        dropout=0.1,
        bidirectional=True,
        shared_lt=None,
        shared_rnn=None,
        input_dropout=0,
        unknown_idx=None,
        sparse=False,
    ):
        """
        Initialize recurrent encoder and context gru.
        """
        super().__init__(
            num_features,
            embeddingsize,
            hiddensize,
            padding_idx,
            rnn_class,
            numlayers,
            dropout,
            bidirectional,
            shared_lt,
            shared_rnn,
            input_dropout,
            unknown_idx,
            sparse,
        )
        self.padding_idx = padding_idx
        self.context_lstm = nn.LSTM(hiddensize, hiddensize, numlayers, batch_first=True).to(device)
        self.priorNet_mulogvar = PriorNetwork(hiddensize, latentsize).to(device)
        self.recogNet_mulogvar = RecognitionNetwork(hiddensize + hiddensize, latentsize).to(device)
        self.bow_project = BowProject(hiddensize + latentsize, num_features).to(device)

    def context_encode(self, xs, context_vec, hist_lens):
        # encode current utterrance
        (enc_state, (hidden_state, cell_state), attn_mask) = super().forward(xs)

        # if all utterances in context vec length 1, unsqueeze to prevent loss of dimensionality
        if len(context_vec.shape) < 2:
            context_vec = context_vec.unsqueeze(1)
        # get utt lengths of each utt in context vector
        utt_lens = torch.sum(context_vec.ne(0).int(), dim=1)
        # sort by lengths descending for utterance encoder
        sorted_lens, sorted_idx = utt_lens.sort(descending=True)
        sorted_context_vec = context_vec[sorted_idx]
        (_, (sorted_hidden_state, _), _) = super().forward(sorted_context_vec)
        sorted_final_hidden_states = sorted_hidden_state[:, -1, :]

        ### reshape and pad hidden states to bsz x max_hist_len x hidden_size using hist_lens
        original_order_final_hidden = torch.zeros_like(
            sorted_final_hidden_states
        ).scatter_(
            0,
            sorted_idx.unsqueeze(1).expand(-1, sorted_final_hidden_states.shape[1]),
            sorted_final_hidden_states,
        )

        # pad to max hist_len
        original_size_final_hidden = self.sequence_to_padding(
            original_order_final_hidden, hist_lens
        )

        # pack padded sequence so that we ignore padding
        original_size_final_hidden_packed = nn.utils.rnn.pack_padded_sequence(
            original_size_final_hidden,
            hist_lens.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )

        # pass through context lstm
        _, (context_h_n, _) = self.context_lstm(original_size_final_hidden_packed)

        return (
            enc_state,
            (hidden_state, cell_state),
            attn_mask,
            _transpose_hidden_state(context_h_n),
        )

    def target_encode(self, ys):
        # encode target utterrance as part of recognition network.
        _ys = ys.clone()
        _ys_len = _ys.ne(0).sum(dim=1)
        sorted_len, len_ix = _ys_len.sort(0, descending=True)

        inv_ix = len_ix.clone()
        inv_ix[len_ix] = torch.arange(0, len(len_ix)).type_as(inv_ix)

        _ys_sorted = _ys[len_ix].contiguous()

        enc_state, (hidden_state, cell_state), attn_mask = super().forward(_ys_sorted)

        enc_state = enc_state[inv_ix].contiguous()
        hidden_state = hidden_state[inv_ix].contiguous()
        attn_mask = attn_mask[inv_ix].contiguous()

        return (
            enc_state,
            (hidden_state, cell_state),
            attn_mask,
            None,
        )

    def sequence_to_padding(self, x, lengths):
        """
        Return padded and reshaped sequence (x) according to tensor lengths
        Example:
            x = tensor([[1, 2], [2, 3], [4, 0], [5, 6], [7, 8], [9, 10]])
            lengths = tensor([1, 2, 2, 1])
        Would output:
            tensor([[[1, 2], [0, 0]],
                    [[2, 3], [4, 0]],
                    [[5, 6], [7, 8]],
                    [[9, 10], [0, 0]]])
        """
        ret_tensor = torch.zeros(
            (lengths.shape[0], torch.max(lengths).int()) + tuple(x.shape[1:])
        ).to(x.device)
        cum_len = 0
        for i, l in enumerate(lengths):
            ret_tensor[i, :l] = x[cum_len : cum_len + l]
            cum_len += l
        return ret_tensor

    def forward(self, xs, ys, context_vec, hist_lens, mode):
        _, _, _, ctx_hidden = self.context_encode(xs, context_vec, hist_lens)

        prior_input = ctx_hidden[:, -1, :]
        prior_mu, prior_logvar = self.priorNet_mulogvar(prior_input)
        mu = prior_mu
        logvar = prior_logvar

        if mode == "train":
            trg_enc_state, (trg_hidden_state, _), trg_attn_mask, _ = self.target_encode(ys)
            recog_input = torch.cat([prior_input, trg_hidden_state[:, -1, :]], 1)
            recog_mu, recog_logvar = self.recogNet_mulogvar(recog_input)
            mu = recog_mu
            logvar = recog_logvar
            latent_sample = sample_gaussian(recog_mu, recog_logvar)
            kld = gaussian_kld(recog_mu, recog_logvar, prior_mu, prior_logvar)
        else:
            latent_sample = sample_gaussian(prior_mu, prior_logvar)
            kld = None

        # for pretrain
        # latent_sample = torch.zeros_like(latent_sample)

        enc_out = torch.cat([prior_input, latent_sample], 1)
        bow_logits = self.bow_project(enc_out)

        mmi = self.calc_mmi(mu, logvar, latent_sample)

        return enc_out, bow_logits, kld, mmi

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


class CvaeDecoder(nn.Module):
    """
    Recurrent decoder module that uses dialog history encoded by context gru.
    """

    def __init__(
        self,
        num_features,
        embeddingsize,
        hiddensize,
        latentsize,
        padding_idx=0,
        rnn_class="lstm",
        numlayers=2,
        dropout=0.1,
        bidir_input=False,
        attn_length=-1,
        sparse=False,
    ):
        """
        Initialize recurrent decoder.
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.layers = numlayers
        self.hsz = hiddensize
        self.esz = embeddingsize

        self.lt = nn.Embedding(
            num_features, embeddingsize, padding_idx=padding_idx, sparse=sparse
        )

        self.rnn = rnn_class(
            embeddingsize,
            hiddensize,
            numlayers,
            dropout=dropout if numlayers > 1 else 0,
            batch_first=True,
        )
        self.dec_init_state_net = nn.Linear(hiddensize + latentsize, hiddensize)


    def forward(self, xs, encoder_output, incremental_state=None):
        """
        Decode from input tokens.

        :param xs: (bsz x seqlen) LongTensor of input token indices
        :param encoder_output: output from HredEncoder. Tuple containing
            (enc_out, enc_hidden, attn_mask, context_hidden) tuple.
        :param incremental_state: most recent hidden state to the decoder.
        :returns: (output, hidden_state) pair from the RNN.
            - output is a bsz x time x latentdim matrix. This value must be passed to
                the model's OutputLayer for a final softmax.
            - hidden_state depends on the choice of RNN
        """
        enc_out, _, _, _ = encoder_output

        if incremental_state is not None:
            # we're doing it piece by piece, so we have a more important hidden
            # seed, and we only need to compute for the final timestep
            hidden = _transpose_hidden_state(incremental_state)
            # only need the last timestep then
            xs = xs[:, -1:]
        else:
            # starting fresh, or generating from scratch. Use the encoder hidden
            # state as our start state
            hidden = self.dec_init_state_net(enc_out).unsqueeze(1).repeat(1, self.layers, 1)
            hidden = _transpose_hidden_state(hidden)
            cell = torch.zeros_like(hidden)
            hidden = (hidden, cell)

        if isinstance(hidden, tuple):
            hidden = tuple(x.contiguous() for x in hidden)
        else:
            hidden = hidden.contiguous()

        # sequence indices => sequence embeddings
        seqlen = xs.size(1)
        xes = self.dropout(self.lt(xs))

        # run through rnn with None as initial decoder state
        # source for zeroes hidden state: http://www.cs.toronto.edu/~lcharlin/papers/vhred_aaai17.pdf
        output, new_hidden = self.rnn(xes, hidden)

        return output, _transpose_hidden_state(new_hidden)


class CvaeModel(TorchGeneratorModel):
    """
    cvae model.
    """

    def __init__(
        self,
        num_features,
        embeddingsize,
        hiddensize,
        latentsize,
        device,
        numlayers=2,
        dropout=0,
        bidirectional=True,
        rnn_class="lstm",
        lookuptable="unique",
        decoder="same",
        numsoftmax=1,
        attention="none",
        attention_length=48,
        attention_time="post",
        padding_idx=0,
        start_idx=1,
        end_idx=2,
        unknown_idx=3,
        input_dropout=0,
        longest_label=1,
    ):

        super().__init__(
            padding_idx=padding_idx,
            start_idx=start_idx,
            end_idx=end_idx,
            unknown_idx=unknown_idx,
            input_dropout=input_dropout,
            longest_label=longest_label,
        )

        rnn_class = nn.LSTM

        self.decoder = CvaeDecoder(
            num_features=num_features,
            embeddingsize=embeddingsize,
            hiddensize=hiddensize,
            latentsize=latentsize,
            dropout=dropout,
            rnn_class=rnn_class,
        )

        self.encoder = CvaeEncoder(
            num_features=num_features,
            embeddingsize=embeddingsize,
            hiddensize=hiddensize,
            latentsize=latentsize,
            shared_lt=self.decoder.lt,
            device=device,
            dropout=dropout,
            rnn_class=rnn_class,
        )

        self.output = OutputLayer(
            num_features=num_features,
            embeddingsize=embeddingsize,
            hiddensize=hiddensize,
            dropout=dropout,
            shared_weight=self.decoder.lt,
        )

    def reorder_encoder_states(self, encoder_states, indices):
        """
        Reorder encoder states according to a new set of indices.
        """
        enc_out, bow_logits, kld, mmi = encoder_states

        if not torch.is_tensor(indices):
            # cast indices to a tensor if needed
            indices = torch.LongTensor(indices).to(hid.device)

        enc_out = enc_out.index_select(0, indices)

        return enc_out, bow_logits, kld, mmi

    def reorder_decoder_incremental_state(self, incremental_state, inds):
        if torch.is_tensor(incremental_state):
            # gru or vanilla rnn
            return torch.index_select(incremental_state, 0, inds).contiguous()
        elif isinstance(incremental_state, tuple):
            return tuple(
                self.reorder_decoder_incremental_state(x, inds)
                for x in incremental_state
            )
