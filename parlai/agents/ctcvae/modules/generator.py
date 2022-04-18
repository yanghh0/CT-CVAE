#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Implements NN code for transformers.

Original paper: https://arxiv.org/abs/1706.03762. (Vaswani, 2017). The
`Annotated Transformer` (Rush, 2018) is an excellent reading guide which explains
much of the mechanics of the Transformer model
(http://nlp.seas.harvard.edu/2018/04/03/attention.html).

This module also supports special segments (ala BERT;
https://arxiv.org/abs/1810.04805), and a few different variations seen in the
literature (BERT and XLM; https://arxiv.org/abs/1901.07291).
"""

from __future__ import annotations
from typing import Dict, Type

import torch
import torch.cuda
import torch.nn.functional as F

from parlai.agents.ctcvae.modules import (
    create_embeddings,
    TransformerDecoder,
    TransformerEncoder,
)
from parlai.agents.ctcvae.modules.modular import swappable
from parlai.core.opt import Opt
from parlai.core.torch_agent import DictionaryAgent
from parlai.core.torch_generator_agent import TorchGeneratorModel
from parlai.utils.torch import neginf


@swappable(encoder=TransformerEncoder, decoder=TransformerDecoder)
class CtcvaeModel(TorchGeneratorModel):
    """
    Implements a full generator model, with one encoder and one decoder.
    """

    @classmethod
    def build_encoder(
        cls,
        opt,
        dictionary,
        embedding=None,
        turn_embedding=None,
        role_embedding=None,
        padding_idx=None,
        reduction_type='mean',
        encoder_class: Type[TransformerEncoder] = TransformerEncoder,
        **kwargs,
    ) -> TransformerEncoder:
        return encoder_class(
            opt=opt,
            embedding=embedding,
            turn_embedding=turn_embedding,
            role_embedding=role_embedding,
            vocabulary_size=len(dictionary),
            padding_idx=padding_idx,
            reduction_type=reduction_type,
            **kwargs,
        )

    @classmethod
    def build_decoder(
        cls,
        opt,
        embedding=None,
        turn_embedding=None,
        role_embedding=None,
        decoder_class: Type[TransformerDecoder] = TransformerDecoder,
        **kwargs,
    ) -> TransformerDecoder:
        return decoder_class(opt=opt, embedding=embedding, turn_embedding=turn_embedding, role_embedding=role_embedding, **kwargs)

    def __init__(self, opt: Opt, dictionary: DictionaryAgent, **kwargs):
        self.pad_idx = dictionary[dictionary.null_token]
        self.start_idx = dictionary[dictionary.start_token]
        self.end_idx = dictionary[dictionary.end_token]
        super().__init__(self.pad_idx, self.start_idx, self.end_idx, **kwargs)
        self.opt = opt
        self.embeddings = create_embeddings(
            dictionary, opt['embedding_size'], self.pad_idx
        )
        self.turn_embeddings = torch.nn.Embedding(50, opt['d_model'])
        self.role_embeddings = torch.nn.Embedding(2, opt['d_model'])

        self.encoder = self.build_encoder(
            opt,
            dictionary,
            self.embeddings,
            self.turn_embeddings,
            self.role_embeddings,
            self.pad_idx,
            reduction_type=None,
            encoder_class=self.swappables.encoder,  # type: ignore
        )
        self.decoder = self.build_decoder(
            opt,
            embedding=self.embeddings,
            turn_embedding=self.turn_embeddings,
            role_embedding=self.role_embeddings,
            decoder_class=self.swappables.decoder,  # type: ignore
        )

        self.encoder.input_layer.weight = self.decoder.input_layer.weight

    def reorder_encoder_states(self, encoder_states, indices):
        """
        Reorder the encoder states.

        See ``TorchGeneratorModel.reorder_encoder_states`` for a description.
        """
        enc, mask, latent_sample, kld, regular_item, bow_logits, ctr_info, mmi = encoder_states
        if not torch.is_tensor(indices):
            indices = torch.LongTensor(indices).to(enc.device)
        enc = torch.index_select(enc, 0, indices)
        mask = torch.index_select(mask, 0, indices)
        latent_sample = torch.index_select(latent_sample, 0, indices)
        if ctr_info is not None:
            if isinstance(ctr_info, torch.Tensor):
                ctr_info = torch.index_select(ctr_info, 0, indices)
            else:
                ctr_info = [torch.index_select(x, 0, indices) for x in ctr_info]
        return enc, mask, latent_sample, kld, regular_item, bow_logits, ctr_info, mmi

    def reorder_decoder_incremental_state(
        self, incremental_state: Dict[int, dict], inds: torch.Tensor
    ) -> Dict[int, dict]:
        """
        Reorder the decoder incremental state.

        See ``TorchGeneratorModel.reorder_decoder_incremental_state`` for a description.

        Here, incremental_state is a dict whose keys are layer indices and whose values
        are dicts containing the incremental state for that layer.
        """
        return {
            idx: layer.reorder_incremental_state(incremental_state[idx], inds)
            for idx, layer in enumerate(self.decoder.layers)
        }

    def output(self, tensor):
        """
        Compute output logits.
        """
        # project back to vocabulary
        output = F.linear(tensor, self.embeddings.weight)
        # compatibility with fairseq: fairseq sometimes reuses BOS tokens and
        # we need to force their probability of generation to be 0.
        output[:, :, self.start_idx] = neginf(output.dtype)
        return output
