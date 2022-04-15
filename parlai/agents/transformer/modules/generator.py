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

from parlai.agents.transformer.modules import (
    create_embeddings,
    TransformerDecoder,
    TransformerEncoder,
)
from parlai.agents.transformer.modules.modular import swappable
from parlai.core.opt import Opt
from parlai.core.torch_agent import DictionaryAgent
from parlai.core.torch_generator_agent import TorchGeneratorModel
from parlai.utils.torch import neginf
from parlai.utils.io import PathManager


@swappable(encoder=TransformerEncoder, decoder=TransformerDecoder)
class TransformerGeneratorModel(TorchGeneratorModel):
    """
    Implements a full generator model, with one encoder and one decoder.
    """

    @classmethod
    def build_encoder(
        cls,
        opt,
        dictionary,
        embedding=None,
        padding_idx=None,
        reduction_type='mean',
        encoder_class: Type[TransformerEncoder] = TransformerEncoder,
        **kwargs,
    ) -> TransformerEncoder:
        return encoder_class(
            opt=opt,
            embedding=embedding,
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
        decoder_class: Type[TransformerDecoder] = TransformerDecoder,
        **kwargs,
    ) -> TransformerDecoder:
        return decoder_class(opt=opt, embedding=embedding, **kwargs)

    def __init__(self, opt: Opt, dictionary: DictionaryAgent, **kwargs):
        self.pad_idx = dictionary[dictionary.null_token]
        self.start_idx = dictionary[dictionary.start_token]
        self.end_idx = dictionary[dictionary.end_token]
        super().__init__(self.pad_idx, self.start_idx, self.end_idx, **kwargs)
        self.opt = opt
        self.embeddings = create_embeddings(
            dictionary, opt['embedding_size'], self.pad_idx
        )

        import parlai.utils.pickle
        with PathManager.open("/root/yanghh/checkpoint/blender90M/pretrain/model", 'rb') as f:
            states = torch.load(f, map_location=lambda cpu, _: cpu, pickle_module=parlai.utils.pickle)
        self.embeddings.weight.data.copy_(states['model']['encoder.embeddings.weight'].data)

        self.encoder = self.build_encoder(
            opt,
            dictionary,
            self.embeddings,
            self.pad_idx,
            reduction_type=None,
            encoder_class=self.swappables.encoder,  # type: ignore
        )
        self.decoder = self.build_decoder(
            opt,
            embedding=self.embeddings,
            decoder_class=self.swappables.decoder,  # type: ignore
        )

        self.encoder.position_embeddings.weight.data.copy_(states['model']['encoder.position_embeddings.weight'].data)
        self.encoder.norm_embeddings.weight.data.copy_(states['model']['encoder.norm_embeddings.weight'].data)
        self.decoder.norm_embeddings.weight.data.copy_(states['model']['decoder.norm_embeddings.weight'].data)
        self.encoder.layers[0].attention.q_lin.weight.data.copy_(states['model']['encoder.layers.0.attention.q_lin.weight'].data)
        self.encoder.layers[0].attention.k_lin.weight.data.copy_(states['model']['encoder.layers.0.attention.k_lin.weight'].data)
        self.encoder.layers[0].attention.v_lin.weight.data.copy_(states['model']['encoder.layers.0.attention.v_lin.weight'].data)
        self.encoder.layers[0].attention.out_lin.weight.data.copy_(states['model']['encoder.layers.0.attention.out_lin.weight'].data)
        self.encoder.layers[0].norm1.weight.data.copy_(states['model']['encoder.layers.0.norm1.weight'].data)
        self.encoder.layers[0].norm2.weight.data.copy_(states['model']['encoder.layers.0.norm2.weight'].data)

        self.encoder.layers[1].attention.q_lin.weight.data.copy_(states['model']['encoder.layers.1.attention.q_lin.weight'].data)
        self.encoder.layers[1].attention.k_lin.weight.data.copy_(states['model']['encoder.layers.1.attention.k_lin.weight'].data)
        self.encoder.layers[1].attention.v_lin.weight.data.copy_(states['model']['encoder.layers.1.attention.v_lin.weight'].data)
        self.encoder.layers[1].attention.out_lin.weight.data.copy_(states['model']['encoder.layers.1.attention.out_lin.weight'].data)
        self.encoder.layers[1].norm1.weight.data.copy_(states['model']['encoder.layers.1.norm1.weight'].data)
        self.encoder.layers[1].norm2.weight.data.copy_(states['model']['encoder.layers.1.norm2.weight'].data)

        self.encoder.layers[2].attention.q_lin.weight.data.copy_(states['model']['encoder.layers.2.attention.q_lin.weight'].data)
        self.encoder.layers[2].attention.k_lin.weight.data.copy_(states['model']['encoder.layers.2.attention.k_lin.weight'].data)
        self.encoder.layers[2].attention.v_lin.weight.data.copy_(states['model']['encoder.layers.2.attention.v_lin.weight'].data)
        self.encoder.layers[2].attention.out_lin.weight.data.copy_(states['model']['encoder.layers.2.attention.out_lin.weight'].data)
        self.encoder.layers[2].norm1.weight.data.copy_(states['model']['encoder.layers.2.norm1.weight'].data)
        self.encoder.layers[2].norm2.weight.data.copy_(states['model']['encoder.layers.2.norm2.weight'].data)

        self.encoder.layers[3].attention.q_lin.weight.data.copy_(states['model']['encoder.layers.3.attention.q_lin.weight'].data)
        self.encoder.layers[3].attention.k_lin.weight.data.copy_(states['model']['encoder.layers.3.attention.k_lin.weight'].data)
        self.encoder.layers[3].attention.v_lin.weight.data.copy_(states['model']['encoder.layers.3.attention.v_lin.weight'].data)
        self.encoder.layers[3].attention.out_lin.weight.data.copy_(states['model']['encoder.layers.3.attention.out_lin.weight'].data)
        self.encoder.layers[3].norm1.weight.data.copy_(states['model']['encoder.layers.3.norm1.weight'].data)
        self.encoder.layers[3].norm2.weight.data.copy_(states['model']['encoder.layers.3.norm2.weight'].data)

        self.encoder.layers[4].attention.q_lin.weight.data.copy_(states['model']['encoder.layers.4.attention.q_lin.weight'].data)
        self.encoder.layers[4].attention.k_lin.weight.data.copy_(states['model']['encoder.layers.4.attention.k_lin.weight'].data)
        self.encoder.layers[4].attention.v_lin.weight.data.copy_(states['model']['encoder.layers.4.attention.v_lin.weight'].data)
        self.encoder.layers[4].attention.out_lin.weight.data.copy_(states['model']['encoder.layers.4.attention.out_lin.weight'].data)
        self.encoder.layers[4].norm1.weight.data.copy_(states['model']['encoder.layers.4.norm1.weight'].data)
        self.encoder.layers[4].norm2.weight.data.copy_(states['model']['encoder.layers.4.norm2.weight'].data)

        self.encoder.layers[5].attention.q_lin.weight.data.copy_(states['model']['encoder.layers.5.attention.q_lin.weight'].data)
        self.encoder.layers[5].attention.k_lin.weight.data.copy_(states['model']['encoder.layers.5.attention.k_lin.weight'].data)
        self.encoder.layers[5].attention.v_lin.weight.data.copy_(states['model']['encoder.layers.5.attention.v_lin.weight'].data)
        self.encoder.layers[5].attention.out_lin.weight.data.copy_(states['model']['encoder.layers.5.attention.out_lin.weight'].data)
        self.encoder.layers[5].norm1.weight.data.copy_(states['model']['encoder.layers.5.norm1.weight'].data)
        self.encoder.layers[5].norm2.weight.data.copy_(states['model']['encoder.layers.5.norm2.weight'].data)

        self.encoder.layers[0].ffn.lin1.weight.data.copy_(states['model']['encoder.layers.0.ffn.lin1.weight'].data)
        self.encoder.layers[0].ffn.lin2.weight.data.copy_(states['model']['encoder.layers.0.ffn.lin2.weight'].data)
        self.encoder.layers[1].ffn.lin1.weight.data.copy_(states['model']['encoder.layers.1.ffn.lin1.weight'].data)
        self.encoder.layers[1].ffn.lin2.weight.data.copy_(states['model']['encoder.layers.1.ffn.lin2.weight'].data)
        self.encoder.layers[2].ffn.lin1.weight.data.copy_(states['model']['encoder.layers.2.ffn.lin1.weight'].data)
        self.encoder.layers[2].ffn.lin2.weight.data.copy_(states['model']['encoder.layers.2.ffn.lin2.weight'].data)
        self.encoder.layers[3].ffn.lin1.weight.data.copy_(states['model']['encoder.layers.3.ffn.lin1.weight'].data)
        self.encoder.layers[3].ffn.lin2.weight.data.copy_(states['model']['encoder.layers.3.ffn.lin2.weight'].data)
        self.encoder.layers[4].ffn.lin1.weight.data.copy_(states['model']['encoder.layers.4.ffn.lin1.weight'].data)
        self.encoder.layers[4].ffn.lin2.weight.data.copy_(states['model']['encoder.layers.4.ffn.lin2.weight'].data)
        self.encoder.layers[5].ffn.lin1.weight.data.copy_(states['model']['encoder.layers.5.ffn.lin1.weight'].data)
        self.encoder.layers[5].ffn.lin2.weight.data.copy_(states['model']['encoder.layers.5.ffn.lin2.weight'].data)

        self.decoder.layers[0].ffn.lin1.weight.data.copy_(states['model']['decoder.layers.0.ffn.lin1.weight'].data)
        self.decoder.layers[0].ffn.lin2.weight.data.copy_(states['model']['decoder.layers.0.ffn.lin2.weight'].data)
        self.decoder.layers[1].ffn.lin1.weight.data.copy_(states['model']['decoder.layers.1.ffn.lin1.weight'].data)
        self.decoder.layers[1].ffn.lin2.weight.data.copy_(states['model']['decoder.layers.1.ffn.lin2.weight'].data)
        self.decoder.layers[2].ffn.lin1.weight.data.copy_(states['model']['decoder.layers.2.ffn.lin1.weight'].data)
        self.decoder.layers[2].ffn.lin2.weight.data.copy_(states['model']['decoder.layers.2.ffn.lin2.weight'].data)
        self.decoder.layers[3].ffn.lin1.weight.data.copy_(states['model']['decoder.layers.3.ffn.lin1.weight'].data)
        self.decoder.layers[3].ffn.lin2.weight.data.copy_(states['model']['decoder.layers.3.ffn.lin2.weight'].data)
        self.decoder.layers[4].ffn.lin1.weight.data.copy_(states['model']['decoder.layers.4.ffn.lin1.weight'].data)
        self.decoder.layers[4].ffn.lin2.weight.data.copy_(states['model']['decoder.layers.4.ffn.lin2.weight'].data)
        self.decoder.layers[5].ffn.lin1.weight.data.copy_(states['model']['decoder.layers.5.ffn.lin1.weight'].data)
        self.decoder.layers[5].ffn.lin2.weight.data.copy_(states['model']['decoder.layers.5.ffn.lin2.weight'].data)
        self.decoder.layers[6].ffn.lin1.weight.data.copy_(states['model']['decoder.layers.6.ffn.lin1.weight'].data)
        self.decoder.layers[6].ffn.lin2.weight.data.copy_(states['model']['decoder.layers.6.ffn.lin2.weight'].data)
        self.decoder.layers[7].ffn.lin1.weight.data.copy_(states['model']['decoder.layers.7.ffn.lin1.weight'].data)
        self.decoder.layers[7].ffn.lin2.weight.data.copy_(states['model']['decoder.layers.7.ffn.lin2.weight'].data)

        self.decoder.layers[0].self_attention.q_lin.weight.data.copy_(states['model']['decoder.layers.0.self_attention.q_lin.weight'].data)
        self.decoder.layers[0].self_attention.k_lin.weight.data.copy_(states['model']['decoder.layers.0.self_attention.k_lin.weight'].data)
        self.decoder.layers[0].self_attention.v_lin.weight.data.copy_(states['model']['decoder.layers.0.self_attention.v_lin.weight'].data)
        self.decoder.layers[0].self_attention.out_lin.weight.data.copy_(states['model']['decoder.layers.0.self_attention.out_lin.weight'].data)
        self.decoder.layers[0].norm1.weight.data.copy_(states['model']['decoder.layers.0.norm1.weight'].data)
        self.decoder.layers[0].norm2.weight.data.copy_(states['model']['decoder.layers.0.norm2.weight'].data)
        self.decoder.layers[0].norm3.weight.data.copy_(states['model']['decoder.layers.0.norm3.weight'].data)

        self.decoder.layers[1].self_attention.q_lin.weight.data.copy_(states['model']['decoder.layers.1.self_attention.q_lin.weight'].data)
        self.decoder.layers[1].self_attention.k_lin.weight.data.copy_(states['model']['decoder.layers.1.self_attention.k_lin.weight'].data)
        self.decoder.layers[1].self_attention.v_lin.weight.data.copy_(states['model']['decoder.layers.1.self_attention.v_lin.weight'].data)
        self.decoder.layers[1].self_attention.out_lin.weight.data.copy_(states['model']['decoder.layers.1.self_attention.out_lin.weight'].data)
        self.decoder.layers[1].norm1.weight.data.copy_(states['model']['decoder.layers.1.norm1.weight'].data)
        self.decoder.layers[1].norm2.weight.data.copy_(states['model']['decoder.layers.1.norm2.weight'].data)
        self.decoder.layers[1].norm3.weight.data.copy_(states['model']['decoder.layers.1.norm3.weight'].data)

        self.decoder.layers[2].self_attention.q_lin.weight.data.copy_(states['model']['decoder.layers.2.self_attention.q_lin.weight'].data)
        self.decoder.layers[2].self_attention.k_lin.weight.data.copy_(states['model']['decoder.layers.2.self_attention.k_lin.weight'].data)
        self.decoder.layers[2].self_attention.v_lin.weight.data.copy_(states['model']['decoder.layers.2.self_attention.v_lin.weight'].data)
        self.decoder.layers[2].self_attention.out_lin.weight.data.copy_(states['model']['decoder.layers.2.self_attention.out_lin.weight'].data)
        self.decoder.layers[2].norm1.weight.data.copy_(states['model']['decoder.layers.2.norm1.weight'].data)
        self.decoder.layers[2].norm2.weight.data.copy_(states['model']['decoder.layers.2.norm2.weight'].data)
        self.decoder.layers[2].norm3.weight.data.copy_(states['model']['decoder.layers.2.norm3.weight'].data)

        self.decoder.layers[3].self_attention.q_lin.weight.data.copy_(states['model']['decoder.layers.3.self_attention.q_lin.weight'].data)
        self.decoder.layers[3].self_attention.k_lin.weight.data.copy_(states['model']['decoder.layers.3.self_attention.k_lin.weight'].data)
        self.decoder.layers[3].self_attention.v_lin.weight.data.copy_(states['model']['decoder.layers.3.self_attention.v_lin.weight'].data)
        self.decoder.layers[3].self_attention.out_lin.weight.data.copy_(states['model']['decoder.layers.3.self_attention.out_lin.weight'].data)
        self.decoder.layers[3].norm1.weight.data.copy_(states['model']['decoder.layers.3.norm1.weight'].data)
        self.decoder.layers[3].norm2.weight.data.copy_(states['model']['decoder.layers.3.norm2.weight'].data)
        self.decoder.layers[3].norm3.weight.data.copy_(states['model']['decoder.layers.3.norm3.weight'].data)

        self.decoder.layers[4].self_attention.q_lin.weight.data.copy_(states['model']['decoder.layers.4.self_attention.q_lin.weight'].data)
        self.decoder.layers[4].self_attention.k_lin.weight.data.copy_(states['model']['decoder.layers.4.self_attention.k_lin.weight'].data)
        self.decoder.layers[4].self_attention.v_lin.weight.data.copy_(states['model']['decoder.layers.4.self_attention.v_lin.weight'].data)
        self.decoder.layers[4].self_attention.out_lin.weight.data.copy_(states['model']['decoder.layers.4.self_attention.out_lin.weight'].data)
        self.decoder.layers[4].norm1.weight.data.copy_(states['model']['decoder.layers.4.norm1.weight'].data)
        self.decoder.layers[4].norm2.weight.data.copy_(states['model']['decoder.layers.4.norm2.weight'].data)
        self.decoder.layers[4].norm3.weight.data.copy_(states['model']['decoder.layers.4.norm3.weight'].data)
        self.decoder.layers[5].norm1.weight.data.copy_(states['model']['decoder.layers.5.norm1.weight'].data)
        self.decoder.layers[5].norm2.weight.data.copy_(states['model']['decoder.layers.5.norm2.weight'].data)
        self.decoder.layers[5].norm3.weight.data.copy_(states['model']['decoder.layers.5.norm3.weight'].data)
        self.decoder.layers[6].norm1.weight.data.copy_(states['model']['decoder.layers.6.norm1.weight'].data)
        self.decoder.layers[6].norm2.weight.data.copy_(states['model']['decoder.layers.6.norm2.weight'].data)
        self.decoder.layers[6].norm3.weight.data.copy_(states['model']['decoder.layers.6.norm3.weight'].data)
        self.decoder.layers[7].norm1.weight.data.copy_(states['model']['decoder.layers.7.norm1.weight'].data)
        self.decoder.layers[7].norm2.weight.data.copy_(states['model']['decoder.layers.7.norm2.weight'].data)
        self.decoder.layers[7].norm3.weight.data.copy_(states['model']['decoder.layers.7.norm3.weight'].data)

        self.decoder.layers[5].self_attention.q_lin.weight.data.copy_(states['model']['decoder.layers.5.self_attention.q_lin.weight'].data)
        self.decoder.layers[5].self_attention.k_lin.weight.data.copy_(states['model']['decoder.layers.5.self_attention.k_lin.weight'].data)
        self.decoder.layers[5].self_attention.v_lin.weight.data.copy_(states['model']['decoder.layers.5.self_attention.v_lin.weight'].data)
        self.decoder.layers[5].self_attention.out_lin.weight.data.copy_(states['model']['decoder.layers.5.self_attention.out_lin.weight'].data)

        self.decoder.layers[6].self_attention.q_lin.weight.data.copy_(states['model']['decoder.layers.6.self_attention.q_lin.weight'].data)
        self.decoder.layers[6].self_attention.k_lin.weight.data.copy_(states['model']['decoder.layers.6.self_attention.k_lin.weight'].data)
        self.decoder.layers[6].self_attention.v_lin.weight.data.copy_(states['model']['decoder.layers.6.self_attention.v_lin.weight'].data)
        self.decoder.layers[6].self_attention.out_lin.weight.data.copy_(states['model']['decoder.layers.6.self_attention.out_lin.weight'].data)

        self.decoder.layers[7].self_attention.q_lin.weight.data.copy_(states['model']['decoder.layers.7.self_attention.q_lin.weight'].data)
        self.decoder.layers[7].self_attention.k_lin.weight.data.copy_(states['model']['decoder.layers.7.self_attention.k_lin.weight'].data)
        self.decoder.layers[7].self_attention.v_lin.weight.data.copy_(states['model']['decoder.layers.7.self_attention.v_lin.weight'].data)
        self.decoder.layers[7].self_attention.out_lin.weight.data.copy_(states['model']['decoder.layers.7.self_attention.out_lin.weight'].data)

        self.decoder.layers[0].encoder_attention.q_lin.weight.data.copy_(states['model']['decoder.layers.0.encoder_attention.q_lin.weight'].data)
        self.decoder.layers[0].encoder_attention.k_lin.weight.data.copy_(states['model']['decoder.layers.0.encoder_attention.k_lin.weight'].data)
        self.decoder.layers[0].encoder_attention.v_lin.weight.data.copy_(states['model']['decoder.layers.0.encoder_attention.v_lin.weight'].data)
        self.decoder.layers[0].encoder_attention.out_lin.weight.data.copy_(states['model']['decoder.layers.0.encoder_attention.out_lin.weight'].data)

        self.decoder.layers[1].encoder_attention.q_lin.weight.data.copy_(states['model']['decoder.layers.1.encoder_attention.q_lin.weight'].data)
        self.decoder.layers[1].encoder_attention.k_lin.weight.data.copy_(states['model']['decoder.layers.1.encoder_attention.k_lin.weight'].data)
        self.decoder.layers[1].encoder_attention.v_lin.weight.data.copy_(states['model']['decoder.layers.1.encoder_attention.v_lin.weight'].data)
        self.decoder.layers[1].encoder_attention.out_lin.weight.data.copy_(states['model']['decoder.layers.1.encoder_attention.out_lin.weight'].data)

        self.decoder.layers[2].encoder_attention.q_lin.weight.data.copy_(states['model']['decoder.layers.2.encoder_attention.q_lin.weight'].data)
        self.decoder.layers[2].encoder_attention.k_lin.weight.data.copy_(states['model']['decoder.layers.2.encoder_attention.k_lin.weight'].data)
        self.decoder.layers[2].encoder_attention.v_lin.weight.data.copy_(states['model']['decoder.layers.2.encoder_attention.v_lin.weight'].data)
        self.decoder.layers[2].encoder_attention.out_lin.weight.data.copy_(states['model']['decoder.layers.2.encoder_attention.out_lin.weight'].data)

        self.decoder.layers[3].encoder_attention.q_lin.weight.data.copy_(states['model']['decoder.layers.3.encoder_attention.q_lin.weight'].data)
        self.decoder.layers[3].encoder_attention.k_lin.weight.data.copy_(states['model']['decoder.layers.3.encoder_attention.k_lin.weight'].data)
        self.decoder.layers[3].encoder_attention.v_lin.weight.data.copy_(states['model']['decoder.layers.3.encoder_attention.v_lin.weight'].data)
        self.decoder.layers[3].encoder_attention.out_lin.weight.data.copy_(states['model']['decoder.layers.3.encoder_attention.out_lin.weight'].data)

        self.decoder.layers[4].encoder_attention.q_lin.weight.data.copy_(states['model']['decoder.layers.4.encoder_attention.q_lin.weight'].data)
        self.decoder.layers[4].encoder_attention.k_lin.weight.data.copy_(states['model']['decoder.layers.4.encoder_attention.k_lin.weight'].data)
        self.decoder.layers[4].encoder_attention.v_lin.weight.data.copy_(states['model']['decoder.layers.4.encoder_attention.v_lin.weight'].data)
        self.decoder.layers[4].encoder_attention.out_lin.weight.data.copy_(states['model']['decoder.layers.4.encoder_attention.out_lin.weight'].data)

        self.decoder.layers[5].encoder_attention.q_lin.weight.data.copy_(states['model']['decoder.layers.5.encoder_attention.q_lin.weight'].data)
        self.decoder.layers[5].encoder_attention.k_lin.weight.data.copy_(states['model']['decoder.layers.5.encoder_attention.k_lin.weight'].data)
        self.decoder.layers[5].encoder_attention.v_lin.weight.data.copy_(states['model']['decoder.layers.5.encoder_attention.v_lin.weight'].data)
        self.decoder.layers[5].encoder_attention.out_lin.weight.data.copy_(states['model']['decoder.layers.5.encoder_attention.out_lin.weight'].data)

        self.decoder.layers[6].encoder_attention.q_lin.weight.data.copy_(states['model']['decoder.layers.6.encoder_attention.q_lin.weight'].data)
        self.decoder.layers[6].encoder_attention.k_lin.weight.data.copy_(states['model']['decoder.layers.6.encoder_attention.k_lin.weight'].data)
        self.decoder.layers[6].encoder_attention.v_lin.weight.data.copy_(states['model']['decoder.layers.6.encoder_attention.v_lin.weight'].data)
        self.decoder.layers[6].encoder_attention.out_lin.weight.data.copy_(states['model']['decoder.layers.6.encoder_attention.out_lin.weight'].data)

        self.decoder.layers[7].encoder_attention.q_lin.weight.data.copy_(states['model']['decoder.layers.7.encoder_attention.q_lin.weight'].data)
        self.decoder.layers[7].encoder_attention.k_lin.weight.data.copy_(states['model']['decoder.layers.7.encoder_attention.k_lin.weight'].data)
        self.decoder.layers[7].encoder_attention.v_lin.weight.data.copy_(states['model']['decoder.layers.7.encoder_attention.v_lin.weight'].data)
        self.decoder.layers[7].encoder_attention.out_lin.weight.data.copy_(states['model']['decoder.layers.7.encoder_attention.out_lin.weight'].data)

    def reorder_encoder_states(self, encoder_states, indices):
        """
        Reorder the encoder states.

        See ``TorchGeneratorModel.reorder_encoder_states`` for a description.
        """
        enc, mask = encoder_states
        if not torch.is_tensor(indices):
            indices = torch.LongTensor(indices).to(enc.device)
        enc = torch.index_select(enc, 0, indices)
        mask = torch.index_select(mask, 0, indices)
        return enc, mask

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
