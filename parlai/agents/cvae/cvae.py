#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from typing import Optional
from parlai.core.params import ParlaiParser
from parlai.core.opt import Opt
from parlai.core.torch_generator_agent import TorchGeneratorAgent, PPLMetric
from parlai.core.metrics import AverageMetric
from .modules import CvaeModel


class CvaeAgent(TorchGeneratorAgent):
    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        """
        Add command-line arguments specifically for this agent.
        """
        agent = parser.add_argument_group("CVAE Arguments")
        agent.add_argument(
            "-esz",
            "--embeddingsize",
            type=int,
            default=300,
            help="size of the token embeddings",
        )
        agent.add_argument(
            "-hs",
            "--hiddensize",
            type=int,
            default=512,
            help="size of the hidden layers",
        )
        agent.add_argument(
            "-lsz",
            "--latentsize",
            type=int,
            default=128,
            help="size of the latent size",
        )
        agent.add_argument(
            "-dr", 
            "--dropout", 
            type=float, 
            default=0.1, 
            help="dropout rate"
        )
        agent.add_argument(
            "--full-kl-step",
            type=int,
            default=500000,
            help="how many batch before KL cost weight reaches 1.0",
        )

        super().add_cmdline_args(parser, partial_opt=partial_opt)
        return agent

    def __init__(self, opt, shared=None):
        """
        Set up model.
        """
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")
        super().__init__(opt, shared)
        self.id = "CVAE"
        self.full_kl_step = opt["full_kl_step"]
        self.global_t = 0
        self.opt = opt

    def build_model(self, states=None):
        opt = self.opt
        if not states:
            states = {}

        model = CvaeModel(
            len(self.dict),
            opt["embeddingsize"],
            opt["hiddensize"],
            opt["latentsize"],
            dropout=opt["dropout"],
            device=self.device,
            padding_idx=self.NULL_IDX,
            start_idx=self.START_IDX,
            end_idx=self.END_IDX,
            unknown_idx=self.dict[self.dict.unk_token],
            longest_label=states.get("longest_label", 1),
        )

        """
        if opt.get("dict_tokenizer") == "bpe" and opt["embedding_type"] != "random":
            print("skipping preinitialization of embeddings for bpe")
        """
        if not states and opt["embedding_type"] != "random":
            # `not states`: only set up embeddings if not loading model
            self._copy_embeddings(model.decoder.lt.weight, opt["embedding_type"])

        if states:
            # set loaded states if applicable
            model.load_state_dict(states["model"])

        if opt["embedding_type"].endswith("fixed"):
            print("PerCVAE: fixing embedding weights.")
            model.decoder.lt.weight.requires_grad = False
            model.encoder.lt.weight.requires_grad = False
            if opt["lookuptable"] in ["dec_out", "all"]:
                model.output.weight.requires_grad = False

        return model

    def batchify(self, obs_batch, sort=True):
        """
        Add action and attribute supervision for batches.

        Store history vec as context_vec.
        """
        batch = super().batchify(obs_batch, sort)

        if batch.valid_indices is None:
            return batch

        # sum here is list concat, not addition
        context_vec, hist_lens_ = self._pad_tensor(
            sum([obs_batch[i]['context_vec'] for i in batch.valid_indices], [])
        )
        batch['context_vec'] = context_vec
        batch['hist_lens'] = torch.LongTensor(
            [len(obs_batch[i]['context_vec']) for i in batch.valid_indices]
        )
        
        return batch

    def _model_input(self, batch):
        return (batch.text_vec, batch.label_vec, batch.context_vec, batch.hist_lens, self.opt['datatype'])

    def _set_text_vec(self, obs, history, truncate):
        """
        Set the 'text_vec' field in the observation.

        Overridden to include both local utterance (text_vec) and full history
        (context_vec)
        """
        if "text" not in obs:
            return obs

        if "text_vec" not in obs:
            # text vec is not precomputed, so we set it using the history
            history_string = history.get_history_str()
            # when text not exist, we get text_vec from history string
            # history could be none if it is an image task and 'text'
            # filed is be empty. We don't want this
            if history_string is None:
                return obs
            obs["full_text"] = history_string
            if history_string:
                history_vec = history.get_history_vec_list()
                obs["text_vec"] = history_vec[-1]
                obs["full_text_vec"] = history.get_history_vec()
                obs["context_vec"] = history_vec

        # check truncation
        if obs.get("text_vec") is not None:
            truncated_vec = self._check_truncate(obs["text_vec"], truncate, True)
            obs.force_set("text_vec", torch.LongTensor(truncated_vec))

        return obs

    def _dummy_batch(self, batchsize, maxlen):
        """
        Overridden to add dummy context vec and hist lens.
        """
        batch = super()._dummy_batch(batchsize, maxlen)
        batch["context_vec"] = batch["text_vec"]
        batch["hist_lens"] = torch.ones(batchsize, dtype=torch.long)
        return batch

    def compute_loss(self, batch, return_output=False):
        """
        Compute and return the loss for the given batch.

        Easily overridable for customized loss functions.

        If return_output is True, the full output from the call to self.model()
        is also returned, via a (loss, model_output) pair.
        """
        if batch.label_vec is None:
            raise ValueError('Cannot compute loss without a label.')
        model_output = self.model(*self._model_input(batch), ys=batch.label_vec)
        scores, preds, encoder_output = model_output
        enc_out, bow_logits, kld, mmi = encoder_output

        score_view = scores.reshape(-1, scores.size(-1))
        rc_loss = self.criterion(score_view, batch.label_vec.view(-1))
        rc_loss = rc_loss.view(scores.shape[:-1]).sum(dim=1)

        # save loss to metrics
        notnull = batch.label_vec.ne(self.NULL_IDX)
        target_tokens = notnull.long().sum(dim=-1)
        correct = ((batch.label_vec == preds) * notnull).sum(dim=-1)

        bow_loss = -bow_logits.gather(1, batch.label_vec) * notnull
        bow_loss = bow_loss.sum(dim=1)

        if self.opt['datatype'] == 'train' and not return_output:
            self.global_t += 1
            kl_weights = min(self.global_t * 1.0 / self.full_kl_step, 1.0)
        else:
            kl_weights = 1.0

        # cross entropy loss
        self.record_local_metric('rc_loss', AverageMetric.many(rc_loss, target_tokens))
        # bow loss
        self.record_local_metric('bow_loss', AverageMetric.many(bow_loss, target_tokens))
        # kld
        if self.opt['datatype'] == 'train':
            self.record_local_metric('kld', AverageMetric.many(kld, target_tokens))
        # mmi
        self.record_local_metric('mmi', AverageMetric.many(torch.FloatTensor(batch.label_vec.size(0)).fill_(mmi)))
        # perplexity
        self.record_local_metric('ppl', PPLMetric.many(rc_loss, target_tokens))
        # kl weights
        self.record_local_metric('kl_weights', AverageMetric.many(torch.FloatTensor(batch.label_vec.size(0)).fill_(kl_weights)))
        # global_t
        self.record_local_metric('global_t', AverageMetric.many(torch.LongTensor(batch.label_vec.size(0)).fill_(self.global_t)))
        # token-wise accuracy
        self.record_local_metric('token_acc', AverageMetric.many(correct, target_tokens))
        # utterance-wise exact match
        self.record_local_metric('token_em', AverageMetric.many(correct == target_tokens))

        # actually do backwards loss
        rc_loss = rc_loss.sum()
        avg_rc_loss = rc_loss / target_tokens.sum()  # average rc loss per token

        bow_loss = bow_loss.sum()
        avg_bow_loss = bow_loss / target_tokens.sum()  # average bow loss per token

        loss = avg_rc_loss + avg_bow_loss

        if self.opt['datatype'] == 'train':
            kld = kld.sum()
            avg_kld = kld / target_tokens.sum()  # average kld per token
            loss += kl_weights * avg_kld

        # total loss
        self.record_local_metric('loss', AverageMetric.many(torch.FloatTensor(batch.label_vec.size(0)).fill_(loss)))

        if return_output:
            return (loss, model_output)
        else:
            return loss
