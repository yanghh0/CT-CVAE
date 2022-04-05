#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Transformer Agents.
"""
from typing import Optional
from parlai.core.params import ParlaiParser
from parlai.core.opt import Opt
from parlai.core.agents import Agent
from parlai.utils.torch import padded_3d, concat_without_padding
from parlai.core.torch_classifier_agent import TorchClassifierAgent
from parlai.core.torch_ranker_agent import TorchRankerAgent
from parlai.core.torch_generator_agent import TorchGeneratorAgent, PPLMetric
from parlai.utils.misc import recursive_getattr
from parlai.utils.logging import logging
from parlai.core.metrics import AverageMetric
from parlai.utils.misc import warn_once
from parlai.core.torch_agent import Output
from parlai.utils.io import PathManager
from parlai.agents.transformer.crossencoder import CrossEncoderModule
from .modules import CtcvaeModel, Length_dict, Symbol_dict, Specific_dict, cond_Spec, cond_Sym

import torch
import random


def add_common_cmdline_args(parser):
    """
    Add common command line args.
    """
    parser.add_argument(
        '-esz',
        '--embedding-size',
        type=int,
        default=300,
        help='Size of all embedding layers. Must be a multiple of --n-heads.',
    )
    parser.add_argument(
        '-nl', '--n-layers', type=int, default=8, help='Number of transformer layers.'
    )
    parser.add_argument(
        '-hid',
        '--ffn-size',
        type=int,
        default=2048,
        help='Hidden size of the FFN layers',
    )
    parser.add_argument(
        '--dropout',
        type=float,
        default=0.1,
        help='Dropout used around embeddings and before layer layer normalizations. '
        'This is used in Vaswani 2017 and works well on large datasets.',
    )
    parser.add_argument(
        '--attention-dropout',
        type=float,
        default=0.0,
        help='Dropout used after attention softmax. This is not used in Vaswani 2017.',
    )
    parser.add_argument(
        '--relu-dropout',
        type=float,
        default=0.0,
        help='Dropout used after the ReLU in the FFN. Not used in Vaswani 2017, '
        'but used in Tensor2Tensor.',
    )
    parser.add_argument(
        '--n-heads', type=int, default=16, help='Number of multihead attention heads'
    )
    parser.add_argument(
        '--d-model', type=int, default=512
    )
    parser.add_argument(
        '--element',
        type=str,
        default='none',
    )
    parser.add_argument(
        '--modulate',
        type=bool,
        default=True,
    )
    parser.add_argument(
        '--regular',
        type=bool,
        default=False,
    )
    parser.add_argument(
        '--latentsize',
        type=int,
        default=128,
    )
    parser.add_argument(
        '--control-embedding', type=int, default=30,
    )
    parser.add_argument(
        '--learn-positional-embeddings',
        type='bool',
        default=False,
        help='If off, sinusoidal embeddings are used. If on, position embeddings are '
        'learned from scratch.',
    )
    parser.add_argument('--embeddings-scale', type='bool', default=True)
    parser.add_argument(
        '--n-positions',
        type=int,
        default=None,
        hidden=True,
        help='Number of positional embeddings to learn. Defaults '
        'to truncate or 1024 if not provided.',
    )
    parser.add_argument(
        '--n-segments',
        type=int,
        default=0,
        help='The number of segments that support the model. '
        'If zero no segment and no langs_embedding.',
    )
    parser.add_argument(
        '--variant',
        choices={'aiayn', 'xlm', 'prelayernorm', 'bart'},
        default='aiayn',
        help='Chooses locations of layer norms, etc. prelayernorm '
        'is used to match some fairseq models',
        recommended='xlm',
    )
    parser.add_argument(
        '--activation',
        choices={'relu', 'gelu'},
        default='relu',
        help='Nonlinear activation to use. AIAYN uses relu, but '
        'more recent papers prefer gelu.',
        recommended='gelu',
    )
    parser.add_argument(
        '--output-scaling',
        type=float,
        default=1.0,
        help='scale the output of every transformer by this quantity.',
    )
    parser.add_argument(
        '--share-word-embeddings',
        type='bool',
        default=True,
        help='Share word embeddings table for candidate and context'
        'in the memory network',
    )
    parser.add_argument(
        '-nel',
        '--n-encoder-layers',
        type=int,
        default=-1,
        help='This will overide the n-layers for asymmetrical transformers',
    )
    parser.add_argument(
        '-ndl',
        '--n-decoder-layers',
        type=int,
        default=-1,
        help='This will overide the n-layers for asymmetrical transformers',
    )
    parser.add_argument(
        '--model-parallel',
        type='bool',
        default=False,
        help='Shard the layers across multiple GPUs.',
    )


class CtcvaeAgent(TorchGeneratorAgent):
    """
    TransformerGeneratorAgent.

    Implementation of TorchGeneratorAgent, where the model is a Transformer
    """

    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        """
        Add command-line arguments specifically for this agent.
        """
        agent = parser.add_argument_group('Transformer Arguments')
        add_common_cmdline_args(agent)
        cls.dictionary_class().add_cmdline_args(parser, partial_opt=partial_opt)

        super().add_cmdline_args(parser, partial_opt=partial_opt)
        return agent

    def build_model(self, states=None):
        """
        Build and return model.
        """
        self.global_t = 0
        self.p1_idx = self.dict["__newln__"]      # 4
        self.p2_idx = self.dict["__newln__"]      # 4
        self.pad_idx = self.dict["__null__"]      # 0

        self.cond_len_space_id = 0
        self.cond_sym_space_id = 0
        self.cond_spe_space_id = 0

        self.max_freq = list(self.dict.freq.values())[5]
        self.min_freq = list(self.dict.freq.values())[-1]

        model = CtcvaeModel(self.opt, self.dict)
        self.flag = 'parallel'  # parallel or single

        if self.opt["element"] == "D+" and self.opt['datatype'] != 'train':
            import parlai.utils.pickle

            with PathManager.open("/root/yanghh/checkpoint/crossencoder/empathetic_dialogues/model", 'rb') as f:
                states = torch.load(
                    f, map_location=lambda cpu, _: cpu, pickle_module=parlai.utils.pickle
                )

            self.crossencoder_opt = Opt.load("/root/yanghh/checkpoint/crossencoder/empathetic_dialogues/model.opt")
            self.crossencoder = CrossEncoderModule(self.crossencoder_opt, self.dict, self.NULL_IDX)
            self.crossencoder.load_state_dict(states['model'])
            self.crossencoder.cuda()

        if self.opt['embedding_type'] != 'random':
            self._copy_embeddings(
                model.encoder.embeddings.weight, self.opt['embedding_type']
            )

        return model

    def _resize_token_embeddings(self, state_dict, msg=None):
        """
        Resize the token embeddings when are adding extra special tokens.
        """
        # map extra special tokens carefully
        new_size = self.model.embeddings.weight.size()[0]
        orig_size = state_dict['embeddings.weight'].size()[0]
        logging.info(f'Resizing token embeddings from {orig_size} to {new_size}')
        if new_size <= orig_size:
            # new size should be greater than original size,
            # as we are adding special tokens
            raise RuntimeError(msg)

        for emb_weights in [
            'embeddings.weight',
            'encoder.embeddings.weight',
            'decoder.embeddings.weight',
        ]:
            # get new_embs
            old_embs = state_dict[emb_weights]
            new_embs = recursive_getattr(self.model, emb_weights).to(old_embs.device)
            # copy over old weights
            new_embs.data[:orig_size, :] = old_embs.data[:orig_size, :]
            # reset in state dict
            state_dict[emb_weights] = new_embs

        return state_dict

    def _model_input(self, batch):
        if self.opt["element"] == "D":
            if self.opt['datatype'] == 'train':
                class_labels = batch.class_labels
            else:
                if self.flag == 'single':
                    class_labels = torch.ones_like(batch.class_labels).fill_(self.cond_spe_space_id)
                elif self.flag == 'parallel':
                    batch.text_vec = batch.text_vec.repeat(len(Specific_dict), 1)
                    batch.turn_vec = batch.turn_vec.repeat(len(Specific_dict), 1)
                    batch.role_vec = batch.role_vec.repeat(len(Specific_dict), 1)
                    batch.full_text_vec = batch.full_text_vec.repeat(len(Specific_dict), 1)
                    batch.add_start_end_text_vec = batch.add_start_end_text_vec.repeat(len(Specific_dict), 1)
                    class_labels = torch.LongTensor(list(range(len(Specific_dict)))).cuda()
        elif "D+" in self.opt["element"]:
            if self.opt['datatype'] == 'train':
                class_labels = [
                    batch.class_labels_specificity,
                    batch.class_labels_symbol, 
                    batch.class_random_labels_symbol, 
                ]
            else:
                if self.flag == 'single':
                    class_labels = [
                        torch.ones_like(batch.class_labels_specificity).fill_(self.cond_spe_space_id),
                        torch.ones_like(batch.class_labels_symbol).fill_(self.cond_sym_space_id),
                    ]
                elif self.flag == 'parallel':
                    batch.text_vec = batch.text_vec.repeat(len(Specific_dict) * 2, 1)
                    batch.turn_vec = batch.turn_vec.repeat(len(Specific_dict) * 2, 1)
                    batch.role_vec = batch.role_vec.repeat(len(Specific_dict) * 2, 1)
                    batch.full_text_vec = batch.full_text_vec.repeat(len(Specific_dict) * 2, 1)
                    batch.add_start_end_text_vec = batch.add_start_end_text_vec.repeat(len(Specific_dict) * 2, 1)
                    class_labels = [
                        torch.LongTensor([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5]).cuda(),
                        torch.LongTensor([1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]).cuda(),
                    ]
        else:
            class_labels = None

        return (batch.text_vec, 
                batch.post_text_vec,
                batch.reverse_post_text_vec,
                class_labels,
                None,
                batch.turn_vec,
                batch.role_vec,
                None,
                batch.post_turn_vec,
                batch.post_role_vec,
                None,
                batch.reverse_post_turn_vec,
                batch.reverse_post_role_vec,
                )

    def gen_length_category(self, batch, obs_batch):
        _mask = (batch.label_vec != self.pad_idx) & (batch.label_vec != self.p1_idx) & (batch.label_vec != self.p2_idx)
        length_labels = []
        random_length_labels = []
        for i in _mask.sum(1).numpy().tolist():
            if i > 30:
                key = 'other'
            elif i % 6 == 0:
                left = i - 5
                right = i
                key = str(left) + '-' + str(right)
            else:
                left = i // 6 * 6 + 1
                right = (i + 6) // 6 * 6
                key = str(left) + '-' + str(right)
            length_labels.append(Length_dict[key])
            random_label = Length_dict[key]
            while random_label == Length_dict[key] or random_label == Length_dict[key] + 1 or random_label == Length_dict[key] - 1:
                random_label = random.randint(0, 5)
            random_length_labels.append(random_label)
        return length_labels, random_length_labels

    def gen_symbol_category(self, batch, obs_batch):
        symbol_labels = []
        for label in batch.label_vec:
            if self.dict["?"] in label:
                symbol_labels.append(Symbol_dict["?"])
            else:
                symbol_labels.append(Symbol_dict["other"])
        return symbol_labels, symbol_labels

    def gen_specificity_category(self, batch, obs_batch):
        specificity_labels = []
        for label in batch.label_vec:
            total = 0.0
            for tok_id in label.numpy().tolist():
                if tok_id in (0, 1, 2, 3, 4):
                    continue
                total += 1 - ((self.dict.freq[self.dict[tok_id]] - self.min_freq) * 1.0 / (self.max_freq - self.min_freq))
            total = int(total) + 1
            if total > 30:
                key = 'other'
            elif total % 6 == 0:
                left = total - 5
                right = total
                key = str(left) + '-' + str(right)
            else:
                left = total // 6 * 6 + 1
                right = (total + 6) // 6 * 6
                key = str(left) + '-' + str(right)
            specificity_labels.append(Specific_dict[key])
        return specificity_labels

    def batchify(self, obs_batch, sort=True):
        """
        Add action and attribute supervision for batches.

        Store history vec as context_vec.
        """
        batch = super().batchify(obs_batch, sort)

        if self.opt["element"] == "D":
            batch['class_labels'] = torch.LongTensor(
                # self.gen_length_category(batch, obs_batch)
                # self.gen_symbol_category(batch, obs_batch)
                self.gen_specificity_category(batch, obs_batch)
            )
        elif "D+" in self.opt["element"]:
            batch['class_labels_specificity'] = torch.LongTensor(self.gen_specificity_category(batch, obs_batch))
            symbol_labels, random_symbol_labels = self.gen_symbol_category(batch, obs_batch)
            batch['class_labels_symbol'] = torch.LongTensor(symbol_labels)
            batch['class_random_labels_symbol'] = torch.LongTensor(random_symbol_labels)

        return batch

    def _dummy_batch(self, batchsize, maxlen):
        """
        Overridden to add dummy context vec and hist lens.
        """
        batch = super()._dummy_batch(batchsize, maxlen)
        if self.opt["element"] == "D":
            batch["class_labels"] = torch.ones(batchsize, dtype=torch.long).cuda()
        elif "D+" in self.opt["element"]:
            batch["class_labels_specificity"] = torch.ones(batchsize, dtype=torch.long).cuda()
            batch["class_labels_symbol"] = torch.ones(batchsize, dtype=torch.long).cuda()
            batch["class_random_labels_symbol"] = torch.ones(batchsize, dtype=torch.long).cuda()

        batch['turn_vec'] = (torch.ones_like(batch.text_vec).cuda())
        batch['role_vec'] = (torch.zeros_like(batch.text_vec).cuda())

        batch['post_text_vec'] = (
            torch.arange(1, maxlen + 1)
            .clamp(max=3)
            .unsqueeze(0)
            .expand(batchsize, maxlen)
            .cuda()
        )
        batch['post_turn_vec'] = (torch.ones_like(batch.post_text_vec).cuda())
        batch['post_role_vec'] = (torch.zeros_like(batch.post_text_vec).cuda())

        batch['reverse_post_text_vec'] = (
            torch.arange(1, maxlen + 1)
            .clamp(max=3)
            .unsqueeze(0)
            .expand(batchsize, maxlen)
            .cuda()
        )
        batch['reverse_post_turn_vec'] = (torch.ones_like(batch.reverse_post_text_vec).cuda())
        batch['reverse_post_role_vec'] = (torch.zeros_like(batch.reverse_post_text_vec).cuda())

        batch['add_start_end_text_vec'] = (
            torch.arange(1, maxlen + 1)
            .clamp(max=3)
            .unsqueeze(0)
            .expand(batchsize, maxlen)
            .cuda()
        )

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
        scores, preds, encoder_states = model_output
        _, _, _, kld, regular_item, bow_logits, _, mmi = encoder_states

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
            kl_weights = min(self.global_t / 250000.0, 1.2)
        else:
            kl_weights = 1.2

        # cross entropy loss
        self.record_local_metric('rc_loss', AverageMetric.many(rc_loss, target_tokens))
        # bow loss
        self.record_local_metric('bow_loss', AverageMetric.many(bow_loss, target_tokens))
        # kld
        if self.opt['datatype'] == 'train':
            self.record_local_metric('kld', AverageMetric.many(kld, target_tokens))
            if self.opt["regular"] == True:
                self.record_local_metric('R1', AverageMetric.many(regular_item, target_tokens))
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

        loss = avg_bow_loss + avg_rc_loss

        if self.opt['datatype'] == 'train':
            kld = kld.sum()
            avg_kld = kld / target_tokens.sum()  # average kld per token
            loss += kl_weights * avg_kld

            if self.opt["regular"] == True:
                avg_regular = regular_item.sum() / target_tokens.sum()
                loss += 0.5 * avg_regular

        # total loss
        self.record_local_metric('loss', AverageMetric.many(torch.FloatTensor(batch.label_vec.size(0)).fill_(loss)))

        if return_output:
            return (loss, model_output)
        else:
            return loss

    def eval_step1(self, batch):
        """
        Evaluate a single batch of examples.
        """
        if batch.text_vec is None and batch.image is None:
            return
        if batch.text_vec is not None:
            bsz = batch.text_vec.size(0)
        else:
            bsz = len(batch.image)
        self.model.eval()
        cand_scores = None
        token_losses = None

        if batch.label_vec is not None:
            # calculate loss on targets with teacher forcing
            loss, model_output = self.compute_loss(batch, return_output=True)
            if self.output_token_losses:
                token_losses = self._construct_token_losses(
                    batch.label_vec, model_output
                )

        preds = None
        if self.skip_generation:
            warn_once("--skip-generation true produces limited metrics")
        else:
            maxlen = self.label_truncate or 256
            beam_preds_scores, beams = self._generate(batch, self.beam_size, maxlen)
            preds, scores = zip(*beam_preds_scores)
            self._add_generation_metrics(batch, preds)

            # bsz x beamsize
            beam_texts: List[List[Tuple[str, float]]] = []
            for beam in beams:
                beam_texts.append([])
                for tokens, score in beam.get_rescored_finished():
                    try:
                        beam_texts[-1].append((self._v2t(tokens), score.item()))
                    except KeyError:
                        logging.error("Decoding error: %s", tokens)
                        continue

        cand_choices = None
        cand_scores = None
        if self.rank_candidates:
            cand_choices, cand_scores = self.rank_eval_label_candidates(batch, bsz)

        text = [self._v2t(p) for p in preds] if preds is not None else None

        if self.opt['datatype'] == 'valid':
            from parlai.utils.misc import msg_to_str
            act = {
                'text': '',
                'labels': [''],
                'cond1': '',
                'cond2': '',
                'episode_done': True,
            }

            label_str_list = [self._v2t(p) for p in batch.label_vec] if batch.label_vec is not None else None
            pred_str_list = text

            for pred_str in pred_str_list:
                pred_str_length = len(pred_str.split())
                self.total_length += pred_str_length
                self.total_sentence += 1
                if pred_str_length > self.max_length:
                    self.max_length = pred_str_length
                if pred_str_length < self.min_length:
                    self.min_length = pred_str_length
                try:
                    self._length[pred_str_length] += 1
                except Exception as e:
                    print(pred_str)

            print(self.total_length * 1.0 / self.total_sentence)
            print(self.total_length, self.total_sentence, self.max_length, self.min_length)
            print(self._length)

            with open("semantic_evaluation.txt", "a") as fs:
                for cnt, label_str in enumerate(label_str_list):
                    act['text'] = label_str
                    act['labels'] = [pred_str_list[cnt]]
                    txt = msg_to_str(act)
                    fs.write(txt + '\n')
                    fs.write('\n')

        if text and self.compute_tokenized_bleu:
            # compute additional bleu scores
            self._compute_fairseq_bleu(batch, preds)
        retval = Output(
            text, cand_choices, token_losses=token_losses, cand_scores=cand_scores
        )

        if not self.skip_generation:
            retval.beam_texts = beam_texts
        return retval

    def eval_step2(self, batch):
        """
        Evaluate a single batch of examples.
        """
        if batch.text_vec is None and batch.image is None:
            return
        if batch.text_vec is not None:
            bsz = batch.text_vec.size(0)
        else:
            bsz = len(batch.image)
        self.model.eval()
        cand_scores = None
        token_losses = None

        if self.opt['datatype'] == 'train':
            if batch.label_vec is not None:
                # calculate loss on targets with teacher forcing
                loss, model_output = self.compute_loss(batch, return_output=True)
                if self.output_token_losses:
                    token_losses = self._construct_token_losses(
                        batch.label_vec, model_output
                    )

            preds = None
            if self.skip_generation:
                warn_once("--skip-generation true produces limited metrics")
        else:
            maxlen = self.label_truncate or 256
            self.crossencoder.eval()

            if self.opt["element"] == "D":                
                _beam_preds_scores, beams = self._generate(batch, self.beam_size, maxlen)
                _preds, _scores = zip(*_beam_preds_scores)

                tokens, segments = concat_without_padding(
                    batch.add_start_end_text_vec, self._pad_tensor(_preds)[0], self.use_cuda, self.NULL_IDX
                )

                scores = self.crossencoder(tokens, segments)
                scores = scores.squeeze(1)

                out, idx = torch.topk(scores, k=2, dim=0)
                select_condition = torch.max(idx).item()

                beam_preds_scores = [_beam_preds_scores[select_condition]]
                preds = (_preds[select_condition], )
                scores = (_scores[select_condition], )

                self._add_generation_metrics(batch, preds)

                batch.text_vec = batch.text_vec[0].unsqueeze(0)
                batch.turn_vec = batch.turn_vec[0].unsqueeze(0)
                batch.role_vec = batch.role_vec[0].unsqueeze(0)
                batch.full_text_vec = batch.full_text_vec[0].unsqueeze(0)
                batch.add_start_end_text_vec = batch.add_start_end_text_vec[0].unsqueeze(0)

            elif self.opt["element"] == "D+":
                _beam_preds_scores, beams = self._generate(batch, self.beam_size, maxlen)
                _preds, _scores = zip(*_beam_preds_scores)

                tokens, segments = concat_without_padding(
                    batch.add_start_end_text_vec, self._pad_tensor(_preds)[0], self.use_cuda, self.NULL_IDX
                )

                scores = self.crossencoder(tokens, segments)
                scores = scores.squeeze(1)

                out, idx = torch.topk(scores, k=2, dim=0)
                select_condition = torch.max(idx).item()

                beam_preds_scores = [_beam_preds_scores[select_condition]]
                preds = (_preds[select_condition], )
                scores = (_scores[select_condition], )

                self._add_generation_metrics(batch, preds)

                batch.text_vec = batch.text_vec[0].unsqueeze(0)
                batch.turn_vec = batch.turn_vec[0].unsqueeze(0)
                batch.role_vec = batch.role_vec[0].unsqueeze(0)
                batch.full_text_vec = batch.full_text_vec[0].unsqueeze(0)
                batch.add_start_end_text_vec = batch.add_start_end_text_vec[0].unsqueeze(0)

            if batch.label_vec is not None:
                if self.opt["element"] == "D":
                    self.cond_spe_space_id = select_condition
                elif self.opt["element"] == "D+":
                    self.cond_spe_space_id = cond_Spec[select_condition]
                    self.cond_sym_space_id = cond_Sym[select_condition]
                self.flag = 'single'
                # calculate loss on targets with teacher forcing
                loss, model_output = self.compute_loss(batch, return_output=True)
                if self.output_token_losses:
                    token_losses = self._construct_token_losses(
                        batch.label_vec, model_output
                    )
                self.flag = 'parallel'

            # bsz x beamsize
            beam_texts: List[List[Tuple[str, float]]] = []
            for beam in beams:
                beam_texts.append([])
                for tokens, score in beam.get_rescored_finished():
                    try:
                        beam_texts[-1].append((self._v2t(tokens), score.item()))
                    except KeyError:
                        logging.error("Decoding error: %s", tokens)
                        continue

            if self.opt["element"] == "D":
                beam_texts = [beam_texts[select_condition]]
            elif self.opt["element"] == "D+":
                beam_texts = [beam_texts[select_condition]]

        cand_choices = None
        cand_scores = None
        if self.rank_candidates:
            cand_choices, cand_scores = self.rank_eval_label_candidates(batch, bsz)

        text = [self._v2t(p) for p in preds] if preds is not None else None

        if text and self.compute_tokenized_bleu:
            # compute additional bleu scores
            self._compute_fairseq_bleu(batch, preds)
        retval = Output(
            text, cand_choices, token_losses=token_losses, cand_scores=cand_scores
        )

        if not self.skip_generation:
            retval.beam_texts = beam_texts
        return retval
