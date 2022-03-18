#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from parlai.core.teachers import DialogTeacher
from parlai.utils.data import DatatypeHelper

import copy
import os
import codecs


def _path(opt, *additions):
    return os.path.join(opt['datapath'], 'Reddit', *additions)


class RedditTeacherForDialog(DialogTeacher):
    def __init__(self, opt, shared=None):
        self.id = 'reddit'
        self.fold = DatatypeHelper.fold(opt['datatype'])
        opt['datafile'] = os.path.join(opt['datapath'], self.fold + '.txt')
        super().__init__(opt, shared)

    def setup_data(self, datafile):
        turns4_file = _path(self.opt, 'original_file', 'reddit_conversations.4turns.' + self.fold + '.txt')
        turns5_file = _path(self.opt, 'original_file', 'reddit_conversations.5turns.' + self.fold + '.txt')

        with codecs.open(turns4_file, 'r') as f:
            for line in f:
                sentence_list = line.strip().split("\t")
                if len(sentence_list) != 4:
                    continue
                for i in [0, 2]:
                    yield {'text': sentence_list[i], 'label': sentence_list[i + 1]}, i == 0

        # with codecs.open(turns5_file, 'r') as f:
        #     for line in f:
        #         sentence_list = line.strip().split("\t")
        #         if len(sentence_list) != 5:
        #             continue
        #         for i in [0, 2]:
        #             yield {'text': sentence_list[i], 'label': sentence_list[i + 1]}, i == 0


class DefaultTeacher(RedditTeacherForDialog):
    pass



    