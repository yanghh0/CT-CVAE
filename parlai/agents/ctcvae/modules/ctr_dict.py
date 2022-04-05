#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


Length_dict = {
    "1-6"  : 0,
    "7-12" : 1,
    "13-18": 2,
    "19-24": 3,
    "25-30": 4,
    "other": 5
}

Symbol_dict = {
    "?"    : 0,
    "other": 1
}

Specific_dict = {
    "1-6"  : 0,
    "7-12" : 1,
    "13-18": 2,
    "19-24": 3,
    "25-30": 4,
    "other": 5
}

cond_Spec = [0, 0,
             1, 1,
             2, 2,
             3, 3,
             4, 4,
             5, 5]

cond_Sym = [1, 0,
            1, 0,
            1, 0,
            1, 0,
            1, 0,
            1, 0]