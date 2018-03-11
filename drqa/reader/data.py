#!/usr/bin/env python3
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Data processing/loading helpers."""

import numpy as np
import logging
import random
import unicodedata

from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
from .vector import vectorize

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------------------
# Dictionary class for tokens.
# ------------------------------------------------------------------------------


class Dictionary(object):
    NULL = '<NULL>'
    UNK = '<UNK>'
    START = 2

    @staticmethod
    def normalize(token):
        return unicodedata.normalize('NFD', token)

    def __init__(self):
        self.tok2ind = {self.NULL: 0, self.UNK: 1}
        self.ind2tok = {0: self.NULL, 1: self.UNK}

    def __len__(self):
        return len(self.tok2ind)

    def __iter__(self):
        return iter(self.tok2ind)

    def __contains__(self, key):
        if type(key) == int:
            return key in self.ind2tok
        elif type(key) == str:
            return self.normalize(key) in self.tok2ind

    def __getitem__(self, key):
        if type(key) == int:
            return self.ind2tok.get(key, self.UNK)
        if type(key) == str:
            return self.tok2ind.get(self.normalize(key),
                                    self.tok2ind.get(self.UNK))

    def __setitem__(self, key, item):
        if type(key) == int and type(item) == str:
            self.ind2tok[key] = item
        elif type(key) == str and type(item) == int:
            self.tok2ind[key] = item
        else:
            raise RuntimeError('Invalid (key, item) types.')

    def add(self, token):
        token = self.normalize(token)
        if token not in self.tok2ind:
            index = len(self.tok2ind)
            self.tok2ind[token] = index
            self.ind2tok[index] = token

    def tokens(self):
        """Get dictionary tokens.

        Return all the words indexed by this dictionary, except for special
        tokens.
        """
        tokens = [k for k in self.tok2ind.keys()
                  if k not in {'<NULL>', '<UNK>'}]
        return tokens


# ------------------------------------------------------------------------------
# PyTorch dataset class for SQuAD (and SQuAD-like) data.
# ------------------------------------------------------------------------------


class ReaderDataset(Dataset):

    def __init__(self, examples, model, single_answer=False):
        self.model = model
        self.examples = examples
        self.single_answer = single_answer

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return vectorize(self.examples[index], self.model, self.single_answer)

    def labels(self):
        return [x['label'] for x in self.examples]

    def lengths(self):
        return [(len(ex['document']), len(ex['question']))
                for ex in self.examples]


# ------------------------------------------------------------------------------
# PyTorch sampler returning batched of sorted lengths (by doc and question).
# ------------------------------------------------------------------------------


class SortedBatchSampler(Sampler):

    def __init__(self, lengths, labels, batch_size, shuffle=True):
        self.lengths = lengths
        self.batch_size = batch_size
        self.shuffle = shuffle
        print(labels)
        self.pos_indices = [i for i,ex in enumerate(labels) if ex==1]
        self.neg_indices = [i for i,ex in enumerate(labels) if ex==0]

    def __iter__(self):

        self.pos_num = len(self.pos_indices)
        self.neg_num = len(self.neg_indices)
        dup = []
        while self.pos_num > self.neg_num:
            dup.extend(random.sample(self.neg_indices, min(self.pos_num - self.neg_num, len(self.neg_indices))))
            self.neg_num = len(dup) + len(self.neg_indices)
        while self.neg_num > self.pos_num:
            dup.extend(random.sample(self.pos_indices, min(len(self.pos_indices), self.neg_num-self.pos_num)))

            self.pos_num = len(dup) + len(self.pos_indices)

        indices = self.pos_indices + self.neg_indices + dup
        print(len(indices))
        random.shuffle(indices)
        batches = [indices[i:i + self.batch_size]
                   for i in range(0, len(indices), self.batch_size)]
        if self.shuffle:
            np.random.shuffle(batches)
        return iter([i for batch in batches for i in batch])

    def __len__(self):
        return 2 * max(len(self.pos_indices), len(self.neg_indices))
