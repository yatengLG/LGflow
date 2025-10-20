# -*- coding: utf-8 -*-
# @Author  : LG


class Dataset(object):
    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

