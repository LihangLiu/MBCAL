from __future__ import print_function
import numpy as np
import os
from os.path import join, dirname, exists
import time
import datetime
import collections
import copy
import logging

import tensorflow as tf

import paddle
from paddle import fluid

import _init_paths

from src.utils import tik, tok, AssertEqual, save_pickle, read_pickle
from src.fluid_utils import (fluid_create_lod_tensor as create_tensor, seq_len_2_lod)
from data.npz_dataset import FakeTensor

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s') # filename='hot_rl.log', 


def add_scalar_summary(summary_writer, index, tag, value):
    logging.info("Step {}: {} {}".format(index, tag, value))
    if summary_writer:
        summary = tf.Summary()
        summary.value.add(tag=tag, simple_value=value)
        summary_writer.add_summary(summary, index)


class BatchData(object):
    def __init__(self, conf, tensor_dict):
        self.conf = conf
        self.tensor_dict = copy.deepcopy(tensor_dict)
        self.align_tensors_with_conf()

        self._decode_len = None

    def seq_lens(self):
        return self.tensor_dict[self.first_item_slot_name()].seq_lens

    def lod(self):
        return self.tensor_dict[self.first_item_slot_name()].lod

    def batch_size(self):
        return len(self.seq_lens())

    def offset(self):
        return self.lod()[0][:-1]

    def pos(self):
        return np.concatenate([np.arange(s) for s in self.seq_lens()], axis=0)

    def total_item_num(self):
        return np.sum(self.seq_lens())

    def decode_len(self):
        return self._decode_len

    def get_values(self, name):
        return self.tensor_dict[name].values

    def get_seq_lens(self, name):
        return self.tensor_dict[name].seq_lens

    def set_decode_len(self, decode_len):
        self._decode_len = np.array(decode_len)

    def first_item_slot_name(self):
        return self.conf.item_slot_names[0]

    def align_tensors_with_conf(self):
        for name, ft in self.tensor_dict.items():
            proper = self.conf.data_attributes[name]
            ft.values = ft.values.astype(proper['dtype'])
            if len(proper['shape']) - len(ft.values.shape) == 1:
                ft.values = np.expand_dims(ft.values, -1)

    def add_last_click_id(self):
        assert 'click_id' in self.tensor_dict, (self.tensor_dict.keys())
        batch_size = self.batch_size()
        click_id = self.get_values('click_id')
        last_click_id = click_2_last_click(click_id.reshape([batch_size, -1]))
        self.tensor_dict['last_click_id'] = FakeTensor(last_click_id.reshape(click_id.shape), 
                                                            self.seq_lens())

    def expand_candidates(self, other_batch_data, lens):
        """
        Regard other_batch_data as a candidate pool
        Only expand item-level values
            1. append values of self and other_batch_data
            2. construct index to get new batch_data
        lens: (batch_size,), len to expand
        
        ignore `last_click_id`
        """
        AssertEqual(len(lens), self.batch_size())

        total_cand_len = other_batch_data.total_item_num()
        total_item_len = self.total_item_num()
        cand_indice = np.arange(total_item_len, total_item_len + total_cand_len)     

        global_item_indice = []
        lod = self.lod()[0]
        for i in range(len(lod) - 1):
            start, end = lod[i], lod[i+1]
            old_indice = np.arange(start, end)
            new_indice = np.random.choice(cand_indice, size=lens[i], replace=False)
            global_item_indice.append(old_indice)
            global_item_indice.append(new_indice)
        global_item_indice = np.concatenate(global_item_indice, axis=0)

        prev_seq_lens = self.seq_lens()
        seq_lens = [s + l for s,l in zip(prev_seq_lens, lens)]
        # update tensor_dict
        for name in self.conf.item_slot_names:
            if name == 'last_click_id':
                continue
            values = np.concatenate([self.tensor_dict[name].values, other_batch_data.tensor_dict[name].values], 0)
            self.tensor_dict[name] = FakeTensor(values[global_item_indice], seq_lens)

    def get_candidates(self, pre_items, stop_flags=None):
        """
        pre_items: len() = batch_size
        stop_flags: (batch_size,)
        return:
            candidate_items: len() = batch_size, e.g. [[2,3,5], [3,4], ...]
        """
        if stop_flags is None:
            stop_flags = np.zeros([len(pre_items)])
        AssertEqual(len(pre_items), len(stop_flags))

        res = []
        for pre, seq_len, stop in zip(pre_items, self.seq_lens(), stop_flags):
            if stop:
                res.append([])
            else:
                full = np.arange(seq_len)
                res.append(np.setdiff1d(full, pre))
        return res

    def get_reordered(self, order, click_id):
        """
        get item-level features by order

        order: (b, order_len)
        click_id: (b, order_len)

        last_click_id will be removed
        """
        AssertEqual(len(order), self.batch_size())
        AssertEqual(len(click_id), self.batch_size())

        global_item_indice = []
        for sub_order, sub_offset in zip(order, self.offset()):
            global_item_indice.append(np.array(sub_order) + sub_offset)
        global_item_indice = np.concatenate(global_item_indice, axis=0)

        new_batch_data = BatchData(self.conf, self.tensor_dict)
        new_seq_lens = [len(od) for od in order]
        for name in new_batch_data.conf.item_slot_names:
            if name == 'last_click_id':
                continue
            else:
                v = new_batch_data.tensor_dict[name].values[global_item_indice]
            new_batch_data.tensor_dict[name] = FakeTensor(v, new_seq_lens)
        new_batch_data.tensor_dict['click_id'] = FakeTensor(click_id.reshape([-1, 1]), new_seq_lens)
        return new_batch_data

    def replace_following_items(self, pos, ref_batch_data):
        """
        Replace items starting from `pos` by items from `ref_batch_data`
        Replace click_id as well.
        Used for credit variance calculation

        Ignore last_click_id.
        """
        batch_size = self.batch_size()
        AssertEqual(batch_size, ref_batch_data.batch_size())

        new_batch_data = BatchData(self.conf, self.tensor_dict)
        for name in new_batch_data.conf.item_slot_names + new_batch_data.conf.label_slot_names:
            if name == 'last_click_id':
                continue
            else:
                v = new_batch_data.tensor_dict[name].values         # (b*seq_len, *)
                ref_v = ref_batch_data.tensor_dict[name].values     # (b*seq_len, *)
                tail_shape = list(v.shape[1:])
                new_v = np.concatenate([v.reshape([batch_size, -1] + tail_shape)[:, :pos],
                                    ref_v.reshape([batch_size, -1] + tail_shape)[:, pos:]], 1)  # (b, seq_len, *)
                new_v = new_v.reshape(v.shape)      # (b*seq_len, *)
                new_batch_data.tensor_dict[name].values = new_v
        return new_batch_data


def click_prob_2_score(click_prob):
    """
    args:
        click_prob: (n, dim)
    return:
        click_score: (n,)
    """
    AssertEqual(len(click_prob.shape), 2)
    dim0, dim1 = click_prob.shape
    weight = np.arange(dim1).reshape([1, -1])
    click_score = np.sum(click_prob * weight, 1)
    return click_score


def click_2_last_click(batch_click):
    """
    batch_click: (b, seq_len)
    """
    assert batch_click.ndim == 2, (batch_click.shape)
    batch_last_click = np.zeros_like(batch_click)
    batch_last_click[:, 1:] = batch_click[:, :-1]
    return batch_last_click




