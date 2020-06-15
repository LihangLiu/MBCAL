from __future__ import print_function
import numpy as np
import copy
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s') # filename='hot_rl.log', 

import paddle
from paddle import fluid

from utils import click_2_last_click

import _init_paths

from src.utils import AssertEqual
from src.fluid_utils import (fluid_create_lod_tensor as create_tensor, seq_len_2_lod)
from data.npz_dataset import FakeTensor


class GenFeedConvertor(object):
    """
    For env, credit or rl models
    """
    @staticmethod
    def infer_init(batch_data, conf=None):
        if conf is None:
            conf = batch_data.conf
            
        place = fluid.CPUPlace()
        feed_dict = {}
        for name in conf.recent_slot_names:
            ft = batch_data.tensor_dict[name]
            feed_dict[name] = create_tensor(ft.values, lod=ft.lod, place=place)
        return feed_dict

    @staticmethod
    def infer_onestep(batch_data, prev_hidden, candidate_items, last_click_id, conf=None):
        """
        batch_data: the complete data
            item_feature: (b*seq_len,)
        prev_hidden: (b, dim)
        last_click_id: (b,)
        candidate_items: 2d list
        """
        if conf is None:
            conf = batch_data.conf

        batch_size = batch_data.batch_size()
        seq_len = batch_data.seq_lens()[0]
        cand_len = len(candidate_items[0])
        batch_offset = batch_data.offset()
            
        # expand last_click_id
        last_click_id = np.repeat(last_click_id, cand_len, axis=0).reshape([-1, 1])   # (b*cand_len, 1)

        place = fluid.CPUPlace()
        feed_dict = {}
        lod = [seq_len_2_lod([1] * (batch_size * cand_len))]
        offset_candidate_items = np.array(candidate_items).flatten() + np.repeat(batch_offset, cand_len, axis=0)    # (b*cand_len)
        for name in conf.item_slot_names:
            if name == 'last_click_id':
                v = last_click_id
            else:
                v = batch_data.tensor_dict[name].values[offset_candidate_items]
            feed_dict[name] = create_tensor(v, lod=lod, place=place)

        # expand prev_hidden
        prev_hidden = np.repeat(prev_hidden, cand_len, axis=0)       # (b*cand_len, dim)
        feed_dict['prev_hidden'] = create_tensor(prev_hidden, lod=lod, place=place)
        return feed_dict


class EnvFeedConvertor(object):
    @staticmethod
    def train_test(batch_data, item_dropout_rate, conf=None):
        """
        Will randomly mask some items.
        """
        if conf is None:
            conf = batch_data.conf

        # item mask
        total_item_num = np.sum(batch_data.seq_lens())
        item_mask = np.random.binomial(1, 1 - item_dropout_rate, total_item_num)    # (b*seq_len,)
            
        batch_data.add_last_click_id()

        place = fluid.CPUPlace()
        feed_dict = {}
        for name in conf.recent_slot_names + \
                    conf.item_slot_names + \
                    conf.label_slot_names:
            ft = batch_data.tensor_dict[name]
            if name in conf.item_slot_names and name != 'last_click_id':
                v = ft.values * item_mask.reshape([-1] + [1] * (len(ft.values.shape) - 1))
            else:
                v = ft.values
            feed_dict[name] = create_tensor(v, lod=ft.lod, place=place)
        return feed_dict


class CFFeedConvertor(object):
    @staticmethod
    def train_test(batch_data, conf=None):
        if conf is None:
            conf = batch_data.conf

        batch_data.add_last_click_id()

        place = fluid.CPUPlace()
        feed_dict = {}
        for name in conf.recent_slot_names + \
                    conf.item_slot_names + \
                    conf.label_slot_names:
            ft = batch_data.tensor_dict[name]
            feed_dict[name] = create_tensor(ft.values, lod=ft.lod, place=place)
        return feed_dict


class CreditFeedConvertor(object):
    @staticmethod
    def train_test(batch_data, credit, conf=None):
        """
        credit: (b*seq_len,)
        """
        if conf is None:
            conf = batch_data.conf
        assert credit.ndim == 1 and len(credit) == np.sum(batch_data.seq_lens()), (credit.shape)
            
        batch_data.add_last_click_id()

        # add credit
        batch_data.tensor_dict['credit'] = FakeTensor(credit.reshape([-1, 1]),
                                                    batch_data.seq_lens())

        place = fluid.CPUPlace()
        feed_dict = {}
        for name in conf.recent_slot_names + \
                    conf.item_slot_names + \
                    conf.label_slot_names:
            ft = batch_data.tensor_dict[name]
            feed_dict[name] = create_tensor(ft.values, lod=ft.lod, place=place)
        return feed_dict

    @staticmethod
    def apply_masks(batch_data, list_item_masks, conf=None):
        """
        list_item_masks: (n_masks, seq_len), a list of 1d item_mask
        Apply mask on item_level_slot_names except last_click_id
        """
        if conf is None:
            conf = batch_data.conf

        batch_size = batch_data.batch_size()
        seq_len = batch_data.seq_lens()[0]
        AssertEqual(len(list_item_masks[0]), seq_len)

        batch_data.add_last_click_id()

        n_masks = len(list_item_masks)
        batch_item_masks = np.tile(np.array(list_item_masks).flatten(), [batch_size])   # (batch_size * n_masks * seq_len)

        place = fluid.CPUPlace()
        feed_dict = {}
        for name in conf.recent_slot_names + \
                    conf.item_slot_names:
            ft = batch_data.tensor_dict[name]
            v = ft.values
            extra_shape = list(v.shape[1:])
            v = v.reshape([batch_size, -1] + extra_shape)  # (batch_size, seq_len/recent_len, ...)
            v = np.repeat(v, n_masks, axis=0)              # (batch_size * n_masks, seq_len/recent_len, ...)
            seq_lens = [v.shape[1]] * (batch_size * n_masks)
            v = v.reshape([-1] + extra_shape)             # (batch_size * n_masks * seq_len/recent_len, ...)
            if name in conf.item_slot_names and name != 'last_click_id':
                v = v * batch_item_masks.reshape([-1] + [1] * (len(v.shape) - 1))
            feed_dict[name] = create_tensor(v, lod=[seq_len_2_lod(seq_lens)], place=place)
        return feed_dict


class RLFeedConvertor(object):
    @staticmethod
    def train_test(batch_data, conf=None):
        """
        """
        if conf is None:
            conf = batch_data.conf
            
        batch_data.add_last_click_id()

        place = fluid.CPUPlace()
        feed_dict = {}
        for name in conf.recent_slot_names + \
                    conf.item_slot_names + \
                    conf.label_slot_names:
            ft = batch_data.tensor_dict[name]
            feed_dict[name] = create_tensor(ft.values, lod=ft.lod, place=place)
        return feed_dict


class DDPGFeedConvertor(object):
    @staticmethod
    def train_test(batch_data, conf=None):
        """
        """
        if conf is None:
            conf = batch_data.conf
            
        batch_data.add_last_click_id()

        place = fluid.CPUPlace()
        feed_dict = {}
        for name in conf.recent_slot_names + \
                    conf.item_slot_names + \
                    conf.label_slot_names:
            ft = batch_data.tensor_dict[name]
            feed_dict[name] = create_tensor(ft.values, lod=ft.lod, place=place)
        return feed_dict

    @staticmethod
    def infer_onestep(batch_data, prev_hidden, candidate_items, last_click_id, last_item, step, conf=None):
        """
        batch_data: the complete data
            item_feature: (b*seq_len,)
        prev_hidden: (b, dim)
        last_click_id: (b,)
        last_item: (b,)
        step: int
        candidate_items: 2d list

        Different from other model, ddpg don't need to expand.
        """
        if conf is None:
            conf = batch_data.conf

        batch_size = batch_data.batch_size()
        seq_len = batch_data.seq_lens()[0]
        cand_len = len(candidate_items[0])
        batch_offset = batch_data.offset()

        place = fluid.CPUPlace()
        feed_dict = {}
        item_slot_names = list(conf.item_slot_names)
        item_slot_names.remove('last_click_id')
        last_item_slot_names = ['last_'+name for name in item_slot_names]

        ### candidates
        lod = [seq_len_2_lod([cand_len] * batch_size)]
        offset_candidate_items = np.array(candidate_items).flatten() + np.repeat(batch_offset, cand_len, axis=0)    # (b*cand_len)
        for name in item_slot_names:
            v = batch_data.tensor_dict[name].values[offset_candidate_items]
            feed_dict[name] = create_tensor(v, lod=lod, place=place)

        ### last item
        lod = [seq_len_2_lod([1] * batch_size)]
        offset_last_item = np.array(last_item).flatten() + batch_offset    # (b,)
        for last_name in last_item_slot_names:
            name = last_name[len('last_'):]
            v = batch_data.tensor_dict[name].values[offset_last_item]
            feed_dict[last_name] = create_tensor(v, lod=lod, place=place)

        ### last click
        lod = [seq_len_2_lod([1] * batch_size)]
        feed_dict['last_click_id'] = create_tensor(last_click_id.reshape([-1,1]), lod=lod, place=place)     # (b, 1)

        ### prev_hidden
        lod = [seq_len_2_lod([1] * batch_size)]
        feed_dict['prev_hidden'] = create_tensor(prev_hidden, lod=lod, place=place)     # (b, dim)

        ### first_step_mask
        lod = [seq_len_2_lod([1] * batch_size)]
        first_step_mask = np.full([batch_size, 1], float(step > 0)).astype('float32')
        feed_dict['first_step_mask'] = create_tensor(first_step_mask, lod=lod, place=place)     # (b, 1)
        return feed_dict


