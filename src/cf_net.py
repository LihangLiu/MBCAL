#!/usr/bin/env python
# coding=utf8
import numpy as np
import sys
import os
from os.path import exists
import copy
import logging
from collections import OrderedDict

from paddle import fluid
from paddle.fluid.param_attr import ParamAttr

PARL_DIR = os.environ.get('PARL_DIR', '')
assert exists(PARL_DIR), ('PARL_DIR', PARL_DIR, 'not exists')
sys.path.insert(0, PARL_DIR)

import parl.layers as layers
from parl.framework.algorithm import Model
from fluid_utils import (fluid_sequence_get_pos, fluid_sequence_first_step, fluid_sequence_index,
                         fluid_sequence_pad, fluid_sequence_get_seq_len)

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s') # filename='hot_rl.log', 


def default_normal_initializer(nf=128):
    return fluid.initializer.TruncatedNormal(loc=0.0, scale=np.sqrt(1.0/nf))

def default_param_clip():
    return fluid.clip.GradientClipByValue(1.0)

def default_fc(size, num_flatten_dims=1, act=None, name=None):
    return layers.fc(size=size,
                   num_flatten_dims=num_flatten_dims,
                   param_attr=ParamAttr(initializer=default_normal_initializer(size),
                                        gradient_clip=default_param_clip()),
                   bias_attr=ParamAttr(initializer=fluid.initializer.Constant(value=0.0),
                                        gradient_clip=default_param_clip()),
                   act=act,
                   name=name)

def default_embedding(size, name, embed_clip, regularizer=None):
    gradient_clip = default_param_clip() if embed_clip else None
    embed = layers.embedding(name=name,
                            size=size,
                            param_attr=ParamAttr(initializer=fluid.initializer.Xavier(),
                                                gradient_clip=gradient_clip,
                                                regularizer=regularizer),
                            is_sparse=False,     # turn on lazy_mode when using Adam
                            is_distributed=False,    # TODO https://github.com/PaddlePaddle/Paddle/issues/15133
                            )
    return embed

def default_drnn(nf, is_reverse=False, name=None):
    return layers.dynamic_gru(size=nf,
                            param_attr=ParamAttr(initializer=default_normal_initializer(nf),
                                                gradient_clip=default_param_clip()),
                            bias_attr=ParamAttr(initializer=fluid.initializer.Constant(value=0.0),
                                                gradient_clip=default_param_clip()),
                            is_reverse=is_reverse,
                            name=name)

def default_lstm(nf, is_reverse=False, name=None):
    return layers.dynamic_lstm(size=nf,
                            param_attr=ParamAttr(initializer=default_normal_initializer(nf),
                                                gradient_clip=default_param_clip()),
                            bias_attr=ParamAttr(initializer=fluid.initializer.Constant(value=0.0),
                                                gradient_clip=default_param_clip()),
                            is_reverse=is_reverse,
                            name=name)


class NeuralCF(Model):
    """
    """
    def __init__(self, conf, npz_config, scope=None, embed_regular=0.0, output_type='click', output_dim=2):
        super(NeuralCF, self).__init__()
        self.conf = conf
        self.npz_config = npz_config
        self.data_attributes = conf.data_attributes
        # feature related initialization
        self.item_slot_names = conf.item_slot_names
        self.recent_slot_names = conf.recent_slot_names
        self.label_slot_names = conf.label_slot_names
        self.shared_embedding_names = conf.shared_embedding_names
        self._embed_regular = embed_regular
        self._output_type = output_type
        self._output_dim = output_dim
        self.hidden_size = 32

        assert self._output_type in ['click', 'credit', 'click_credit', 'rate'], (self._output_type)

        self.scope = fluid.global_scope() if scope is None else scope
        with fluid.scope_guard(scope):
            with fluid.unique_name.guard():
                self._create_params()

    def get_input_specs(self):
        """ignore"""
        return []

    def get_action_specs(self):
        """ignore"""
        return []

    def _create_params(self):
        ### embed
        self.dict_data_embed_op = {}
        list_names = self.item_slot_names + self.recent_slot_names
        regularizer = fluid.regularizer.L2Decay(self._embed_regular) if self._embed_regular > 0 else None
        for name in list_names:
            embed_name = self._get_embed_name(name)
            vob_size = self.npz_config['embedding_size'][embed_name] + 1
            embed_size = 16
            if embed_name not in self.dict_data_embed_op:
                self.dict_data_embed_op[embed_name] = \
                        default_embedding([vob_size, embed_size], 
                                          'embed_' + embed_name, 
                                          embed_clip=None, 
                                          regularizer=regularizer)

        self.recent_fc_op = default_fc(self.hidden_size * 3, act='relu', name='recent_fc')
        self.recent_gru_op = default_drnn(self.hidden_size, name='recent_gru')
        self.user_feature_fc_op = default_fc(self.hidden_size, act='relu', name='user_feature_fc')

        self.item_fc_op = default_fc(self.hidden_size, act='relu', name='item_fc')

        if 'click' in self._output_type:
            self.out_click_fc1_op = default_fc(self.hidden_size, act='relu', name='out_click_fc1')
            self.out_click_fc2_op = default_fc(self._output_dim, act='softmax', name='out_click_fc2')
        if 'credit' in self._output_type:
            self.out_credit_fc1_op = default_fc(self.hidden_size, act='relu', name='out_credit_fc1')
            self.out_credit_fc2_op = default_fc(1, act=None, name='out_credit_fc2')
        if 'rate' in self._output_type:
            self.out_rate_fc1_op = default_fc(self.hidden_size, act='relu', name='out_rate_fc1')
            self.out_rate_fc2_op = default_fc(1, act=None, name='out_rate_fc2')

    def _get_embed_name(self, name):
        """map a slot_name to a embed_name"""
        if name in self.shared_embedding_names:
            return self.shared_embedding_names[name]
        return name

    def _build_embeddings(self, inputs, list_names):
        list_embed = []
        for name in list_names:
            embed_name = self._get_embed_name(name)
            c_embed = self.dict_data_embed_op[embed_name](inputs[name])
            if len(c_embed.shape) == 3:                             # squeeze (batch*num_items, None, 16)
                c_embed = layers.reduce_sum(c_embed, dim=1)
            list_embed.append(c_embed)                              # (batch*num_items, 16)
        concated_embed = layers.concat(input=list_embed, axis=1)    # (batch*num_items, concat_dim)
        concated_embed = layers.softsign(concated_embed)
        return concated_embed

    def create_inputs(self, mode):
        """create layers.data here"""
        inputs = OrderedDict()
        data_attributes = copy.deepcopy(self.data_attributes)

        if mode == 'infer_init':
            list_names = self.recent_slot_names

        elif mode == 'infer_onestep':
            list_names = self.item_slot_names

        elif mode in ['train', 'test']:
            list_names = self.item_slot_names + self.recent_slot_names + self.label_slot_names

        elif mode in ['inference']:
            list_names = self.item_slot_names + self.recent_slot_names

        else:
            raise NotImplementedError(mode)
            
        for name in list_names:
            proper = self.data_attributes[name]
            inputs[name] = fluid.layers.data(name=name,
                                             shape=proper['shape'],
                                             dtype=proper['dtype'],
                                             lod_level=proper['lod_level'])

        if mode == 'infer_onestep':
            inputs['user_feature'] = fluid.layers.data(name='user_feature',
                                             shape=(-1, self.hidden_size),
                                             dtype='float32',
                                             lod_level=1)

        return inputs

    def user_encode(self, recent_embedding):
        """user encode part"""
        recent_concat_fc = self.recent_fc_op(recent_embedding)
        recent_feature_gru = self.recent_gru_op(recent_concat_fc)
        recent_feature_gru_last = layers.sequence_pool(input=recent_feature_gru, pool_type="last")
        user_feature = self.user_feature_fc_op(recent_feature_gru_last)
        return user_feature

    ###################
    ### main functions
    ###################

    def forward(self, inputs, mode):
        """forward"""
        item_embedding = self._build_embeddings(inputs, self.item_slot_names)            
        recent_embedding = self._build_embeddings(inputs, self.recent_slot_names)

        user_feature = self.user_encode(recent_embedding)   # (batch, dim)

        item_fc = self.item_fc_op(item_embedding)           # (batch*seq_len, dim)
        hidden = layers.concat([layers.sequence_expand(user_feature, item_fc),
                                item_fc], 1)

        output_dict = OrderedDict()
        if 'click' in self._output_type:
            output_dict['click_prob'] = self.out_click_fc2_op(self.out_click_fc1_op(hidden))
        if 'credit' in self._output_type:
            output_dict['credit_pred'] = self.out_credit_fc2_op(self.out_credit_fc1_op(hidden))
        if 'rate' in self._output_type:
            output_dict['rate_pred'] = self.out_rate_fc2_op(self.out_rate_fc1_op(hidden))
        return output_dict

    def infer_init(self, inputs):
        """inference only the init part"""
        recent_embedding = self._build_embeddings(inputs, self.recent_slot_names)
        user_feature = self.user_encode(recent_embedding)

        output_dict = OrderedDict()
        output_dict['user_feature'] = user_feature
        return output_dict

    def infer_onestep(self, inputs):
        """inference the gru-unit by one step"""
        user_feature = inputs['user_feature']
        item_embedding = self._build_embeddings(inputs, self.item_slot_names)            

        item_fc = self.item_fc_op(item_embedding)
        hidden = layers.concat([user_feature, item_fc], 1)

        output_dict = OrderedDict()
        if 'click' in self._output_type:
            output_dict['click_prob'] = self.out_click_fc2_op(self.out_click_fc1_op(hidden))
        if 'credit' in self._output_type:
            output_dict['credit_pred'] = self.out_credit_fc2_op(self.out_credit_fc1_op(hidden))
        if 'rate' in self._output_type:
            output_dict['rate_pred'] = self.out_rate_fc2_op(self.out_rate_fc1_op(hidden))
        return output_dict

