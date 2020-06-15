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


class UniRNN(Model):
    """
    """
    def __init__(self, conf, npz_config, scope=None, cell_type='gru', embed_regular=0.0, output_type='click', output_dim=2):
        super(UniRNN, self).__init__()
        self.conf = conf
        self.npz_config = npz_config
        self.data_attributes = conf.data_attributes
        # feature related initialization
        self.item_slot_names = conf.item_slot_names
        self.recent_slot_names = conf.recent_slot_names
        self.label_slot_names = conf.label_slot_names
        self.shared_embedding_names = conf.shared_embedding_names
        self._cell_type = cell_type
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

        if self._cell_type == 'gru':
            self.recent_fc_op = default_fc(self.hidden_size * 3, act='relu', name='recent_fc')
            self.recent_gru_op = default_drnn(self.hidden_size, name='recent_gru')
        elif self._cell_type == 'lstm':
            self.recent_fc_op = default_fc(self.hidden_size * 4, act='relu', name='recent_fc')
            self.recent_lstm_op = default_lstm(self.hidden_size * 4, name='recent_lstm')
        self.user_feature_fc_op = default_fc(self.hidden_size, act='relu', name='user_feature_fc')

        if self._cell_type == 'gru':
            self.item_fc_op = default_fc(self.hidden_size * 3, act='relu', name='item_fc')
            self.item_gru_op = default_drnn(self.hidden_size, name='item_gru')
        elif self._cell_type == 'lstm':
            self.item_fc_op = default_fc(self.hidden_size * 4, act='relu', name='item_fc')
            self.item_lstm_op = default_lstm(self.hidden_size * 4, name='item_lstm')

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
            if self._cell_type == 'gru':
                size = self.hidden_size
            elif self._cell_type == 'lstm':
                size = self.hidden_size * 2
            inputs['prev_hidden'] = fluid.layers.data(name='prev_hidden',
                                             shape=(-1, size),
                                             dtype='float32',
                                             lod_level=1)

        return inputs

    def user_encode(self, recent_embedding):
        """user encode part"""
        recent_concat_fc = self.recent_fc_op(recent_embedding)
        if self._cell_type == 'gru':
            recent_feature_gru = self.recent_gru_op(recent_concat_fc)    
        elif self._cell_type == 'lstm':
            hidden, _ = self.recent_lstm_op(recent_concat_fc)
            recent_feature_gru = hidden
        recent_feature_gru_last = layers.sequence_pool(input=recent_feature_gru, pool_type="last")
        user_feature = self.user_feature_fc_op(recent_feature_gru_last)
        return user_feature

    def _lstm_merge_hidden_cell(self, hidden, cell_state):
        return layers.concat([hidden, cell_state], 1)

    def _lstm_split_hidden_cell(self, merged_hidden_cell):
        return layers.split(merged_hidden_cell, 2, dim=1)

    ###################
    ### main functions
    ###################

    def forward(self, inputs, mode):
        """forward"""
        item_embedding = self._build_embeddings(inputs, self.item_slot_names)            
        recent_embedding = self._build_embeddings(inputs, self.recent_slot_names)

        user_feature = self.user_encode(recent_embedding)

        item_fc = self.item_fc_op(item_embedding)
        if self._cell_type == 'gru':
            hidden = self.item_gru_op(item_fc, h_0=user_feature)
        elif self._cell_type == 'lstm':
            hidden, _ = self.item_lstm_op(item_fc, h_0=user_feature, c_0=user_feature*0)

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
        if self._cell_type == 'gru':
            output_dict['user_feature'] = user_feature
        elif self._cell_type == 'lstm':
            output_dict['user_feature'] = self._lstm_merge_hidden_cell(user_feature, user_feature*0)
        return output_dict

    def infer_onestep(self, inputs):
        """inference the gru-unit by one step"""
        prev_hidden = inputs['prev_hidden']
        item_embedding = self._build_embeddings(inputs, self.item_slot_names)            

        item_fc = self.item_fc_op(item_embedding)
        if self._cell_type == 'gru':
            hidden = self.item_gru_op(item_fc, h_0=prev_hidden)
        elif self._cell_type == 'lstm':
            h_0, c_0 = self._lstm_split_hidden_cell(prev_hidden)
            hidden, cell_state = self.item_lstm_op(item_fc, h_0=h_0, c_0=c_0)

        output_dict = OrderedDict()
        if self._cell_type == 'gru':
            output_dict['hidden'] = hidden
        elif self._cell_type == 'lstm':
            output_dict['hidden'] = self._lstm_merge_hidden_cell(hidden, cell_state)
        if 'click' in self._output_type:
            output_dict['click_prob'] = self.out_click_fc2_op(self.out_click_fc1_op(hidden))
        if 'credit' in self._output_type:
            output_dict['credit_pred'] = self.out_credit_fc2_op(self.out_credit_fc1_op(hidden))
        if 'rate' in self._output_type:
            output_dict['rate_pred'] = self.out_rate_fc2_op(self.out_rate_fc1_op(hidden))
        return output_dict


class RLRNN(Model):
    """
    """
    def __init__(self, conf, npz_config):
        super(RLRNN, self).__init__()
        self.conf = conf
        self.npz_config = npz_config
        self.data_attributes = conf.data_attributes
        # feature related initialization
        self.item_slot_names = conf.item_slot_names
        self.recent_slot_names = conf.recent_slot_names
        self.label_slot_names = conf.label_slot_names
        self.shared_embedding_names = conf.shared_embedding_names
        self.hidden_size = 32
        self.BIG_VALUE = 1e6

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
        for name in list_names:
            embed_name = self._get_embed_name(name)
            vob_size = self.npz_config['embedding_size'][embed_name] + 1
            embed_size = 16
            if embed_name not in self.dict_data_embed_op:
                self.dict_data_embed_op[embed_name] = \
                        default_embedding([vob_size, embed_size], 'embed_' + embed_name, embed_clip=None)

        self.recent_fc_op = default_fc(self.hidden_size * 3, act='relu', name='recent_fc')
        self.recent_gru_op = default_drnn(self.hidden_size, name='recent_gru')
        self.user_feature_fc_op = default_fc(self.hidden_size, act='relu', name='user_feature_fc')

        self.item_fc_op = default_fc(self.hidden_size * 3, act='relu', name='item_fc')
        self.item_gru_op = default_drnn(self.hidden_size, name='item_gru')

        self.out_Q_fc1_op = default_fc(self.hidden_size * 2, act='relu', name='out_Q_fc1_op')
        self.out_Q_fc2_op = default_fc(1, act=None, name='out_Q_fc2_op')

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
            list_embed.append(c_embed)                              # (batch*num_items, 16)
        concated_embed = layers.concat(input=list_embed, axis=1)    # (batch*num_items, concat_dim)
        concated_embed = layers.softsign(concated_embed)
        return concated_embed

    def eps_greedy_sampling(self, scores, mask, eps):
        scores = scores * mask
        scores_padded = layers.squeeze(fluid_sequence_pad(scores, 0, maxlen=128), [2])  # (b*s, 1) -> (b, s, 1) -> (b, s)
        mask_padded = layers.squeeze(fluid_sequence_pad(mask, 0, maxlen=128), [2])
        seq_lens = fluid_sequence_get_seq_len(scores)

        def get_greedy_prob(scores_padded, mask_padded):
            s = scores_padded - (mask_padded*(-1) + 1) * self.BIG_VALUE
            max_value = layers.reduce_max(s, dim=1, keep_dim=True)
            greedy_prob = layers.cast(s >= max_value, 'float32')
            return greedy_prob
        greedy_prob = get_greedy_prob(scores_padded, mask_padded)
        eps_prob = mask_padded * eps / layers.reduce_sum(mask_padded, dim=1, keep_dim=True)

        final_prob = (greedy_prob + eps_prob) * mask_padded
        final_prob = final_prob / layers.reduce_sum(final_prob, dim=1, keep_dim=True)

        sampled_id = layers.reshape(layers.sampling_id(final_prob), [-1, 1])
        max_id = layers.cast(layers.cast(seq_lens, 'float32') - 1, 'int64')
        sampled_id = layers.elementwise_min(sampled_id, max_id)
        return layers.cast(sampled_id, 'int64')

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
            inputs['prev_hidden'] = fluid.layers.data(name='prev_hidden',
                                             shape=(-1, self.hidden_size),
                                             dtype='float32',
                                             lod_level=1)

        return inputs

    def user_encode(self, recent_embedding):
        """user encode part"""
        recent_concat_fc = self.recent_fc_op(recent_embedding)
        recent_feature_gru = self.recent_gru_op(recent_concat_fc)
        recent_feature_gru_last = layers.sequence_pool(input=recent_feature_gru, pool_type="last")
        lod_reference = fluid_sequence_first_step(recent_feature_gru)
        lod_reference.stop_gradient = True
        recent_feature_gru_last = layers.lod_reset(recent_feature_gru_last, lod_reference)
        user_feature = self.user_feature_fc_op(recent_feature_gru_last)
        return user_feature

    def dynamic_rnn(self, item_fc, h_0, output_type=None, double_type=None, double_id=None):
        drnn = fluid.layers.DynamicRNN()
        pos = fluid_sequence_get_pos(item_fc)
        with drnn.block():
            cur_item_fc = drnn.step_input(item_fc)
            cur_h_0 = drnn.memory(init=h_0, need_reorder=True)

            cur_item_fc = layers.lod_reset(cur_item_fc, cur_h_0)
            next_h_0 = self.simple_step_rnn(cur_item_fc, h_0=cur_h_0)

            if output_type == 'c_Q':
                Q = self.out_Q_fc2_op(self.out_Q_fc1_op(next_h_0))
                drnn.output(Q)

            elif output_type in ['max_Q', 'double_Q']:
                # batch_size = 2
                # item_fc: lod = [0,4,7]
                # cur_h_0: lod = [0,1,2]
                item_fc = drnn.static_input(item_fc)
                pos = drnn.static_input(pos)
                cur_step = drnn.memory(shape=[1], dtype='int64', value=0)

                expand_h_0 = layers.sequence_expand(cur_h_0, item_fc)               # lod = [0,1,2,3,4,5,6,7]
                new_item_fc = layers.lod_reset(item_fc, expand_h_0)                 # lod = [0,1,2,3,4,5,6,7]
                next_expand_h_0 = self.simple_step_rnn(new_item_fc, expand_h_0)     # lod = [0,1,2,3,4,5,6,7]
                next_expand_h_0 = layers.lod_reset(next_expand_h_0, item_fc)        # lod = [0,4,7]

                expand_Q = self.out_Q_fc2_op(self.out_Q_fc1_op(next_expand_h_0))
                cur_step_id = layers.slice(cur_step, axes=[0, 1], starts=[0, 0], ends=[1, 1])
                mask = layers.cast(pos >= cur_step_id, 'float32')
                expand_Q = expand_Q * mask

                if output_type == 'max_Q':
                    max_Q = layers.sequence_pool(expand_Q, 'max')                       # lod = [0,1,2]
                    drnn.output(max_Q)
                elif output_type == 'double_Q':
                    if double_type == 'max_id':
                        max_id = self.eps_greedy_sampling(expand_Q, mask, eps=0)
                        drnn.output(max_id)
                    elif double_type == 'double_Q':
                        cur_double_id = drnn.step_input(double_id)

                        double_Q = fluid_sequence_index(expand_Q, cur_double_id)
                        drnn.output(double_Q)

                # update
                next_step = cur_step + 1
                drnn.update_memory(cur_step, next_step)

            elif output_type == 'hidden':
                drnn.output(next_h_0)                

            else:
                raise NotImplementedError(output_type)

            # update
            drnn.update_memory(cur_h_0, next_h_0)

        drnn_output = drnn()
        return drnn_output

    def simple_step_rnn(self, item_fc, h_0):
        """
        The same as self.dynamic_rnn(item_fc, h_0, output_type='hidden') for a single step
        """
        next_h_0 = self.item_gru_op(item_fc, h_0=h_0)
        return next_h_0

    ###################
    ### main functions
    ###################

    def forward(self, inputs, output_type, double_type=None, double_id=None):
        """forward"""
        assert output_type in ['c_Q', 'max_Q', 'double_Q']

        item_embedding = self._build_embeddings(inputs, self.item_slot_names)            
        recent_embedding = self._build_embeddings(inputs, self.recent_slot_names)

        user_feature = self.user_encode(recent_embedding)

        item_fc = self.item_fc_op(item_embedding)
        item_Q = self.dynamic_rnn(item_fc, 
                                  h_0=user_feature, 
                                  output_type=output_type, 
                                  double_type=double_type, 
                                  double_id=double_id)

        output_dict = OrderedDict()
        output_dict['Q'] = item_Q
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
        prev_hidden = inputs['prev_hidden']
        item_embedding = self._build_embeddings(inputs, self.item_slot_names)

        item_fc = self.item_fc_op(item_embedding)
        # item_hidden = self.dynamic_rnn(item_fc, h_0=prev_hidden, output_type='hidden')
        item_hidden = self.simple_step_rnn(item_fc, h_0=prev_hidden)

        output_dict = OrderedDict()
        output_dict['hidden'] = item_hidden
        output_dict['Q'] = self.out_Q_fc2_op(self.out_Q_fc1_op(item_hidden))
        return output_dict




