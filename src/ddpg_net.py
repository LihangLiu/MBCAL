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
                         fluid_sequence_pad, fluid_sequence_get_seq_len, fluid_sequence_advance)

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


class DDPGRNN(Model):
    """
    """
    def __init__(self, conf, npz_config):
        super(DDPGRNN, self).__init__()
        self.conf = conf
        self.npz_config = npz_config
        self.data_attributes = conf.data_attributes
        # feature related initialization
        self.item_slot_names = list(conf.item_slot_names)
        self.item_slot_names.remove('last_click_id')
        self.last_click_slot_names = ['last_click_id']
        self.last_item_slot_names = ['last_'+name for name in self.item_slot_names]
        self.recent_slot_names = conf.recent_slot_names
        self.label_slot_names = conf.label_slot_names

        self.shared_embedding_names = conf.shared_embedding_names
        for name in self.item_slot_names:
            self.shared_embedding_names['last_'+name] = name

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
        list_names = self.item_slot_names + self.last_click_slot_names + self.recent_slot_names
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

        self.item_fc_op = default_fc(self.hidden_size, act='tanh', name='item_fc')
        self.item_gru_fc_op = default_fc(self.hidden_size * 3, act='relu', name='item_gru_fc')
        self.item_gru_op = default_drnn(self.hidden_size, name='item_gru')

        self.actor_fc1_op = default_fc(self.hidden_size, act='relu', name='actor_fc1')
        self.actor_fc2_op = default_fc(self.hidden_size, act='tanh', name='actor_fc2')      # tanh: try to prevent actor loss go to infinity.
        self.critic_fc1_op = default_fc(self.hidden_size, act='relu', name='critic_fc1')
        self.critic_fc2_op = default_fc(self.hidden_size, act='relu', name='critic_fc2')
        self.critic_fc3_op = default_fc(1, act=None, name='critic_fc3')

        self.actor_ops = []
        # self.actor_ops += list(self.dict_data_embed_op.values())      # will crash
        # self.actor_ops.append(self.recent_fc_op)
        # self.actor_ops.append(self.recent_gru_op)
        # self.actor_ops.append(self.user_feature_fc_op)
        # self.actor_ops.append(self.item_fc_op)
        # self.actor_ops.append(self.item_gru_fc_op)
        # self.actor_ops.append(self.item_gru_op)
        self.actor_ops.append(self.actor_fc1_op)
        self.actor_ops.append(self.actor_fc2_op)
        self.actor_param_names = []
        for op in self.actor_ops:
            self.actor_param_names.append(op.param_name)
            if not op.bias_name is None:
                self.actor_param_names.append(op.bias_name)
        print('actor_param_names', self.actor_param_names)

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
        # concated_embed = layers.softsign(concated_embed)
        return concated_embed

    def create_inputs(self, mode):
        """create layers.data here"""
        inputs = OrderedDict()
        data_attributes = copy.deepcopy(self.data_attributes)

        if mode == 'infer_init':
            list_names = self.recent_slot_names

        elif mode == 'infer_onestep':
            for last_name in self.last_item_slot_names:
                name = last_name[len('last_'):]
                data_attributes[last_name] = copy.deepcopy(data_attributes[name])
            list_names = self.item_slot_names +\
                         self.last_click_slot_names +\
                         self.last_item_slot_names

        elif mode in ['train', 'test']:
            list_names = self.item_slot_names +\
                         self.last_click_slot_names +\
                         self.recent_slot_names +\
                         self.label_slot_names

        else:
            raise NotImplementedError(mode)
            
        for name in list_names:
            proper = data_attributes[name]
            inputs[name] = fluid.layers.data(name=name,
                                             shape=proper['shape'],
                                             dtype=proper['dtype'],
                                             lod_level=proper['lod_level'])

        if mode == 'infer_onestep':
            inputs['prev_hidden'] = fluid.layers.data(name='prev_hidden',
                                             shape=(-1, self.hidden_size),
                                             dtype='float32',
                                             lod_level=1)
            # use first_step_mask to change last_item_fc to zeros
            inputs['first_step_mask'] = fluid.layers.data(name='first_step_mask',
                                             shape=(-1, 1),
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

    def train_rnn(self, item_fc, last_click_embedding, h_0, output_type=''):
        """
        BUG: will crash.
        """
        shifted_item_fc = fluid_sequence_advance(item_fc, OOV=0)
        drnn = fluid.layers.DynamicRNN()
        with drnn.block():
            last_item_fc = drnn.step_input(shifted_item_fc)
            last_click_embedding = drnn.step_input(last_click_embedding)
            cur_h_0 = drnn.memory(init=h_0, need_reorder=True)

            # step_input will remove lod info
            last_item_fc = layers.lod_reset(last_item_fc, cur_h_0)
            last_click_embedding = layers.lod_reset(last_click_embedding, cur_h_0)

            next_h_0 = self.simple_step_rnn(last_item_fc, last_click_embedding, h_0=cur_h_0)

            if output_type == 'c_Q':
                cur_item_fc = drnn.step_input(item_fc)
                cur_item_fc = layers.lod_reset(cur_item_fc, cur_h_0)
                Q = self.critic_value(next_h_0, cur_item_fc)
                drnn.output(Q)

            elif output_type == 'max_Q':
                action_hat = self.actor_policy(next_h_0)
                max_Q = self.critic_value(next_h_0, action_hat)
                drnn.output(max_Q)

            else:
                raise NotImplementedError(output_type)

            # update
            drnn.update_memory(cur_h_0, next_h_0)

        drnn_output = drnn()
        return drnn_output

    def train_rnn2(self, item_fc, last_click_embedding, h_0, output_type=''):
        shifted_item_fc = fluid_sequence_advance(item_fc, OOV=0)
        next_h_0 = self.simple_step_rnn(shifted_item_fc, last_click_embedding, h_0=h_0)  
        if output_type == 'c_Q':
            Q = self.critic_value(next_h_0, item_fc)
            return Q
        elif output_type == 'max_Q':
            action_hat = self.actor_policy(next_h_0)
            max_Q = self.critic_value(next_h_0, action_hat)
            return max_Q

    def actor_policy(self, state):
        action_hat = self.actor_fc2_op(self.actor_fc1_op(state))
        return action_hat

    def critic_value(self, state, action):
        Q = self.critic_fc3_op(self.critic_fc2_op(self.critic_fc1_op(layers.concat([state, action], 1))))
        return Q

    def simple_step_rnn(self, item_fc, last_click_embedding, h_0):
        """
        The same as self.train_rnn(item_fc, h_0, output_type='hidden') for a single step
        """
        input_fc = self.item_gru_fc_op(layers.concat([item_fc, last_click_embedding], 1))
        next_h_0 = self.item_gru_op(input_fc, h_0=h_0)
        return next_h_0

    ###################
    ### main functions
    ###################

    def forward(self, inputs, output_type):
        """forward"""
        assert output_type in ['c_Q', 'max_Q']

        item_embedding = self._build_embeddings(inputs, self.item_slot_names)            
        last_click_embedding = self._build_embeddings(inputs, self.last_click_slot_names)            
        recent_embedding = self._build_embeddings(inputs, self.recent_slot_names)

        user_feature = self.user_encode(recent_embedding)

        item_fc = self.item_fc_op(item_embedding)
        item_Q = self.train_rnn2(item_fc, 
                                last_click_embedding,
                                h_0=user_feature, 
                                output_type=output_type)

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
        first_step_mask = inputs['first_step_mask']
        item_embedding = self._build_embeddings(inputs, self.item_slot_names)   # (b*cand_len, dim), as candidates
        last_click_embedding = self._build_embeddings(inputs, self.last_click_slot_names) # (b, dim)
        last_item_embedding = self._build_embeddings(inputs, self.last_item_slot_names) # (b, dim)

        item_fc = self.item_fc_op(item_embedding)
        last_item_fc = self.item_fc_op(last_item_embedding) * first_step_mask
        item_hidden = self.simple_step_rnn(last_item_fc, last_click_embedding, h_0=prev_hidden)
        action_hat = self.actor_policy(item_hidden)

        # inner product
        expand_action_hat = layers.sequence_expand(action_hat, item_fc)     # (b*cand_len, dim)
        scores = layers.reduce_sum(expand_action_hat * item_fc, 1)

        output_dict = OrderedDict()
        output_dict['hidden'] = item_hidden
        output_dict['scores'] = scores
        return output_dict




