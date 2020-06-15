"""
Defines the optimizer and the network outputs
"""
#!/usr/bin/env python
# coding=utf8

import os
from os.path import exists
import sys
from collections import OrderedDict
import numpy as np
from copy import deepcopy
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

PARL_DIR = os.environ.get('PARL_DIR', '')
assert exists(PARL_DIR), ('PARL_DIR', PARL_DIR, 'not exists')
sys.path.append(PARL_DIR)

from paddle import fluid
import parl.layers as layers
from parl.framework.algorithm import Algorithm


class CFAlgorithm(Algorithm):
    """
    For generation tasks
    return:
        {
            feed_names: list of ordered feed names, optional.
            fetch_dict: orderedDict of fetch vars
        }
    """
    def __init__(self, model, optimizer, lr=None, hyperparas=None, gpu_id=-1, credit_scale=None):
        hyperparas = {} if hyperparas is None else hyperparas
        super(CFAlgorithm, self).__init__(model, hyperparas=hyperparas, gpu_id=gpu_id)
        # self.target_model = deepcopy(model)

        self.optimizer = optimizer
        self.lr = lr
        self._learn_cnt = 0
        self.gpu_id = gpu_id
        self._credit_scale = credit_scale

        self._safe_eps = 1e-5
        self._output_type = self.model._output_type
        self._rate_scale = 0.1

    def train(self):
        """train"""
        inputs = self.model.create_inputs(mode='train')
        output_dict = self.model.forward(inputs, mode='train')

        total_loss = 0
        if 'click' in self._output_type:
            click_id = inputs['click_id']
            click_prob = output_dict['click_prob']
            click_loss = layers.reduce_mean(layers.cross_entropy(input=click_prob, label=click_id))
            total_loss += click_loss
        if 'credit' in self._output_type:
            credit = inputs['credit'] * self._credit_scale
            credit_pred = output_dict['credit_pred']
            credit_loss = layers.reduce_mean(layers.square_error_cost(input=credit_pred, label=credit))
            total_loss += credit_loss
        if 'rate' in self._output_type:
            rate = layers.cast(inputs['click_id'], 'float32') * self._rate_scale
            rate_pred = output_dict['rate_pred']
            rate_loss = layers.reduce_mean(layers.square_error_cost(input=rate_pred, label=rate))
            total_loss += rate_loss

        if self.optimizer == 'Adam':
            optimizer = fluid.optimizer.Adam(learning_rate=self.lr, epsilon=1e-4)
        elif self.optimizer == 'SGD':
            optimizer = fluid.optimizer.SGD(learning_rate=self.lr)
        optimizer.minimize(total_loss)

        fetch_dict = OrderedDict()
        fetch_dict['loss'] = total_loss             # don't rename 'loss', which will be used in parallel exe in computational task
        if 'click' in self._output_type:
            fetch_dict['click_prob'] = click_prob
            fetch_dict['click_id'] = click_id
            fetch_dict['click_loss'] = click_loss
        if 'credit' in self._output_type:
            fetch_dict['credit_pred'] = credit_pred / self._credit_scale
            fetch_dict['credit'] = credit / self._credit_scale
            fetch_dict['credit_loss'] = credit_loss
        if 'rate' in self._output_type:
            fetch_dict['rate_pred'] = rate_pred / self._rate_scale
            fetch_dict['rate'] = rate / self._rate_scale
            fetch_dict['rate_loss'] = rate_loss
        return {'fetch_dict': fetch_dict}

    def test(self):
        """test"""
        inputs = self.model.create_inputs(mode='test')
        output_dict = self.model.forward(inputs, mode='test')

        fetch_dict = OrderedDict()
        if 'click' in self._output_type:
            fetch_dict['click_prob'] = output_dict['click_prob']
            fetch_dict['click_id'] = inputs['click_id'] + layers.reduce_mean(output_dict['click_prob']) * 0     # IMPORTANT!!! equals to label = label, otherwise parallel executor won't get this variable
        if 'credit' in self._output_type:
            fetch_dict['credit_pred'] = output_dict['credit_pred'] / self._credit_scale
            fetch_dict['credit'] = inputs['credit'] + layers.reduce_mean(output_dict['credit_pred']) * 0
        if 'rate' in self._output_type:
            fetch_dict['rate_pred'] = output_dict['rate_pred'] / self._rate_scale
            fetch_dict['rate'] = layers.cast(inputs['click_id'], 'float32') \
                                 + layers.reduce_mean(output_dict['rate_pred']) * 0
        return {'fetch_dict': fetch_dict}

    def inference(self):
        """inference"""
        inputs = self.model.create_inputs(mode='inference')
        output_dict = self.model.forward(inputs, mode='inference')

        fetch_dict = OrderedDict()
        if 'click' in self._output_type:
            fetch_dict['click_prob'] = output_dict['click_prob']
        if 'credit' in self._output_type:
            fetch_dict['credit_pred'] = output_dict['credit_pred'] / self._credit_scale
        if 'rate' in self._output_type:
            fetch_dict['rate_pred'] = output_dict['rate_pred'] / self._rate_scale
        return {'fetch_dict': fetch_dict}

    def infer_init(self):
        """inference only the init part"""
        inputs = self.model.create_inputs(mode='infer_init')
        output_dict = self.model.infer_init(inputs)

        fetch_dict = OrderedDict()
        fetch_dict['user_feature'] = output_dict['user_feature']
        return {'feed_names': inputs.keys(),
                'fetch_dict': fetch_dict}

    def infer_onestep(self):
        """inference the gru-unit by one step"""
        inputs = self.model.create_inputs(mode='infer_onestep')
        output_dict = self.model.infer_onestep(inputs)
        # click_prob = layers.slice(click_prob, axes=[1], starts=[1], ends=[2])

        fetch_dict = OrderedDict()
        if 'click' in self._output_type:
            fetch_dict['click_prob'] = output_dict['click_prob']
        if 'credit' in self._output_type:
            fetch_dict['credit_pred'] = output_dict['credit_pred'] / self._credit_scale
        if 'rate' in self._output_type:
            fetch_dict['rate_pred'] = output_dict['rate_pred'] / self._rate_scale
        return {'feed_names': inputs.keys(),
                'fetch_dict': fetch_dict}





