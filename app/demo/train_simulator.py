"""
train the simulator
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import scipy.stats as stats
import sys
import math
import random
import copy
import time
import datetime
import os
from os.path import basename, join, exists, dirname
from sklearn.metrics import roc_auc_score
from multiprocessing import Manager
from threading import Thread, Lock
import argparse
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s') # filename='hot_rl.log', 

import tensorflow as tf

import paddle
from paddle import fluid
import paddle.fluid.profiler as profiler

from config_sim import Config_Sim
from utils import BatchData, add_scalar_summary, click_prob_2_score, click_2_last_click

import _init_paths

from src.credit_assignment_net import UniRNN
from src.gen_algorithm import GenAlgorithm
from src.gen_computation_task import GenComputationTask

from src.utils import (read_json, print_args, tik, tok, threaded_generator, print_once,
                        RMSEMetrics, AccuracyMetrics, AssertEqual)
from src.fluid_utils import (fluid_create_lod_tensor as create_tensor, 
                            concat_list_array, seq_len_2_lod, get_num_devices)
from data.npz_dataset import NpzDataset, FakeTensor

#########
# utils
#########

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', help="Exp id, used for logs or savings")
    parser.add_argument('--use_cuda', default = 1, type = int, help = "")
    parser.add_argument('--train_mode', 
                        default = 'single', 
                        choices = ['single', 'parallel'],
                        type = str, 
                        help = "single: use the first gpu, parallel: use all gpus")
    parser.add_argument('--task', 
                        default = 'train', 
                        choices = ['train', 'test', 'debug', 'generate_credit', 'eval_list'],
                        type = str, 
                        help = "")
    
    # model settings
    parser.add_argument('--cell_type', type=str, choices=['gru', 'lstm'], default='gru', help='')
    parser.add_argument('--output_dim', type=int, default=6, help='for output_type=click')

    # dataset
    parser.add_argument('--train_npz_list', type=str, default='', help='')
    parser.add_argument('--test_npz_list', type=str, default='', help='')
    return parser


def get_ct_sim(exp, use_cuda, train_mode, cell_type, output_dim):
    conf = Config_Sim(exp)
    npz_config = read_json(conf.npz_config_path)
    scope = fluid.Scope()
    model = UniRNN(conf, npz_config, 
                   scope=scope, 
                   cell_type=cell_type,
                   output_type='click', 
                   output_dim=output_dim)
    algorithm = GenAlgorithm(model, optimizer=conf.optimizer, lr=conf.lr, gpu_id=(0 if use_cuda else -1))
    ct = GenComputationTask(algorithm, model_dir=conf.model_dir, mode=train_mode)
    return ct


class SimFeedConvertor(object):
    @staticmethod
    def train_test(batch_data):
        # # add last_click_id
        # batch_size = batch_data.batch_size()
        # click_id = batch_data.get_values('click_id')
        # last_click_id = click_2_last_click(click_id.reshape([batch_size, -1]))
        # batch_data.tensor_dict['last_click_id'] = FakeTensor(last_click_id.reshape(click_id.shape), 
        #                                                     batch_data.get_seq_lens('click_id'))
        batch_data.add_last_click_id()

        place = fluid.CPUPlace()
        feed_dict = {}
        for name in batch_data.conf.recent_slot_names + \
                    batch_data.conf.item_slot_names + \
                    batch_data.conf.label_slot_names:
            ft = batch_data.tensor_dict[name]
            feed_dict[name] = create_tensor(ft.values, lod=ft.lod, place=place)
        return feed_dict

    @staticmethod
    def inference(batch_data):
        batch_data.add_last_click_id()

        place = fluid.CPUPlace()
        feed_dict = {}
        for name in batch_data.conf.recent_slot_names + \
                    batch_data.conf.item_slot_names:
            ft = batch_data.tensor_dict[name]
            feed_dict[name] = create_tensor(ft.values, lod=ft.lod, place=place)
        return feed_dict

    @staticmethod
    def infer_init(batch_data):
        place = fluid.CPUPlace()
        feed_dict = {}
        for name in batch_data.conf.recent_slot_names:
            ft = batch_data.tensor_dict[name]
            feed_dict[name] = create_tensor(ft.values, lod=ft.lod, place=place)
        return feed_dict

    @staticmethod
    def infer_onestep(batch_data, prev_hidden, selected_items, last_click_id):
        """
        batch_data: the complete data
            item_feature: (b*seq_len,)
        prev_hidden: (b, dim)
        last_click_id: (b,)
        selected_items: (b,)
        """
        batch_size = batch_data.batch_size()
        seq_len = batch_data.seq_lens()[0]
        batch_offset = batch_data.offset()

        place = fluid.CPUPlace()
        feed_dict = {}
        lod = [seq_len_2_lod([1] * batch_size)]
        offset_selected_items = np.array(selected_items) + batch_offset    # (b,)
        for name in batch_data.conf.item_slot_names:
            if name == 'last_click_id':
                v = last_click_id.reshape([-1, 1])
            else:
                v = batch_data.tensor_dict[name].values[offset_selected_items]
            feed_dict[name] = create_tensor(v, lod=lod, place=place)

        # prev_hidden
        feed_dict['prev_hidden'] = create_tensor(prev_hidden, lod=lod, place=place)
        return feed_dict


############
# main
############

def main(args):
    print_args(args, 'args')
    ct = get_ct_sim(args.exp, 
                  args.use_cuda==1, 
                  args.train_mode, 
                  args.cell_type, 
                  args.output_dim)
    conf = ct.alg.model.conf

    ###########
    ### other tasks
    ###########
    if args.task == 'test':
        test(ct, args, conf, None, ct.ckp_step)
        exit()
    elif args.task == 'eval_list':
        return eval_list(ct, args, conf, ct.ckp_step, args.eval_npz_list)

    ##################
    ### start training
    ##################
    summary_writer = tf.summary.FileWriter(conf.summary_dir)
    for epoch_id in range(ct.ckp_step + 1, conf.max_train_steps + 1):
        train(ct, args, conf, summary_writer, epoch_id)
        ct.save_model(epoch_id)
        test(ct, args, conf, summary_writer, epoch_id)


def train(ct, args, conf, summary_writer, epoch_id):
    """train for conf.train_interval steps"""
    dataset = NpzDataset(args.train_npz_list, 
                        conf.npz_config_path, 
                        conf.requested_names,
                        if_random_shuffle=True)
    data_gen = dataset.get_data_generator(conf.batch_size)

    list_loss = []
    list_epoch_loss = []
    for batch_id, tensor_dict in enumerate(threaded_generator(data_gen, capacity=100)):
        batch_data = BatchData(conf, tensor_dict)
        fetch_dict = ct.train(SimFeedConvertor.train_test(batch_data))
        list_loss.append(np.array(fetch_dict['loss']))
        list_epoch_loss.append(np.mean(np.array(fetch_dict['loss'])))
        if batch_id % conf.prt_interval == 0:
            logging.info('batch_id:%d loss:%f' % (batch_id, np.mean(list_loss)))
            list_loss = []

    add_scalar_summary(summary_writer, epoch_id, 'train/loss', np.mean(list_epoch_loss))


def test(ct, args, conf, summary_writer, epoch_id, item_shuffle=False):
    """eval auc on the full test dataset"""
    dataset = NpzDataset(args.test_npz_list, 
                        conf.npz_config_path, 
                        conf.requested_names,
                        if_random_shuffle=True)
    data_gen = dataset.get_data_generator(conf.batch_size)

    click_rmse_metric = RMSEMetrics()
    click_accu_metric = AccuracyMetrics()
    for batch_id, tensor_dict in enumerate(threaded_generator(data_gen, capacity=100)):
        batch_data = BatchData(conf, tensor_dict)
        fetch_dict = ct.train(SimFeedConvertor.train_test(batch_data))
        click_id = np.array(fetch_dict['click_id']).flatten()
        click_score = click_prob_2_score(np.array(fetch_dict['click_prob'])).flatten()
        click_rmse_metric.add(labels=click_id, preds=click_score)
        click_accu_metric.add(labels=click_id, probs=np.array(fetch_dict['click_prob']))

    add_scalar_summary(summary_writer, epoch_id, 'test/click_rmse', click_rmse_metric.overall_rmse())
    add_scalar_summary(summary_writer, epoch_id, 'test/click_accuracy', click_accu_metric.overall_accuracy())
    for key, value in click_accu_metric.overall_metrics().items():
        add_scalar_summary(summary_writer, epoch_id, 'test/%s' % key, value)


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args) 


