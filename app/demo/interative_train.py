"""

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
import collections
import os
from os.path import basename, join, exists, dirname
import argparse
import warnings
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s') # filename='hot_rl.log', 

import tensorflow as tf

import paddle
from paddle import fluid
from paddle.fluid import profiler

from config_env import Config_Env
from train_simulator import get_ct_sim, SimFeedConvertor
from utils import BatchData, click_prob_2_score, add_scalar_summary
from gen_utils import GenFeedConvertor, EnvFeedConvertor, CreditFeedConvertor, RLFeedConvertor
from gen_utils import DDPGFeedConvertor, CFFeedConvertor

import _init_paths

from src.credit_assignment_net import UniRNN, RLRNN
from src.ddpg_net import DDPGRNN
from src.cf_net import NeuralCF
from src.gen_algorithm import GenAlgorithm
from src.rl_algorithm import RLAlgorithm
from src.ddpg_algorithm import DDPGAlgorithm
from src.cf_algorithm import CFAlgorithm
from src.gen_computation_task import GenComputationTask
from src.rl_computation_task import RLComputationTask

from src.utils import (read_json, print_args, tik, tok, save_pickle, threaded_generator, 
                        AUCMetrics, AssertEqual, AccuracyMetrics)
from src.fluid_utils import (fluid_create_lod_tensor as create_tensor, 
                            concat_list_array, seq_len_2_lod, get_num_devices)
from data.npz_dataset import NpzDataset, FakeTensor


#########
# utils
#########

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_cuda', default = 1, type = int, help = "")
    parser.add_argument('--train_mode', 
                        default = 'single', 
                        choices = ['single', 'parallel'],
                        type = str, 
                        help = "single: use the first gpu, parallel: use all gpus")
    parser.add_argument('--task', 
                        choices = ['online', 'batchrl', 'credit_variance'],
                        type = str, 
                        help = "")

    ### model settings
    # simulator
    parser.add_argument('--sim_exp', type=str, help='')
    parser.add_argument('--sim_cell_type', choices=['gru', 'lstm'], help='')
    parser.add_argument('--output_dim', type=int, help='')
    parser.add_argument('--gen_type', choices=['env', 'env_credit', 'rl', 'cf', 'env_rl', 'mc_credit', 'ddpg'], help='')
    parser.add_argument('--infer_eps', type=float, default=0.1, help='eps-greedy used in online_inference')
    # environment
    parser.add_argument('--env_exp', type=str, help='')
    parser.add_argument('--env_output_type', choices=['click', 'rate'], help='')
    parser.add_argument('--env_item_dropout_rate', type=float, default=0.2, help='')
    # credit
    parser.add_argument('--credit_exp', type=str, help='')
    parser.add_argument('--credit_type', choices=['gt_base', 'gt', 'follow_click', 'gt_globbase'], help='')
    parser.add_argument('--credit_gamma', type=float, help='')
    parser.add_argument('--credit_scale', type=float, help='')
    # rl
    parser.add_argument('--rl_exp', type=str, help='')
    parser.add_argument('--rl_gamma', type=float, help='')
    parser.add_argument('--rl_Q_type', type=str, choices=['Q_learning', 'SARSA', 'double_Q'], help='')
    # ddpg
    parser.add_argument('--ddpg_exp', type=str, help='')
    parser.add_argument('--ddpg_gamma', type=float, help='')
    # cf
    parser.add_argument('--cf_exp', type=str, help='')
    parser.add_argument('--cf_output_type', choices=['click', 'rate'], help='')

    ### dataset
    parser.add_argument('--train_npz_list', type=str, help='')
    parser.add_argument('--test_npz_list', type=str, help='')

    ### log dir
    parser.add_argument('--summary_dir', type=str, help='')    
    return parser


def get_ct_env(exp, use_cuda, train_mode, output_type, output_dim):
    conf = Config_Env(exp)
    npz_config = read_json(conf.npz_config_path)
    scope = fluid.Scope()
    model = UniRNN(conf, npz_config, 
                   scope=scope, 
                   cell_type='gru',
                   output_type=output_type, 
                   output_dim=output_dim)
    algorithm = GenAlgorithm(model, optimizer=conf.optimizer, lr=conf.lr, gpu_id=(0 if use_cuda else -1))
    ct = GenComputationTask(algorithm, model_dir=conf.model_dir, mode=train_mode)
    return ct


def get_ct_credit(exp, use_cuda, train_mode, credit_scale):
    conf = Config_Env(exp, label_type='credit')
    npz_config = read_json(conf.npz_config_path)
    scope = fluid.Scope()
    model = UniRNN(conf, npz_config, 
                   scope=scope, 
                   cell_type='gru',
                   output_type='credit')
    algorithm = GenAlgorithm(model, 
                            optimizer=conf.optimizer, 
                            lr=conf.lr, 
                            gpu_id=(0 if use_cuda else -1), 
                            credit_scale=credit_scale)
    ct = GenComputationTask(algorithm, model_dir=conf.model_dir, mode=train_mode)
    return ct


def get_ct_rl(exp, use_cuda, train_mode, gamma, Q_type):
    conf = Config_Env(exp)
    npz_config = read_json(conf.npz_config_path)
    scope = fluid.Scope()
    with fluid.scope_guard(scope):
        with fluid.unique_name.guard():
            model = RLRNN(conf, npz_config)
            algorithm = RLAlgorithm(model,
                                    optimizer=conf.optimizer, 
                                    lr=conf.lr,
                                    gpu_id=(0 if use_cuda == 1 else -1),
                                    gamma=gamma,
                                    Q_type=Q_type)
    ct = RLComputationTask(algorithm, model_dir=conf.model_dir, mode=train_mode, scope=scope)
    return ct


def get_ct_ddpg(exp, use_cuda, train_mode, gamma):
    conf = Config_Env(exp)
    npz_config = read_json(conf.npz_config_path)
    scope = fluid.Scope()
    with fluid.scope_guard(scope):
        with fluid.unique_name.guard():
            model = DDPGRNN(conf, npz_config)
            algorithm = DDPGAlgorithm(model,
                                    optimizer=conf.optimizer, 
                                    lr=conf.lr,
                                    gpu_id=(0 if use_cuda == 1 else -1),
                                    gamma=gamma)
    ct = RLComputationTask(algorithm, model_dir=conf.model_dir, mode=train_mode, scope=scope)
    return ct


def get_ct_cf(exp, use_cuda, train_mode, output_type, output_dim):
    conf = Config_Env(exp)
    npz_config = read_json(conf.npz_config_path)
    scope = fluid.Scope()
    model = NeuralCF(conf, npz_config, 
                   scope=scope, 
                   output_type=output_type, 
                   output_dim=output_dim)
    algorithm = CFAlgorithm(model, optimizer=conf.optimizer, lr=conf.lr, gpu_id=(0 if use_cuda else -1))
    ct = GenComputationTask(algorithm, model_dir=conf.model_dir, mode=train_mode)
    return ct


################
# for one batch
################

def inference_one_batch(gen_type, ct_sim, dict_gen_ct, batch_data, eps):
    """
    Do inference for one batch `batch_data`

    dict_gen_ct: {
        'env': ct_env,
        'credit': ct_credit,
        'rl': ct_rl,
        'cf': ct_cf,
    }
    gen_type='env': 'env'
    gen_type='env_credit': 'env', 'credit'
    gen_type='rl': 'rl'
    gen_type='cf': 'cf'

    For 'env', (prev_hidden, item_feature) -> infer_onestep -> (next_hidden, scores)
    For 'cf', (user_feature, item_feature) -> infer_onestep -> (user_feature, scores)
    """
    def eps_greedy(x, eps):
        """
        x: (batch_size, len)
        eps: prob of random indice
        selected_indice: (batch_size,)
        """
        n1, n2 = x.shape
        max_indice = np.argmax(x, 1)
        random_indice = np.random.randint(n2, size=[n1])
        p_random = np.random.binomial(1, eps, size=[n1])
        selected_indice = p_random * random_indice + (1 - p_random) * max_indice
        return selected_indice

    def sampling(click_prob):
        """
        click_prob: (n, n_class)
        """
        n_class = click_prob.shape[1]
        return np.int64([np.random.choice(n_class, 1, p=p) for p in click_prob]).reshape([-1])

    def infer_init(name, ct, batch_data):
        if name == 'sim':
            fetch_dict = ct.infer_init(SimFeedConvertor.infer_init(batch_data))
        elif name in ['env', 'credit', 'rl', 'cf', 'ddpg']:
            fetch_dict = ct.infer_init(GenFeedConvertor.infer_init(batch_data, ct.alg.model.conf))
        if name == 'cf':
            fetch_dict['prev_hidden'] = fetch_dict['user_feature']
        return np.array(fetch_dict['prev_hidden'])

    def gen_infer_onestep(name, ct, batch_data, prev_hidden, candidate_items, last_click_id):
        batch_size = batch_data.batch_size()
        cand_len = len(candidate_items[0])
        feed_dict = GenFeedConvertor.infer_onestep(batch_data, prev_hidden, candidate_items, last_click_id, conf=ct.alg.model.conf)
        if name == 'cf':
            feed_dict['user_feature'] = feed_dict['prev_hidden']
            del feed_dict['prev_hidden']
            fetch_dict = ct.infer_onestep(feed_dict)
            fetch_dict['prev_hidden'] = np.array(feed_dict['user_feature'])
        else:
            fetch_dict = ct.infer_onestep(feed_dict)
        next_hidden = np.array(fetch_dict['prev_hidden'])   # (b*cand_len, dim)
        if name == 'env':
            env_output_type = ct.alg.model._output_type
            if env_output_type == 'click':
                scores = click_prob_2_score(np.array(fetch_dict['click_prob'])) # (b*cand_len)
            elif env_output_type == 'rate':
                scores = np.array(fetch_dict['rate_pred']) # (b*cand_len, 1)
            else:
                raise NotImplementedError(env_output_type)
        elif name == 'credit':
            scores = np.array(fetch_dict['credit_pred'])    # (b*cand_len)
        elif name == 'rl':
            scores = np.array(fetch_dict['c_Q'])    # (b*cand_len)
        elif name == 'cf':
            cf_output_type = ct.alg.model._output_type
            if cf_output_type == 'click':
                scores = click_prob_2_score(np.array(fetch_dict['click_prob'])) # (b*cand_len)
            elif cf_output_type == 'rate':
                scores = np.array(fetch_dict['rate_pred']) # (b*cand_len, 1)
            else:
                raise NotImplementedError(cf_output_type)
        else:
            raise NotImplementedError(name)
        next_hidden = next_hidden.reshape([batch_size, cand_len, next_hidden.shape[-1]]) # (b, cand_len, dim)
        scores = scores.reshape([batch_size, cand_len])     # (b, cand_len)
        return next_hidden, scores

    def ddpg_infer_onestep(name, ct, batch_data, prev_hidden, candidate_items, last_click_id, last_item, step):
        batch_size = batch_data.batch_size()
        cand_len = len(candidate_items[0])
        feed_dict = DDPGFeedConvertor.infer_onestep(batch_data, prev_hidden, candidate_items, last_click_id, last_item, step, conf=ct.alg.model.conf)
        fetch_dict = ct.infer_onestep(feed_dict)
        next_hidden = np.array(fetch_dict['prev_hidden'])   # (b, dim)
        scores = np.array(fetch_dict['scores'])    # (b*cand_len)
        
        # next_hidden = next_hidden.reshape([batch_size, cand_len, next_hidden.shape[-1]]) # (b, cand_len, dim)
        scores = scores.reshape([batch_size, cand_len])     # (b, cand_len)
        return next_hidden, scores

    def sim_infer_onestep(ct, batch_data, prev_hidden, selected_item, last_click_id):
        feed_dict = SimFeedConvertor.infer_onestep(batch_data, prev_hidden, selected_item, last_click_id)
        fetch_dict = ct.infer_onestep(feed_dict)
        sim_response = sampling(np.array(fetch_dict['click_prob'])) # (b,)
        next_hidden = np.array(fetch_dict['prev_hidden'])   # (b, dim)
        return next_hidden, sim_response

    if gen_type == 'env':
        assert 'env' in dict_gen_ct, (dict_gen_ct.keys())
    elif gen_type == 'env_credit':
        assert 'env' in dict_gen_ct, (dict_gen_ct.keys())
        assert 'credit' in dict_gen_ct, (dict_gen_ct.keys())
    elif gen_type == 'rl':
        assert 'rl' in dict_gen_ct, (dict_gen_ct.keys())
    elif gen_type == 'ddpg':
        assert 'ddpg' in dict_gen_ct, (dict_gen_ct.keys())
    elif gen_type == 'cf':
        assert 'cf' in dict_gen_ct, (dict_gen_ct.keys())
    elif gen_type == 'env_rl':
        assert 'env' in dict_gen_ct, (dict_gen_ct.keys())
        assert 'rl' in dict_gen_ct, (dict_gen_ct.keys())
    elif gen_type == 'mc_credit':
        assert 'credit' in dict_gen_ct, (dict_gen_ct.keys())
    else:
        raise NotImplementedError(gen_type)

    batch_size = batch_data.batch_size()
    decode_len = batch_data.decode_len()[0]
    seq_len = batch_data.seq_lens()[0]

    sim_prev_hidden = infer_init('sim', ct_sim, batch_data)
    gen_names = dict_gen_ct.keys()
    dict_prev_hidden = {}
    for name in gen_names:
        dict_prev_hidden[name] = infer_init(name, dict_gen_ct[name], batch_data)
    last_click_id = np.zeros([batch_size]).astype('int64')  # (b,)
    selected_items = np.array([[] for _ in range(batch_size)]).astype('int64')
    
    sim_responses = []
    for step in range(decode_len):
        candidate_items = np.array([np.setdiff1d(np.arange(seq_len), x) for x in selected_items])   # (b, cand_len)
        cand_len = len(candidate_items[0])
        dict_next_hidden = {}
        dict_scores = {}

        # get gen scores
        for name in gen_names:
            if gen_type == 'ddpg':
                # last_item of the first step will be disabled by `first_step_mask`
                if len(selected_items[0]) == 0:
                    last_item = np.zeros([batch_size, 1]).astype('int64')
                else:
                    last_item = selected_items[:, -1]
                next_hidden, scores = ddpg_infer_onestep(name, dict_gen_ct[name], batch_data, dict_prev_hidden[name], candidate_items, last_click_id, last_item, step)
            else:
                next_hidden, scores = gen_infer_onestep(name, dict_gen_ct[name], batch_data, dict_prev_hidden[name], candidate_items, last_click_id)
            if gen_type == 'env_rl' and name == 'env':
                scores *= 0
            dict_next_hidden[name] = next_hidden    # (b, cand_len, dim)
            dict_scores[name] = scores              # (b, cand_len)

        # selection
        gen_scores = np.sum(dict_scores.values(), 0)    # (b, cand_len)
        selected_index = eps_greedy(gen_scores, eps)    # (b,)
        selected_item = candidate_items[np.arange(batch_size), selected_index]    # (b,)

        # feedback from simulator
        next_hidden, sim_response = sim_infer_onestep(ct_sim, batch_data, sim_prev_hidden, selected_item, last_click_id)
        sim_prev_hidden = next_hidden           # (b, dim)

        # update for next step
        last_click_id = sim_response.astype('int64')
        selected_items = np.concatenate([selected_items, selected_item.reshape([-1, 1])], 1)   # (b, cur_len+1)
        if gen_type == 'ddpg':      # ddpg doesn't need expanding
            pass
        else:
            for name in gen_names:
                dict_prev_hidden[name] = dict_next_hidden[name][np.arange(batch_size), selected_index]
        sim_responses.append(sim_response)

    sim_responses = np.array(sim_responses).T   # (b, decode_len)
    return selected_items, sim_responses


def get_globbase(replay_memory):
    list_click_id = []
    for batch_data in replay_memory:
        batch_size = batch_data.batch_size()
        seq_len = batch_data.seq_lens()[0]
        click_id = batch_data.get_values('click_id').reshape([batch_size, seq_len])
        list_click_id.append(click_id)
    globbase = np.mean(np.concatenate(list_click_id, 0), 0)     # (seq_len,)
    return globbase


def generate_credit_one_batch(ct_env, batch_data, credit_type, credit_gamma, globbase):
    """
    generate credit for one batch `batch_data`
    """
    def credit_gamma_mask(gamma, batch_size, seq_len):
        """
        [[1, gamma, gamma^2, gamma^3, ...],
         [0, 1, gamma^1, gamma^2, ...],
         ...
        ]
        shape: (batch_size, seq_len, seq_len)
        """
        ones = np.ones([batch_size, seq_len, seq_len])
        mask = np.tril(ones) + np.triu(ones, 1) * gamma
        mask = np.cumprod(mask, 2)
        return mask

    def get_masked_score(ct_env, batch_data, list_item_masks):
        feed_dict = CreditFeedConvertor.apply_masks(batch_data, list_item_masks, ct_env.alg.model.conf)
        fetch_dict = ct_env.inference(feed_dict)
        env_output_type = ct_env.alg.model._output_type
        if env_output_type == 'click':
            mask_click_prob = np.array(fetch_dict['click_prob'])    # (b * n_masks * seq_len, n_class)
            click_score = click_prob_2_score(mask_click_prob)       # (b * n_masks * seq_len,)
        elif env_output_type == 'rate':
            click_score = np.array(fetch_dict['rate_pred'])      # (b * n_masks * seq_len, 1)
        else:
            raise NotImplementedError(env_output_type)
        
        return click_score

    def mc_gt_base(ct_env, batch_data, gamma_mask):
        ones_diag = np.ones([seq_len, seq_len]).astype('int64')
        np.fill_diagonal(ones_diag, 0)
        ones = np.ones([seq_len, seq_len]).astype('int64')
        list_item_masks = np.concatenate([ones_diag, ones], 0)

        click_score = get_masked_score(ct_env, batch_data, list_item_masks)

        mask = np.triu(np.ones([batch_size, seq_len, seq_len]), 1)                  # (b, seq_len, seq_len)
        mask *= gamma_mask  # add gamma decay
        click_score = click_score.reshape([batch_size, seq_len * 2, seq_len])       # (b, seq_len * 2, seq_len)
        score_ones_diag, score_ones = np.split(click_score, 2, axis=1)              # (b, seq_len, seq_len), (b, seq_len, seq_len)
        credit = np.sum(score_ones * mask - score_ones_diag * mask, 2)              # (b, seq_len)
        return credit

    def mc_gt(ct_env, batch_data, gamma_mask):
        ones = np.ones([seq_len, seq_len]).astype('int64')
        list_item_masks = ones

        click_score = get_masked_score(ct_env, batch_data, list_item_masks)

        mask = np.triu(np.ones([batch_size, seq_len, seq_len]), 1)                  # (b, seq_len, seq_len)
        mask *= gamma_mask  # add gamma decay
        click_score = click_score.reshape([batch_size, seq_len, seq_len])           # (b, seq_len, seq_len)
        credit = np.sum(click_score * mask, 2)                                      # (b, seq_len)
        return credit

    def follow_click(batch_data, gamma_mask):
        click_id = np.array(batch_data.get_values('click_id')).reshape([batch_size, seq_len])   # (batch_size, seq_len)
        click_id = np.repeat(click_id, seq_len, axis=0).reshape([batch_size, seq_len, seq_len]) # (batch_size, seq_len, seq_len)

        click_score = click_id

        mask = np.triu(np.ones([batch_size, seq_len, seq_len]), 1)                  # (batch_size, seq_len, seq_len)
        mask *= gamma_mask  # add gamma decay
        credit = np.sum(click_score * mask, 2)                                      # (b, seq_len)
        return credit

    def mc(batch_data, gamma_mask):
        click_id = np.array(batch_data.get_values('click_id')).reshape([batch_size, seq_len])   # (batch_size, seq_len)
        click_id = np.repeat(click_id, seq_len, axis=0).reshape([batch_size, seq_len, seq_len]) # (batch_size, seq_len, seq_len)

        click_score = click_id

        mask = np.triu(np.ones([batch_size, seq_len, seq_len]), 0)                  # (batch_size, seq_len, seq_len)
        mask *= gamma_mask  # add gamma decay
        credit = np.sum(click_score * mask, 2)                                      # (b, seq_len)
        return credit

    def mc_gt_globbase(ct_env, batch_data, gamma_mask, globbase):
        ones = np.ones([seq_len, seq_len]).astype('int64')
        list_item_masks = ones

        click_score = get_masked_score(ct_env, batch_data, list_item_masks)

        mask = np.triu(np.ones([batch_size, seq_len, seq_len]), 1)                  # (batch_size, seq_len, seq_len)
        mask *= gamma_mask  # add gamma decay
        click_score = click_score.reshape([batch_size, seq_len, seq_len])           # (batch_size, seq_len, seq_len)
        globbase_expand = np.tile(globbase.reshape([1, 1, -1]), [batch_size, seq_len, 1])   # (batch_size, seq_len, seq_len)
        credit = np.sum((click_score - globbase_expand) * mask, 2)                  # (batch_size, seq_len)
        return credit

    batch_size = batch_data.batch_size()
    seq_len = batch_data.seq_lens()[0]

    # mask for gamma decay
    gamma_mask = credit_gamma_mask(credit_gamma, batch_size, seq_len)          # (b, seq_len, seq_len)

    if credit_type == 'gt_base':
        credit = mc_gt_base(ct_env, batch_data, gamma_mask)
    elif credit_type == 'gt':
        credit = mc_gt(ct_env, batch_data, gamma_mask)
    elif credit_type == 'follow_click':
        credit = follow_click(batch_data, gamma_mask)
    elif credit_type == 'mc':
        credit = mc(batch_data, gamma_mask)
    elif credit_type == 'gt_globbase':
        credit = mc_gt_globbase(ct_env, batch_data, gamma_mask, globbase)
    else:
        raise NotImplementedError(credit_type)

    return credit.flatten().astype('float32')     # (b*seq_len,)


################
# for one epoch
################

def online_inference(args, epoch_id, max_steps, data_gen, ct_sim, dict_gen_ct, summary_writer, if_print=True):
    """
    Do inference for `max_steps` batches.
    """
    sim_conf = ct_sim.alg.model.conf

    replay_memory = []
    list_sim_responses = []
    ### online inference
    last_batch_data = BatchData(sim_conf, data_gen.next())
    for batch_id in range(max_steps):
        np.random.seed(epoch_id * max_steps + batch_id)
        tensor_dict = data_gen.next()
        batch_data = BatchData(sim_conf, tensor_dict)
        batch_data.set_decode_len(batch_data.seq_lens())
        batch_data.expand_candidates(last_batch_data, batch_data.seq_lens())
        np.random.seed(None)
        del batch_data.tensor_dict['click_id']

        if batch_data.batch_size() == 1:    # otherwise, rl will crash
            continue

        orders, sim_responses = inference_one_batch(args.gen_type, ct_sim, dict_gen_ct, batch_data, eps=args.infer_eps) # , (b, decode_len)

        # save to replay memory
        sim_batch_data = batch_data.get_reordered(orders, sim_responses)
        replay_memory.append(sim_batch_data)
        list_sim_responses.append(sim_responses)
        last_batch_data = BatchData(sim_conf, tensor_dict)

        if batch_id % 100 == 0 and if_print:
            logging.info('inference epoch %d batch %d' % (epoch_id, batch_id))

    if if_print:
        list_sum_response = np.sum(np.concatenate(list_sim_responses, 0), 1)    # (b,)
        add_scalar_summary(summary_writer, epoch_id, 'inference/sim_responses', np.mean(list_sum_response))
    return replay_memory


def online_inference_for_test(args, epoch_id, max_steps, ct_sim, dict_gen_ct, summary_writer):
    """
    Do inference on the test test.
    """
    sim_conf = ct_sim.alg.model.conf
    dataset = NpzDataset(args.test_npz_list, 
                        sim_conf.npz_config_path, 
                        sim_conf.requested_names,
                        if_random_shuffle=False,
                        one_pass=True)
    data_gen = dataset.get_data_generator(sim_conf.batch_size)
    thread_data_gen = threaded_generator(data_gen, capacity=100)

    list_sim_responses = []
    ### online inference
    last_batch_data = BatchData(sim_conf, thread_data_gen.next())
    for batch_id, tensor_dict in enumerate(thread_data_gen):
        if batch_id > max_steps:
            break
        np.random.seed(batch_id)
        batch_data = BatchData(sim_conf, tensor_dict)
        batch_data.set_decode_len(batch_data.seq_lens())
        batch_data.expand_candidates(last_batch_data, batch_data.seq_lens())
        np.random.seed(None)
        del batch_data.tensor_dict['click_id']

        orders, sim_responses = inference_one_batch(args.gen_type, ct_sim, dict_gen_ct, batch_data, eps=0) # , (b, decode_len)

        # save to replay memory
        sim_batch_data = batch_data.get_reordered(orders, sim_responses)
        list_sim_responses.append(sim_responses)
        last_batch_data = BatchData(sim_conf, tensor_dict)

        if batch_id % 100 == 0:
            logging.info('inference test batch %d' % batch_id)

    list_sum_response = np.sum(np.concatenate(list_sim_responses, 0), 1)    # (b,)
    add_scalar_summary(summary_writer, epoch_id, 'inference/test_sim_responses', np.mean(list_sum_response))
    

def offline_training(args, epoch_id, replay_memory, dict_gen_ct, summary_writer, if_save=True, env_rl_data_gen=None):
    """
    Do offline train on the replay_memory.
    """
    ### offline train env model
    if args.gen_type in ['env', 'env_credit', 'env_rl']:
        list_loss = []
        for sim_batch_data in replay_memory:
            ct_env = dict_gen_ct['env']
            fetch_dict = ct_env.train(EnvFeedConvertor.train_test(sim_batch_data, args.env_item_dropout_rate, ct_env.alg.model.conf))
            list_loss.append(np.array(fetch_dict['loss']))
        if if_save:
            add_scalar_summary(summary_writer, epoch_id, 'train/env_loss', np.mean(list_loss))
            ct_env.save_model(epoch_id)

    ### offline train credit
    if args.gen_type == 'env_credit':
        if args.credit_type == 'gt_globbase':
            globbase = get_globbase(replay_memory)  # (seq_len,)
            print('globbase', globbase.tolist())
        else:
            globbase = None
        list_loss = []
        for sim_batch_data in replay_memory:
            ct_env = dict_gen_ct['env']
            ct_credit = dict_gen_ct['credit']
            credit = generate_credit_one_batch(ct_env, 
                                            sim_batch_data, 
                                            credit_type=args.credit_type, 
                                            credit_gamma=args.credit_gamma,
                                            globbase=globbase)
            fetch_dict = ct_credit.train(CreditFeedConvertor.train_test(sim_batch_data, credit, ct_credit.alg.model.conf))
            list_loss.append(np.array(fetch_dict['loss']))
        if if_save:
            add_scalar_summary(summary_writer, epoch_id, 'train/credit_loss', np.mean(list_loss))
            ct_credit.save_model(epoch_id)

    ### offline train mc_credit
    if args.gen_type == 'mc_credit':
        list_loss = []
        for sim_batch_data in replay_memory:
            ct_credit = dict_gen_ct['credit']
            credit = generate_credit_one_batch(None, 
                                            sim_batch_data, 
                                            credit_type='mc', 
                                            credit_gamma=args.credit_gamma,
                                            globbase=None)
            fetch_dict = ct_credit.train(CreditFeedConvertor.train_test(sim_batch_data, credit, ct_credit.alg.model.conf))
            list_loss.append(np.array(fetch_dict['loss']))
        if if_save:
            add_scalar_summary(summary_writer, epoch_id, 'train/credit_loss', np.mean(list_loss))
            ct_credit.save_model(epoch_id)

    ### offline train rl
    if args.gen_type == 'rl':
        list_loss = []
        for sim_batch_data in replay_memory:
            ct_rl = dict_gen_ct['rl']
            fetch_dict = ct_rl.train(RLFeedConvertor.train_test(sim_batch_data, ct_rl.alg.model.conf))
            list_loss.append(np.array(fetch_dict['loss']))
        if if_save:
            add_scalar_summary(summary_writer, epoch_id, 'train/rl_loss', np.mean(list_loss))  
            ct_rl.save_model(epoch_id)  

    ### offline train ddpg
    if args.gen_type == 'ddpg':
        list_actor_loss = []
        list_critic_loss = []
        for batch_id, sim_batch_data in enumerate(replay_memory):
            ct_ddpg = dict_gen_ct['ddpg']
            fetch_dict = ct_ddpg.train(DDPGFeedConvertor.train_test(sim_batch_data, ct_ddpg.alg.model.conf))
            list_actor_loss.append(np.array(fetch_dict['actor_loss']))
            list_critic_loss.append(np.array(fetch_dict['critic_loss']))
            if batch_id % 100 == 0:
                print(epoch_id, batch_id, 'train/ddpg_actor_loss', np.mean(np.array(fetch_dict['actor_loss'])))
                print(epoch_id, batch_id, 'train/ddpg_critic_loss', np.mean(np.array(fetch_dict['critic_loss'])))
        if if_save:
            add_scalar_summary(summary_writer, epoch_id, 'train/ddpg_total_loss', np.mean(list_actor_loss) + np.mean(list_critic_loss))  
            add_scalar_summary(summary_writer, epoch_id, 'train/ddpg_actor_loss', np.mean(list_actor_loss))  
            add_scalar_summary(summary_writer, epoch_id, 'train/ddpg_critic_loss', np.mean(list_critic_loss))  
            ct_ddpg.save_model(epoch_id)  

    ### offline train cf model
    if args.gen_type == 'cf':
        list_loss = []
        for sim_batch_data in replay_memory:
            ct_cf = dict_gen_ct['cf']
            fetch_dict = ct_cf.train(CFFeedConvertor.train_test(sim_batch_data, ct_cf.alg.model.conf))
            list_loss.append(np.array(fetch_dict['loss']))
        if if_save:
            add_scalar_summary(summary_writer, epoch_id, 'train/cf_loss', np.mean(list_loss))
            ct_cf.save_model(epoch_id)

    ### offline train rl with additional data from env
    if args.gen_type == 'env_rl':
        ct_env = dict_gen_ct['env']
        max_env_train_steps = len(replay_memory)
        env_replay_memory = online_inference(args, 0, max_env_train_steps, env_rl_data_gen, ct_env, dict_gen_ct, None, if_print=False)
        list_loss = []
        for sim_batch_data in replay_memory + env_replay_memory:
            ct_rl = dict_gen_ct['rl']
            fetch_dict = ct_rl.train(RLFeedConvertor.train_test(sim_batch_data, ct_rl.alg.model.conf))
            list_loss.append(np.array(fetch_dict['loss']))
        if if_save:
            add_scalar_summary(summary_writer, epoch_id, 'train/rl_loss', np.mean(list_loss))  
            ct_rl.save_model(epoch_id)  


#######
# main
#######

def main_online_iterate(args):
    """
    Include online inference and offline training.
    """
    ct_sim = get_ct_sim(args.sim_exp, args.use_cuda, args.train_mode, args.sim_cell_type, args.output_dim)
    assert ct_sim.ckp_step > 0, (ct_sim.ckp_step)
    dict_gen_ct = {}
    if args.gen_type in ['env', 'env_credit', 'env_rl']:
        if args.gen_type == 'env_rl':
            assert args.env_output_type == 'click', \
                ('env_rl only support click env, which will be used as a simulator', args.env_output_type)
        ct_env = get_ct_env(args.env_exp, args.use_cuda, args.train_mode, args.env_output_type, args.output_dim)
        dict_gen_ct['env'] = ct_env
    if args.gen_type in ['env_credit', 'mc_credit']:
        ct_credit = get_ct_credit(args.credit_exp, args.use_cuda, args.train_mode, args.credit_scale)
        dict_gen_ct['credit'] = ct_credit
    if args.gen_type in ['rl', 'env_rl']:
        ct_rl = get_ct_rl(args.rl_exp, args.use_cuda, args.train_mode, args.rl_gamma, args.rl_Q_type)
        dict_gen_ct['rl'] = ct_rl
    if args.gen_type == 'ddpg':
        ct_ddpg = get_ct_ddpg(args.ddpg_exp, args.use_cuda, args.train_mode, args.ddpg_gamma)
        dict_gen_ct['ddpg'] = ct_ddpg
    if args.gen_type == 'cf':
        ct_cf = get_ct_cf(args.cf_exp, args.use_cuda, args.train_mode, args.cf_output_type, args.output_dim)
        dict_gen_ct['cf'] = ct_cf

    ### dataset
    np.random.seed(21)  # IMPORTANT! so different exp can be comparable
    sim_conf = ct_sim.alg.model.conf
    dataset = NpzDataset(args.train_npz_list, 
                        sim_conf.npz_config_path, 
                        sim_conf.requested_names,
                        if_random_shuffle=True,
                        one_pass=False)
    data_gen = dataset.get_data_generator(sim_conf.batch_size)
    thread_data_gen = threaded_generator(data_gen, capacity=50)

    if args.gen_type == 'env_rl':       # env_rl will need data from env_conf
        env_conf = ct_env.alg.model.conf
        env_dataset = NpzDataset(args.train_npz_list, 
                                env_conf.npz_config_path, 
                                env_conf.requested_names,
                                if_random_shuffle=True,
                                one_pass=False)
        env_data_gen = env_dataset.get_data_generator(env_conf.batch_size)
        thread_env_data_gen = threaded_generator(env_data_gen, capacity=50)
    else:
        thread_env_data_gen = None

    summary_writer = tf.summary.FileWriter(args.summary_dir)
    max_steps = 1000
    max_test_steps = 1000
    for epoch_id in range(50):
        replay_memory = online_inference(args, epoch_id, max_steps, thread_data_gen, ct_sim, dict_gen_ct, summary_writer)
        if epoch_id % 1 == 0:
            online_inference_for_test(args, epoch_id, max_test_steps, ct_sim, dict_gen_ct, summary_writer)
        offline_training(args, epoch_id, replay_memory, dict_gen_ct, summary_writer, env_rl_data_gen=thread_env_data_gen)


def main_batch_rl(args):
    """
    Include online inference and offline training.
    """
    ct_sim = get_ct_sim(args.sim_exp, args.use_cuda, args.train_mode, args.sim_cell_type, args.output_dim)
    assert ct_sim.ckp_step > 0, (ct_sim.ckp_step)
    dict_gen_ct = {}
    if args.gen_type in ['env', 'env_credit', 'env_rl']:
        if args.gen_type == 'env_rl':
            assert args.env_output_type == 'click', \
                ('env_rl only support click env, which will be used as a simulator', args.env_output_type)
        ct_env = get_ct_env(args.env_exp, args.use_cuda, args.train_mode, args.env_output_type, args.output_dim)
        dict_gen_ct['env'] = ct_env
    if args.gen_type in ['env_credit', 'mc_credit']:
        ct_credit = get_ct_credit(args.credit_exp, args.use_cuda, args.train_mode, args.credit_scale)
        dict_gen_ct['credit'] = ct_credit
    if args.gen_type in ['rl', 'env_rl']:
        ct_rl = get_ct_rl(args.rl_exp, args.use_cuda, args.train_mode, args.rl_gamma, args.rl_Q_type)
        dict_gen_ct['rl'] = ct_rl
    if args.gen_type == 'ddpg':
        ct_ddpg = get_ct_ddpg(args.ddpg_exp, args.use_cuda, args.train_mode, args.ddpg_gamma)
        dict_gen_ct['ddpg'] = ct_ddpg

    ### dataset
    sim_conf = ct_sim.alg.model.conf
    dataset = NpzDataset(args.train_npz_list, 
                        sim_conf.npz_config_path, 
                        sim_conf.requested_names,
                        if_random_shuffle=True,
                        one_pass=True)

    if args.gen_type == 'env_rl':       # env_rl will need data from env_conf
        env_conf = ct_env.alg.model.conf
        env_dataset = NpzDataset(args.train_npz_list, 
                                env_conf.npz_config_path, 
                                env_conf.requested_names,
                                if_random_shuffle=True,
                                one_pass=False)

    summary_writer = tf.summary.FileWriter(args.summary_dir)
    max_test_steps = 1000
    for epoch_id in range(50):
        if args.gen_type == 'env_rl':
            env_data_gen = env_dataset.get_data_generator(env_conf.batch_size)
            thread_env_data_gen = threaded_generator(env_data_gen, capacity=10)
        else:
            thread_env_data_gen = None

        data_gen = dataset.get_data_generator(sim_conf.batch_size)
        thread_data_gen = threaded_generator(data_gen, capacity=100)
        for batch_id, tensor_dict in enumerate(thread_data_gen):
            if_save = True if batch_id == 0 else False
            batch_data = BatchData(sim_conf, tensor_dict)
            if batch_data.batch_size() == 1:    # otherwise, rl will crash
                continue
            offline_training(args, epoch_id, [batch_data], dict_gen_ct, summary_writer, if_save=if_save, env_rl_data_gen=thread_env_data_gen)
        if epoch_id % 1 == 0:
            online_inference_for_test(args, epoch_id, max_test_steps, ct_sim, dict_gen_ct, summary_writer)


def main_calculate_credit_variance(args, credit_type):
    """
    Calculate variance of credit by varing following items.
    """
    def sampling(click_prob):
        """
        click_prob: (n, n_class)
        """
        n_class = click_prob.shape[1]
        return np.int64([np.random.choice(n_class, 1, p=p) for p in click_prob]).reshape([-1])

    assert args.gen_type == 'env'
    ct_sim = get_ct_sim(args.sim_exp, args.use_cuda, args.train_mode, args.sim_cell_type, args.output_dim)
    assert ct_sim.ckp_step > 0, (ct_sim.ckp_step)
    ct_env = get_ct_env(args.env_exp, args.use_cuda, args.train_mode, args.env_output_type, args.output_dim)
    assert ct_env.ckp_step > 0, (ct_env.ckp_step)

    ### dataset
    sim_conf = ct_sim.alg.model.conf
    dataset = NpzDataset(args.train_npz_list, 
                        sim_conf.npz_config_path, 
                        sim_conf.requested_names,
                        if_random_shuffle=True,
                        one_pass=True)
    data_gen = dataset.get_data_generator(sim_conf.batch_size)
    thread_data_gen = threaded_generator(data_gen, capacity=100)

    n_vary = 64
    base_batch_data = BatchData(sim_conf, thread_data_gen.next())
    batch_size = base_batch_data.batch_size()
    batch_credits = []
    for pos in range(base_batch_data.seq_lens()[0]):
        list_credits = []
        for batch_id, tensor_dict in enumerate(thread_data_gen):
            if len(list_credits) == n_vary:
                break
            ref_batch_data = BatchData(sim_conf, tensor_dict)
            if ref_batch_data.batch_size() != batch_size:
                continue
            mix_batch_data = base_batch_data.replace_following_items(pos + 1, ref_batch_data)
            sim_fetch_dict = ct_sim.inference(SimFeedConvertor.inference(mix_batch_data))
            sim_response = sampling(np.array(sim_fetch_dict['click_prob'])).reshape([-1, 1]).astype('int64')
            mix_batch_data.tensor_dict['click_id'] = FakeTensor(sim_response, mix_batch_data.seq_lens())
            credit = generate_credit_one_batch(ct_env, 
                                            mix_batch_data, 
                                            credit_type=credit_type,
                                            credit_gamma=args.credit_gamma,
                                            globbase=None)
            credit = credit.reshape([batch_size, -1])
            list_credits.append(credit[:, pos].reshape(-1, 1))
        list_credits = np.concatenate(list_credits, 1)  # (b, n_vary)
        batch_credits.append(list_credits)
    batch_credits = np.concatenate(batch_credits, 0)    # (seq_len*b, n_vary)
    print(credit_type)
    print(batch_credits.shape)
    print('(s,a)-wise credit variance', np.mean(np.std(batch_credits, 1)))


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    if args.task == 'online':
        main_online_iterate(args)
    elif args.task == 'batchrl':
        main_batch_rl(args)
    elif args.task == 'credit_variance':
        list_credit_types = ['gt_base', 'follow_click', 'gt']
        for credit_type in list_credit_types:
            main_calculate_credit_variance(args, credit_type)
    else:
        raise NotImplementedError(args.task)



