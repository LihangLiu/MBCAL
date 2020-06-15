import numpy as np
from cPickle import dumps
import time
import os
import copy
import logging
import thread
from Queue import Queue
from os.path import join, dirname, basename, isfile

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

import _init_paths
from src.utils import read_json, save_json, seq_len_2_lod, tik, tok, AssertEqual


class FakeTensor(object):
    """
    values, lod, seq_lens
    """
    def __init__(self, values, seq_lens=None):
        self.values = values
        self.seq_lens = seq_lens
        if seq_lens is None:
            self.lod = []
        else:
            AssertEqual(len(values), np.sum(seq_lens))
            self.lod = [seq_len_2_lod(seq_lens)]


class NpzDataset:
    def __init__(self, 
                list_txt_path, 
                npz_config_path, 
                fetched_slot_names, 
                credit_dataset=None,
                if_random_shuffle=True,
                one_pass=True):
        self.list_npz_file = self.read_list_txt(list_txt_path)
        self.npz_config = read_json(npz_config_path)
        self.fetched_slot_names = set(fetched_slot_names)
        self.credit_dataset = credit_dataset
        self.if_random_shuffle = if_random_shuffle
        self.one_pass = one_pass

        self._check_credits_exists()
    
    def read_list_txt(self, list_txt_path):
        list_npz = []
        base_dir = dirname(list_txt_path)
        with open(list_txt_path, 'r') as f:
            for line in f:
                list_npz.append(join(base_dir, line.strip()))
        return list_npz

    def _check_credits_exists(self):
        # check credit files
        if self.credit_dataset:
            for npz_file in self.list_npz_file:
                assert self.credit_dataset.has_credit(npz_file), (npz_file)

    @property
    def size(self):
        return len(self.list_npz_file)

    def fetch_numpy_data(self, data_dict, npz_config, credits=None, start=None, end=None):
        """
        args:
            list_values: from .npz file. a list of 1d array
            list_seq_lens: from .npz file. a list of 1d array
            credits: 2d list
            start, end: start list_id and end list_id
        """
        var_feature_size = npz_config['VarFeatureSize']

        tensor_dict = {}
        for name in self.fetched_slot_names:
            values = data_dict['values'][name]
            seq_lens = data_dict['seq_lens'][name]
            lod = data_dict['lod'][name]
            sub_value = np.array(values[lod[start]:lod[end]])
            sub_seq_len = np.array(seq_lens[start:end])
            if name in var_feature_size:
                sub_value = np.reshape(sub_value, [-1, var_feature_size[name]])
                sub_seq_len = [x/var_feature_size[name] for x in sub_seq_len]

            tensor_dict[name] = FakeTensor(sub_value, sub_seq_len)

        # get credit
        if credits:
            sub_credits = credits[start: end]
            sub_value = np.concatenate(sub_credits, 0)
            sub_seq_len = np.array([len(c) for c in sub_credits])
            tensor_dict['credit'] = FakeTensor(sub_value, sub_seq_len)
        return tensor_dict

    def get_data_generator(self, batch_size):
        """
        return:
            batch_data
        """
        while True:
            if self.if_random_shuffle:
                np.random.shuffle(self.list_npz_file)
            ### go over all files
            for npz_file in self.list_npz_file:
                def _npz_list_2_dict(npz_file):
                    """
                    list_values: from .npz file. a list of 1d array
                    list_seq_lens: from .npz file. a list of 1d array
                    """
                    data = np.load(npz_file)    # {'values':[], 'seq_len':[]}
                    list_values, list_seq_lens = data['values'], data['seq_len']

                    npz_names = self.npz_config['names']
                    data_dict = {'values':{}, 'seq_lens':{}, 'lod':{}}
                    for name in self.fetched_slot_names:
                        index = npz_names.index(name)
                        seq_lens = list_seq_lens[index]
                        data_dict['values'][name] = list_values[index]
                        data_dict['seq_lens'][name] = seq_lens
                        data_dict['lod'][name] = seq_len_2_lod(seq_lens)
                    return data_dict
                
                data_dict = _npz_list_2_dict(npz_file)
                credits = self.credit_dataset.get_credit(npz_file) if self.credit_dataset else None
                num_list = len(data_dict['seq_lens'].values()[0])
                for start in range(0, num_list, batch_size):
                    end = min(num_list, start + batch_size)
                    tensor_dict = self.fetch_numpy_data(data_dict,
                                                        self.npz_config, 
                                                        credits=credits,
                                                        start=start, 
                                                        end=end)
                    yield tensor_dict

            if self.one_pass:
                break



        