"""
configuration file
"""

import os
from os.path import join, dirname, basename
from collections import OrderedDict

DATA_DIR = os.environ.get('DATA_DIR', '')
OUTPUT_DIR = os.environ.get('OUTPUT_DIR', '')        

class Config(object):
    """
    config file
    """
    def __init__(self, exp):
        self.name = basename(__file__)
        self.exp = exp

        ##########
        # logging
        ##########
        self.summary_dir = join(OUTPUT_DIR, 'logs', exp)
        self.model_dir = join(OUTPUT_DIR, 'params', exp)
        self.inference_model_dir = join(OUTPUT_DIR, 'inference_models', exp)
        self.test_results_dir = join(OUTPUT_DIR, 'test_results', exp)
        self.credit_dir = join(OUTPUT_DIR, 'credits', exp)

        ####################
        # train settings
        ####################
        self.max_train_steps = 20
        self.prt_interval = 1000
        self.optimizer = 'Adam'
        self.lr = 1e-3

        ##########
        # dataset
        ##########
        self.batch_size = 256
        # self.train_npz_list = join(DATA_DIR, 'train_npz_list.txt')
        # self.test_npz_list = join(DATA_DIR, 'test_npz_list.txt')
        self.npz_config_path = join(DATA_DIR, 'conf/npz_config.json')

        ##########
        # fluid model
        ##########
        ### definitions
        item_attributes = self._get_item_attributes()
        recent_attributes = self._get_recent_attributes()
        label_attributes = self._get_label_attributes()

        ### used to build embeddings
        self.item_slot_names = item_attributes.keys()
        self.recent_slot_names = recent_attributes.keys()

        #
        self.label_slot_names = label_attributes.keys()

        ### used as reference to build fluid.layers.data
        self.data_attributes = OrderedDict()
        self.data_attributes.update(item_attributes)
        self.data_attributes.update(recent_attributes)
        self.data_attributes.update(label_attributes)

        ### passed to NpzDataset
        self.requested_names = self.data_attributes.keys()
        self.requested_names.remove('last_click_id')    # will be made up in train scripts
        if 'credit' in self.requested_names:
            self.requested_names.remove('credit')       # will be made up in train scripts

        self.shared_embedding_names = self._get_shared_embedding_names()

    def _get_item_attributes(self):
        raise NotImplementedError()

    def _get_recent_attributes(self):
        raise NotImplementedError()

    def _get_label_attributes(self):
        raise NotImplementedError()

    def _get_shared_embedding_names(self):
        raise NotImplementedError()


