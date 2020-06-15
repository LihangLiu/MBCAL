import numpy as np
from collections import OrderedDict
import os
from os.path import exists, basename, dirname, join
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

import _init_paths
from src.utils import save_lines_2_txt

class CreditSaver(object):
    """docstring for CreditSaver"""
    def __init__(self, credits_list_txt, relative_file_dir):
        super(CreditSaver, self).__init__()
        self._credits_list_txt = credits_list_txt
        self._relative_file_dir = relative_file_dir
        self._CREDIT_FILE_SUFFIX = ".credit"

        if not exists(dirname(credits_list_txt)):
            os.makedirs(dirname(credits_list_txt))
        assert not exists(credits_list_txt), ('please remove', credits_list_txt)
        self._credits_list_txt_f = open(credits_list_txt, 'w')

        self._last_npz_file = None
        self.credits = []

    def add(self, npz_file, credit):
        """
        credit: 1d list
        """
        if npz_file != self._last_npz_file:
            self._save_last_credits()
            self.credits = []
        self.credits.append(credit)
        self._last_npz_file = npz_file

    def batch_add(self, npz_file, list_credit):
        """
        list_credit: a list of 1d list
        """
        if npz_file != self._last_npz_file:
            self._save_last_credits()
            self.credits = []
        self.credits += list_credit
        self._last_npz_file = npz_file

    def close(self):
        self._save_last_credits()
        self._credits_list_txt_f.close()

    def _save_last_credits(self):
        if self._last_npz_file is None:
            return
        relative_credit_file = join(self._relative_file_dir, basename(self._last_npz_file) + self._CREDIT_FILE_SUFFIX)
        credit_file = join(dirname(self._credits_list_txt), relative_credit_file)
        self._save_to_credit_file(credit_file, self.credits)
        self._credits_list_txt_f.write('%s %s\n' % (relative_credit_file, self._last_npz_file))
        self._credits_list_txt_f.flush()
        logging.info('saved to %s' % self._credits_list_txt)

    def _save_to_credit_file(self, file, credits):
        credits = [' '.join([str(x) for x in credit]) for credit in credits]
        save_lines_2_txt(file, credits)

class CreditDataset(object):
    """docstring for CreditDataset"""
    def __init__(self, credits_list_txt):
        super(CreditDataset, self).__init__()
        self.map_npz_credit = self._read_list_txt(credits_list_txt)

    def _read_list_txt(self, list_txt_path):
        """
        txt_format:
            credit_file npz_file
        TODO: only the basename of npz_file will be used as the identifier, 
                be careful if the basenames conflict
        """
        map_npz_credit = {}
        base_dir = dirname(list_txt_path)
        with open(list_txt_path, 'r') as f:
            for line in f:
                credit_file, npz_file = line.strip().split()
                assert basename(npz_file) not in map_npz_credit, \
                        ('please check if basenames conflicts', npz_file)
                map_npz_credit[basename(npz_file)] = join(base_dir, credit_file)
        return map_npz_credit
        
    def _read_credit_file(self, filename):
        """
        txt_format: 
            x1 x2 x3
            x4 x5 x6 x7
            x8
        return:
            [[x1, x2, x3], [x4, x5, x6, x7], [x8]]
        """
        credits = []
        with open(filename, 'r') as f:
            for line in f.readlines():
                credit = [float(x) for x in line.strip().split()]
                credits.append(credit)
        return credits

    def get_credit(self, npz_file):
        npz_file = basename(npz_file)
        assert npz_file in self.map_npz_credit, (npz_file)
        return self._read_credit_file(self.map_npz_credit[npz_file])

    def has_credit(self, npz_file):
        return basename(npz_file) in self.map_npz_credit
        

