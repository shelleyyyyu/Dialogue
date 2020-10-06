import sys
import os
import time

import cPickle as pickle
import tensorflow as tf
import numpy as np

import utils.reader as reader
import models.joint_net as joint_net
import utils.evaluation as eva
#for douban
#import utils.douban_evaluation as eva

import bin.train_and_evaluate as train
import bin.test_and_evaluate as test

# configure

conf = {
    "data_path": "./data/douban/data.pkl",
    "word_emb_init": './data/douban/word_embedding.pkl',
    "save_path": "./output/douban/",
    "init_model": None,  # should be set for test

    "rand_seed": None,

    "drop_dense": None,
    "drop_attention": None,

    "is_mask": True,
    "is_layer_norm": True,
    "is_positional": False,

    "stack_num": 5,
    "c_stack_num": 5,
    "attention_type": "dot",

    "learning_rate": 1e-3,
    "vocab_size": 172130,
    "batch_size": 100, #256 fot train 200 for test
    "emb_size": 80,

    "max_turn_num": 1,
    "max_turn_len": 30,

    "max_to_keep": 1,
    "num_scan_data": 1,
    "_EOS_": 1, #1 for douban data 28270 for ubuntu
    "final_n_class": 1,

    "calibration_type": 0  # 0: labels 1: logits
}

joint_model = joint_net.Net(conf)
test.test(conf, joint_model)

