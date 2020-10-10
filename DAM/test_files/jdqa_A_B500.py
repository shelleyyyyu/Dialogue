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


conf = {
    "data_path": "./data/jdqa/data.pkl",
    "word_emb_init": "./data/jdqa/word_embedding.pkl",
    "save_path": "./output/jdqa_A_B500/",
    "init_model": "./output/jdqa_A_B500/joint_learning_model.ckpt.1", #should be set for test

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
    "batch_size": 256, #256 fot train 200 for test
    "vocab_size": 256357,
    "emb_size": 80,

    "max_turn_num": 1,
    "max_turn_len": 30,

    "max_to_keep": 1,
    "num_scan_data": 5,
    "_EOS_": 1, #1 for douban data; 28270 for ubuntu
    "final_n_class": 1,

    "matching_pretrain_epoch": 1,
    "calibration_pretrain_epoch": 1,

    "calibration_type": 0, #0: labels 1: logits

    "calibration_max_step": 1000000000000,
    "matching_max_step": 1000000000000,
    "validation_step": 500, #correspond to n
    "validation_update_batch_percentage": 1.0, #correspond to m

    "decay_steps": 1000,
    "decay_rate": 1.0,
    #cross_entropy / hinge
    "matching_loss_type": 'cross_entropy',
    #"calibration_loss_type": 'hinge',
    #"matching_loss_type": 'hinge',
    "calibration_loss_type": 'cross_entropy'
}

joint_model = joint_net.Net(conf)
test.test(conf, joint_model, type='test')
joint_model = joint_net.Net(conf)
test.test(conf, joint_model, type='valid')
joint_model = joint_net.Net(conf)
test.test(conf, joint_model, type='train')

