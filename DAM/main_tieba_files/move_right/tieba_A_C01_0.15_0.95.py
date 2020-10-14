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
    "data_path": "./data/tieba/data.pkl",
    "word_emb_init": './data/tieba/word_embedding.pkl',
    "word_to_id": "./data/tieba/word2id",
    "save_path": "./output/tieba_A_C01_015_095/",
    "init_model": None, #should be set for test

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

    "calibration_max_step": 10000000000,
    "matching_max_step": 10000000000,
    "validation_step": 100, #correspond to n
    "validation_update_batch_percentage": 0.5, #correspond to m

    "decay_steps": 1000,
    "decay_rate": 0.9,
    #cross_entropy / hinge
    "matching_loss_type": 'cross_entropy',
    #"calibration_loss_type": 'hinge',
    #"matching_loss_type": 'hinge',
    "calibration_loss_type": 'cross_entropy',

    "positive_sample_threshold": 0.95,
    "negative_sample_threshold": 0.15,

    "update_cmodel_epoch_end": True
}

joint_model = joint_net.Net(conf)
final_model_save_name = train.train(conf, joint_model)
conf['init_model'] = final_model_save_name
print('=' * 60 + '\n' + 'Start Testing' + '\n' + '=' * 60)
print('Init model ckpt: %s'%final_model_save_name)
joint_test_model = joint_net.Net(conf)
test.test(conf, joint_test_model, type='test')
