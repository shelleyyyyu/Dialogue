import sys
import os
import time

import cPickle as pickle
import tensorflow as tf
import numpy as np

import utils.reader as reader
import models.net as net
import utils.evaluation as eva
#for douban
#import utils.douban_evaluation as eva

import bin.train_and_evaluate as train
import bin.test_and_evaluate as test

# configure

conf = {
    "data_path": "./data/jdqa/data.pkl",
    "save_path": "./output/jdqa/",
    "word_emb_init": "./data/jdqa/word_embedding.pkl",
    "init_model": './output/jdqa/model.ckpt.10', #should be set for test

    "rand_seed": None, 

    "drop_dense": None,
    "drop_attention": None,

    "is_mask": True,
    "is_layer_norm": True,
    "is_positional": False,  

    "stack_num": 5,  
    "attention_type": "dot",

    "learning_rate": 1e-3,
    "vocab_size": 256358, #ubuntu: 434512; jdqa: 256358; douban: 172130
    "batch_size": 32, #200 for test
    "emb_size": 80,

    "max_turn_num": 1,
    "max_turn_len": 30,

    "max_to_keep": 1,
    "num_scan_data": 5,
    "_EOS_": 1, #1 for douban data 28270 for ubuntu
    "final_n_class": 1,
}


model = net.Net(conf)
#train.train(conf, model)

#test and evaluation, init_model in conf should be set
test.test(conf, model)

