import sys
import os
import time

import cPickle as pickle
import tensorflow as tf
import numpy as np

import utils.reader as reader
import utils.douban_evaluation as eva
from sklearn.metrics import accuracy_score

def test(conf, _model):

    # load data
    print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())) + ' - starting loading data')
    train_data, val_data, test_data, validation_data = pickle.load(open(conf["data_path"], 'rb'))
    print(str(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))) + ' - finish loading data')
    test_batches = reader.build_batches(validation_data, conf)
    print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())) + " - finish building test batches")

    # refine conf
    test_batch_num = len(test_batches["response"])
    print('configurations: %s' %conf)
    _graph = _model.build_graph()
    print(str(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))) + ' - build model graph success')

    with tf.Session(graph=_graph) as _sess:
        # Init calibration model sess
        tf.global_variables_initializer().run()

        if conf["init_model"]:
            _model.saver.restore(_sess, conf["init_model"])
            print("success init calibration model %s" %conf["init_model"])

        average_correction_rate = 0.0

        score_file_path = conf['save_path'] + 'score.valid'
        score_file = open(score_file_path, 'w')

        print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())) + ' - starting test')

        for batch_index in xrange(test_batch_num):

            _feed = {
                _model.is_pretrain_calibration: False,
                _model.is_pretrain_matching: False,
                _model.is_backprop_calibration: False,
                _model.is_backprop_matching: False,
                _model.calibration_type: conf['calibration_type'],
                _model._turns: test_batches["turns"][batch_index],
                _model._tt_turns_len: test_batches["tt_turns_len"][batch_index],
                _model._every_turn_len: test_batches["every_turn_len"][batch_index],
                _model._response: test_batches["response"][batch_index],
                _model._response_len: test_batches["response_len"][batch_index],
                _model._label: test_batches["label"][batch_index],
                }

            _, c_y_pred, _, m_y_pred = _sess.run([_model.c_logits, _model.c_y_pred, _model.m_logits, _model.m_y_pred], feed_dict=_feed)

            calibrated_label = ['1' if scores[1] > scores[0] else '0' for scores in c_y_pred]
            calibrated_rate = 1 - accuracy_score(calibrated_label, test_batches["label"][batch_index])
            average_correction_rate += calibrated_rate

            for i in xrange(conf["batch_size"]):
                score_file.write(
                    str(c_y_pred[i][-1]) + '\t' +
                    str(m_y_pred[i][-1]) + '\t' +
                    str(test_batches["label"][batch_index][i]) + '\n')
        
        print('Data Calibration Rate: %.4f' % (average_correction_rate/test_batch_num))
        score_file.close()
        print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())) + ' - finish test')
        
        #write evaluation result
        result = eva.evaluate(score_file_path)
        result.update(eva.evaluate_auc_from_file(score_file_path))
        result_file_path = conf["save_path"] + "result.valid"
        with open(result_file_path, 'w') as out_file:
            for key in result.keys():
                r = '%.4f'%result[key]
                out_file.write(str(key) + '\t' + r + '\n')
        print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())) + ' - finish evaluation')
        

                    
