import sys
import os
import time

import cPickle as pickle
import tensorflow as tf
import numpy as np

import utils.reader as reader
import utils.douban_evaluation as eva
from sklearn.metrics import accuracy_score

def test(conf, m_model, c_model):

    if not os.path.exists(conf['save_path']):
        os.makedirs(conf['save_path'])

    # load data
    print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())) + ' - starting loading data')
    train_data, val_data, test_data = pickle.load(open(conf["data_path"], 'rb'))    
    print(str(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))) + ' - finish loading data')

    test_batches = reader.build_batches(test_data, conf)

    print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())) + " - finish building test batches")

    # refine conf
    test_batch_num = len(test_batches["response"])

    print('configurations: %s' %conf)

    c_graph = c_model.build_graph()
    print(str(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))) + ' - build calibration model graph success')
    m_graph = m_model.build_graph()
    print(str(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))) + ' - build matching model graph success')

    with tf.Session(graph=c_graph) as c_sess, tf.Session(graph=m_graph) as m_sess:

        # Init calibration model sess
        c_model.init.run(session=c_sess)
        if conf["calibration_init_model"]:
            c_model.saver.restore(c_sess, conf["calibration_init_model"])
            print("success init calibration model %s" %conf["calibration_init_model"])

        # Init matching model sess
        m_model.init.run(session=m_sess)
        if conf["matching_init_model"]:
            m_model.saver.restore(m_sess, conf["matching_init_model"])
            print("success init matching model %s" %conf["matching_init_model"])


        batch_index = 0
        step = 0
        average_correction_rate = 0.0

        score_file_path = conf['save_path'] + 'score.test'
        score_file = open(score_file_path, 'w')

        print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())) + ' - starting test')

        for batch_index in xrange(test_batch_num):

            # -------------------- Data Calibration Model ------------------- #
            c_feed = { 
                c_model.turns: test_batches["turns"][batch_index],
                c_model.tt_turns_len: test_batches["tt_turns_len"][batch_index],
                c_model.every_turn_len: test_batches["every_turn_len"][batch_index],
                c_model.response: test_batches["response"][batch_index],
                c_model.response_len: test_batches["response_len"][batch_index],
                c_model.label: test_batches["label"][batch_index]
                }

            _, c_y_pred = c_sess.run([c_model.logits, c_model.y_pred], feed_dict=c_feed)

            calibrated_label = ['1' if scores[1] > scores[0] else '0' for scores in c_y_pred]
            calibrated_rate = 1 - accuracy_score(calibrated_label, test_batches["label"][batch_index])
            average_correction_rate += calibrated_rate

            # -------------------- Matching Model ------------------- #
            m_feed = {
                m_model.turns: test_batches["turns"][batch_index],
                m_model.tt_turns_len: test_batches["tt_turns_len"][batch_index],
                m_model.every_turn_len: test_batches["every_turn_len"][batch_index],
                m_model.response: test_batches["response"][batch_index],
                m_model.response_len: test_batches["response_len"][batch_index],
                m_model.label: calibrated_label
                }
            m_logits, m_y_pred = m_sess.run([m_model.logits, m_model.y_pred], feed_dict=m_feed)
            
            for i in xrange(conf["batch_size"]):
                score_file.write(
                    str(m_y_pred[i][-1]) + '\t' +
                    str(test_batches["label"][batch_index][i]) + '\n')
        
        print('Data Calibration Rate: %.4f' % (average_correction_rate/test_batch_num))
        score_file.close()
        print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())) + ' - finish test')
        
        #write evaluation result
        result = eva.evaluate(score_file_path)
        result_file_path = conf["save_path"] + "result.test"
        with open(result_file_path, 'w') as out_file:
            for p_at in result:
                out_file.write(str(p_at) + '\n')
        print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())) + ' - finish evaluation')
        

                    
