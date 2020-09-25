import sys
import os
import time

import cPickle as pickle
import tensorflow as tf
import numpy as np

import utils.reader as reader
import utils.evaluation as eva
from sklearn.metrics import accuracy_score


def train(conf, m_model, c_model):
    
    if conf['rand_seed'] is not None:
        np.random.seed(conf['rand_seed'])

    if not os.path.exists(conf['save_path']):
        os.makedirs(conf['save_path'])

    # load data
    print(str(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))) + ' - starting loading data')
    train_data, val_data, test_data = pickle.load(open(conf["data_path"], 'rb'))
    print(str(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))) + ' - finish loading data')
    val_batches = reader.build_batches(val_data, conf)
    print(str(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))) + " - finish building test batches")

    # refine conf
    batch_num = int(len(train_data['y']) / conf["batch_size"])
    val_batch_num = len(val_batches["response"])

    conf["train_steps"] = conf["num_scan_data"] * batch_num
    conf["save_step"] = int(max(1, batch_num / 10))
    conf["print_step"] = int(max(1, batch_num / 100))

    print('configurations: %s' %conf)

    print(str(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))) + ' - model success')

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

        # Indicators
        average_loss = 0.0
        average_correction_rate = 0.0
        batch_index = 0
        step = 0
        best_result = [0, 0, 0, 0]

        for step_i in xrange(conf["num_scan_data"]):
            print('starting shuffle train data')
            shuffle_train = reader.unison_shuffle(train_data)
            train_batches = reader.build_batches(shuffle_train, conf)
            print('finish building train data')
            for batch_index in range(batch_num):

                # -------------------- Data Calibration Model ------------------- #
                c_feed = {
                    c_model.turns: train_batches["turns"][batch_index],
                    c_model.tt_turns_len: train_batches["tt_turns_len"][batch_index],
                    c_model.every_turn_len: train_batches["every_turn_len"][batch_index],
                    c_model.response: train_batches["response"][batch_index],
                    c_model.response_len: train_batches["response_len"][batch_index],
                    c_model.label: train_batches["label"][batch_index]
                }

                batch_index = (batch_index + 1) % batch_num

                #_, c_curr_loss, c_logits, c_y_pred = c_sess.run([c_model.g_updates, c_model.loss, c_model.logits, c_model.y_pred], feed_dict=c_feed)
                c_curr_loss, c_logits, c_y_pred = c_sess.run([c_model.loss, c_model.logits, c_model.y_pred], feed_dict=c_feed)

                calibrated_label = ['1' if scores[1] > scores[0] else '0' for scores in c_y_pred]
                calibrated_rate = 1 - accuracy_score(calibrated_label, train_batches["label"][batch_index])

                # -------------------- Matching Model ------------------- #
                m_feed = {

                    m_model.turns: train_batches["turns"][batch_index],
                    m_model.tt_turns_len: train_batches["tt_turns_len"][batch_index],
                    m_model.every_turn_len: train_batches["every_turn_len"][batch_index],
                    m_model.response: train_batches["response"][batch_index],
                    m_model.response_len: train_batches["response_len"][batch_index],
                    m_model.label: calibrated_label
                }
                _, m_curr_loss, m_logits, m_y_pred = m_sess.run([m_model.g_updates, m_model.loss, m_model.logits, m_model.y_pred], feed_dict=m_feed)

                average_correction_rate += calibrated_rate
                average_loss += m_curr_loss

                step += 1

                if step % conf["print_step"] == 0 and step > 0 and False:
                    m_g_step, m_lr = m_sess.run([m_model.global_step, m_model.learning_rate])
                    print('Matching Model - step: %s, lr: %s' %(m_g_step, m_lr))
                    print("processed: [" + str(step * 1.0 / batch_num) + "] loss: [" + str(average_loss / conf["print_step"]) + "] calibrated_rate: [" + '%.4f'%(average_correction_rate / conf["print_step"]) + ']')
                    print('-'*30)
                    average_loss = 0
                    average_correction_rate = 0

                
                if step % conf["save_step"] == 0 and step > 0:
                    index = step / conf['save_step']
                    score_file_path = conf['save_path'] + 'score.' + str(index)
                    score_file = open(score_file_path, 'w')
                    print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())) + ' - save step: %s' %index)
                    average_correction_rate = 0.0
                    for batch_index in xrange(val_batch_num):
                
                        # -------------------- Data Calibration Model ------------------- #
                        c_feed = {
                            c_model.turns: val_batches["turns"][batch_index],
                            c_model.tt_turns_len: val_batches["tt_turns_len"][batch_index],
                            c_model.every_turn_len: val_batches["every_turn_len"][batch_index],
                            c_model.response: val_batches["response"][batch_index],
                            c_model.response_len: val_batches["response_len"][batch_index],
                            c_model.label: val_batches["label"][batch_index]
                        }   
                
                        _, c_y_pred = c_sess.run([c_model.logits, c_model.y_pred], feed_dict=c_feed)

                        calibrated_label = ['1' if scores[1] > scores[0] else '0' for scores in c_y_pred]
                        calibrated_rate = 1 - accuracy_score(calibrated_label, val_batches["label"][batch_index])
                        average_correction_rate += calibrated_rate
                        # -------------------- Matching Model ------------------- #
                        m_feed = {
                            m_model.turns: val_batches["turns"][batch_index],
                            m_model.tt_turns_len: val_batches["tt_turns_len"][batch_index],
                            m_model.every_turn_len: val_batches["every_turn_len"][batch_index],
                            m_model.response: val_batches["response"][batch_index],
                            m_model.response_len: val_batches["response_len"][batch_index],
                            m_model.label: calibrated_label
                        }
                        m_logits, m_y_pred = m_sess.run([m_model.logits, m_model.y_pred], feed_dict=m_feed)
                        for i in xrange(conf["batch_size"]):
                            score_file.write(
                                str(m_y_pred[i][-1]) + '\t' +
                                str(val_batches["label"][batch_index][i]) + '\n')
                    print('Data Calibration Rate: %.4f' % (average_correction_rate/val_batch_num))
                    score_file.close()

                    #write evaluation result
                    result = eva.evaluate_auc(score_file_path)
                    result_file_path = conf["save_path"] + "result." + str(index)
                    with open(result_file_path, 'w') as out_file:
                        for p_at in result:
                            out_file.write(str(p_at) + '\n')
                    print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())) + ' - finish evaluation')

                    if result[0] + result[1] > best_result[0] + best_result[1]:
                        best_result = result
                        c_save_path = c_model.saver.save(c_sess, conf["save_path"] + "c_model.ckpt." + str(step / conf["save_step"]))
                        m_save_path = m_model.saver.save(m_sess, conf["save_path"] + "m_model.ckpt." + str(step / conf["save_step"]))
                        print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())) + " - success saving model in " + c_save_path + "(Data Calibration Model); " + m_save_path + " (Matching Model)")

                

