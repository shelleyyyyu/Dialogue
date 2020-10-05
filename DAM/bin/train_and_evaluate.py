import sys
import os
import time

import cPickle as pickle
import tensorflow as tf
import numpy as np

from utils.gumbel_softmax import gumbel_softmax
import utils.reader as reader
import utils.evaluation as eva
from sklearn.metrics import accuracy_score

def _pretrain_calibration(_sess, _graph, _model, conf, train_data, dev_batches):
    _pretrain_update_model_save_name = None
    #conf
    batch_num = int(len(train_data['y']) / conf["batch_size"])
    dev_batch_num = len(dev_batches["response"])
    conf["train_steps"] = conf['calibration_pretrain_epoch'] * batch_num
    conf["save_step"] = int(max(1, batch_num / 10))
    conf["print_step"] = int(max(1, batch_num / 100))

    # Indicators
    average_loss, step, best_result = 0.0, 0, 0.0

    for epoch in xrange(conf['calibration_pretrain_epoch']):
        #starting shuffle train data
        shuffle_train = reader.unison_shuffle(train_data)
        train_batches = reader.build_batches(shuffle_train, conf)
        for batch_index in range(batch_num):
            _feed = {
                _model.is_pretrain_calibration: True,
                _model.is_pretrain_matching: True,
                _model.is_joint_learning: False,
                _model.calibration_type: conf['calibration_type'],
                _model._turns: train_batches["turns"][batch_index],
                _model._tt_turns_len: train_batches["tt_turns_len"][batch_index],
                _model._every_turn_len: train_batches["every_turn_len"][batch_index],
                _model._response: train_batches["response"][batch_index],
                _model._response_len: train_batches["response_len"][batch_index],
                _model._label: train_batches["label"][batch_index]
            }

            _, _curr_loss, _logits, _y_pred = _sess.run([_model.c_g_updates, _model.c_loss, _model.c_logits, _model.c_y_pred], feed_dict=_feed)
            average_loss += _curr_loss

            step += 1

            if step % conf["print_step"] == 0 and step > 0:
                _g_step, _lr = _sess.run([_model.c_global_step, _model.c_learning_rate])
                print('Pretrain Calibration Model - step: %s, lr: %s' % (_g_step, _lr))
                print("processed: [%.4f] loss [%.4f]" %(float(step * 1.0 / batch_num), float(average_loss / conf["print_step"])))
                average_loss = 0
            if step % conf["save_step"] == 0 and step > 0:
                _y_pred_list = []
                label_list = []
                index = step / conf['save_step']
                print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())) + ' - save step: %s' % index)
                for batch_index in xrange(dev_batch_num):
                    _feed = {
                        _model.is_pretrain_calibration: True,
                        _model.is_pretrain_matching: False,
                        _model.is_joint_learning: False,
                        _model.calibration_type: conf['calibration_type'],
                        _model._turns: dev_batches["turns"][batch_index],
                        _model._tt_turns_len: dev_batches["tt_turns_len"][batch_index],
                        _model._every_turn_len: dev_batches["every_turn_len"][batch_index],
                        _model._response: dev_batches["response"][batch_index],
                        _model._response_len: dev_batches["response_len"][batch_index],
                        _model._label: dev_batches["label"][batch_index]
                    }

                    _, _y_pred = _sess.run([_model.c_logits, _model.c_y_pred], feed_dict=_feed)
                    label_list.extend(dev_batches["label"][batch_index])
                    _y_pred_list.extend(list(_y_pred[:,-1]))
                # write evaluation result
                result = eva.evaluate_auc(_y_pred_list, label_list)
                print('[Pretrain Calibration] Epoch %d - Accuracy: %.3f' % (epoch, result))
                if result > best_result:
                    best_result = result
                    save_path = _model.saver.save(_sess, conf["save_path"] + "pretrain_calibration_model.ckpt." + str(
                        step / conf["save_step"]))
                    _pretrain_update_model_save_name = "pretrain_calibration_model.ckpt." + str(int(step / conf["save_step"]))
                    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())) + " - success saving model - " + _pretrain_update_model_save_name + " - in " + save_path)
                if step >= conf["calibration_max_step"]:
                    break
    return _model, _pretrain_update_model_save_name

def _pretrain_matching(_sess, _graph, _model, conf, train_data, dev_batches):
    _pretrain_update_model_save_name = None
    #conf
    batch_num = int(len(train_data['y']) / conf["batch_size"])
    dev_batch_num = len(dev_batches["response"])
    conf["train_steps"] = conf['matching_pretrain_epoch'] * batch_num
    conf["save_step"] = int(max(1, batch_num / 10))
    conf["print_step"] = int(max(1, batch_num / 100))

    # Indicators
    average_loss, step, best_result = 0.0, 0, 0.0

    for epoch in xrange(conf['matching_pretrain_epoch']):
        #starting shuffle train data
        shuffle_train = reader.unison_shuffle(train_data)
        train_batches = reader.build_batches(shuffle_train, conf)
        for batch_index in range(batch_num):

            _feed = {
                _model.is_pretrain_calibration: False,
                _model.is_pretrain_matching: True,
                _model.is_joint_learning: False,
                _model.calibration_type: conf['calibration_type'],
                _model._turns: train_batches["turns"][batch_index],
                _model._tt_turns_len: train_batches["tt_turns_len"][batch_index],
                _model._every_turn_len: train_batches["every_turn_len"][batch_index],
                _model._response: train_batches["response"][batch_index],
                _model._response_len: train_batches["response_len"][batch_index],
                _model._label: train_batches["label"][batch_index]
            }

            _, _curr_loss, _logits, _y_pred = _sess.run([_model.m_g_updates, _model.m_loss, _model.m_logits, _model.m_y_pred], feed_dict=_feed)
            average_loss += _curr_loss

            step += 1

            if step % conf["print_step"] == 0 and step > 0:
                _g_step, _lr = _sess.run([_model.m_global_step, _model.m_learning_rate])
                print('Pretrain Matching Model - step: %s, lr: %s' % (_g_step, _lr))
                print("processed: [%.4f] loss [%.4f]" %(float(step * 1.0 / batch_num), float(average_loss / conf["print_step"])))
                average_loss = 0
            if step % conf["save_step"] == 0 and step > 0:
                _y_pred_list = []
                label_list = []
                index = step / conf['save_step']
                print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())) + ' - save step: %s' % index)
                for batch_index in xrange(dev_batch_num):
                    _feed = {
                        _model.is_pretrain_calibration: False,
                        _model.is_pretrain_matching: True,
                        _model.is_joint_learning: False,
                        _model.calibration_type: conf['calibration_type'],
                        _model._turns: dev_batches["turns"][batch_index],
                        _model._tt_turns_len: dev_batches["tt_turns_len"][batch_index],
                        _model._every_turn_len: dev_batches["every_turn_len"][batch_index],
                        _model._response: dev_batches["response"][batch_index],
                        _model._response_len: dev_batches["response_len"][batch_index],
                        _model._label: dev_batches["label"][batch_index]
                    }

                    _, _y_pred = _sess.run([_model.m_logits, _model.m_y_pred], feed_dict=_feed)
                    label_list.extend(dev_batches["label"][batch_index])
                    _y_pred_list.extend(list(_y_pred[:,-1]))
                # write evaluation result
                result = eva.evaluate_auc(_y_pred_list, label_list)
                print('[Pretrain Matching] Epoch %d - Accuracy: %.3f' % (epoch, result))
                if result > best_result:
                    best_result = result
                    save_path = _model.saver.save(_sess, conf["save_path"] + "pretrain_matching_model.ckpt." + str(
                        step / conf["save_step"]))
                    _pretrain_update_model_save_name = "pretrain_matching_model.ckpt." + str(int(step / conf["save_step"]))
                    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())) + " - success saving model - " + _pretrain_update_model_save_name + " - in " + save_path)
                if step >= conf["matching_max_step"]:
                    break
    return _model, _pretrain_update_model_save_name

def train(conf, _model):
    
    if conf['rand_seed'] is not None:
        np.random.seed(conf['rand_seed'])

    if not os.path.exists(conf['save_path']):
        os.makedirs(conf['save_path'])

    # load data
    #print(str(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))) + ' - start loading data')
    train_data, dev_data, test_data, validation_data = pickle.load(open(conf["data_path"], 'rb'))
    #print(str(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))) + ' - finish loading data')

    # load dev_batches
    #print(str(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))) + " - start building dev/validation batches")
    dev_batches = reader.build_batches(dev_data, conf)
    validation_batches = reader.build_batches(validation_data, conf)
    #print(str(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))) + " - finish building dev/validation batches")
    print(str(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))) + ' - Finish Data Pre-processing')

    #Print configuration setting
    print('configurations: %s' %conf)

    _graph = _model.build_graph()
    #print(str(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))) + ' - build model graph success')
    #tensorboard_data = './log/DAM_JDQA'
    with tf.Session(graph=_graph) as _sess:
        #train_summary_writer = tf.summary.FileWriter(tensorboard_data, _sess._graph)
        tf.global_variables_initializer().run()

        if conf["init_model"] is not None:
            _model.saver.restore(_sess, conf["init_model"])
            print("success init model %s" % conf["init_model"])

        print('=' * 60 + '\n' + 'Calibration Network Pre-training' + '\n' + '=' * 60)
        _model, _pretrain_model_name = _pretrain_calibration(_sess, _graph, _model, conf, train_data, dev_batches)
        if _pretrain_model_name != '' or None: conf["init_model"] = str(_pretrain_model_name)
        print('Pretrained Model Save Name (Calibration): %s' %(str(_pretrain_model_name)))

        print('=' * 60 + '\n' + 'Matching Network Pre-training' + '\n' + '=' * 60)
        _model, _pretrain_model_name = _pretrain_matching(_sess, _graph, _model, conf, train_data, dev_batches)
        if _pretrain_model_name != '' or None: conf["init_model"] = str(_pretrain_model_name)
        print('Pretrained Model Save Name (Matching): %s' %(str(_pretrain_model_name)))

        # refine conf
        batch_num = int(len(train_data['y']) / conf["batch_size"])
        dev_batch_num = len(dev_batches["response"])
        validation_batch_num = len(validation_batches["response"])

        conf["train_steps"] = conf["num_scan_data"] * batch_num
        conf["save_step"] = int(max(1, batch_num / 10))
        conf["print_step"] = int(max(1, batch_num / 100))

        # Indicators
        c_average_m_loss, c_average_c_loss, m_average_m_loss, m_average_c_loss, average_correction_rate, step, best_result = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        #c_summaries, m_summaries = None, None
        for epoch in xrange(conf["num_scan_data"]):
            print('starting shuffle train data')
            shuffle_train = reader.unison_shuffle(train_data)
            train_batches = reader.build_batches(shuffle_train, conf)
            print('finish building train data')
            for batch_index in range(batch_num):

                # -------------------- Matching Model Optimisation ------------------- #
                _feed = {
                    _model.is_pretrain_calibration: False,
                    _model.is_pretrain_matching: False,
                    _model.is_joint_learning: True,
                    _model.calibration_type: conf['calibration_type'],
                    _model._turns: train_batches["turns"][batch_index],
                    _model._tt_turns_len: train_batches["tt_turns_len"][batch_index],
                    _model._every_turn_len: train_batches["every_turn_len"][batch_index],
                    _model._response: train_batches["response"][batch_index],
                    _model._response_len: train_batches["response_len"][batch_index],
                    _model._label: train_batches["label"][batch_index]
                }

                #c_loss_summary = tf.summary.scalar("train/matching_opt/c_loss", _model.c_loss)
                #m_loss_summary = tf.summary.scalar("train/matching_opt/m_loss", _model.m_loss)
                #train_summary_m_op = tf.summary.merge([c_loss_summary, m_loss_summary])

                m_target_var_name, c_curr_loss, c_logits, c_y_pred, c_gumbel_softmax , m_g_updates, m_curr_loss, m_logits, m_y_pred = \
                    _sess.run([_model.m_target_var_names, _model.c_loss, _model.c_logits, _model.c_y_pred,
                               _model.c_gumbel_softmax, _model.m_g_updates, _model.m_loss,
                               _model.m_logits, _model.m_y_pred], feed_dict=_feed)

                #if batch_index == 0:
                #    print('[1st Step] Length of optimised variables: %d' %len(m_target_var_name))
                #calibrated_label = ['1' if scores[1] > scores[0] else '0' for scores in c_gumbel_softmax]
                #calibrated_rate = 1 - accuracy_score(calibrated_label, train_batches["label"][batch_index])

                #average_correction_rate += calibrated_rate
                m_average_m_loss += m_curr_loss
                m_average_c_loss += c_curr_loss

                #batch_index = (batch_index + 1) % batch_num

                # -------------------- Calibration Model Optimisation ------------------- #
                for validation_batch_index in xrange(validation_batch_num):
                    _feed = {
                        _model.is_pretrain_calibration: False,
                        _model.is_pretrain_matching: False,
                        _model.is_joint_learning: True,
                        _model.calibration_type: conf['calibration_type'],
                        _model._turns: validation_batches["turns"][validation_batch_index],
                        _model._tt_turns_len: validation_batches["tt_turns_len"][validation_batch_index],
                        _model._every_turn_len: validation_batches["every_turn_len"][validation_batch_index],
                        _model._response: validation_batches["response"][validation_batch_index],
                        _model._response_len: validation_batches["response_len"][validation_batch_index],
                        _model._label: validation_batches["label"][validation_batch_index]
                    }

                    #c_loss_summary = tf.summary.scalar("train/calibration_opt/c_loss", _model.c_loss)
                    #m_loss_summary = tf.summary.scalar("train/calibration_opt/m_loss", _model.m_loss)
                    #train_summary_c_op = tf.summary.merge([c_loss_summary, m_loss_summary])

                    c_target_var_names, c_curr_loss, c_logits, c_y_pred, c_gumbel_softmax, c_g_updates, m_curr_loss, m_logits, m_y_pred = \
                        _sess.run([_model.c_target_var_names, _model.c_loss,
                                   _model.c_logits, _model.c_y_pred, _model.c_gumbel_softmax,
                                   _model.c_g_updates, _model.m_loss, _model.m_logits, _model.m_y_pred],
                                  feed_dict=_feed)
                    #if batch_index == 0 and validation_batch_index == 0:
                    #    print('[2nd Step] Length of optimised variables: %d' % len(c_target_var_names))
                    c_average_m_loss += m_curr_loss
                    c_average_c_loss += c_curr_loss

                step += 1

                if step % conf["print_step"] == 0 and step > 0:
                    c_g_step, c_lr = _sess.run([_model.c_global_step, _model.c_learning_rate])
                    m_g_step, m_lr = _sess.run([_model.m_global_step, _model.m_learning_rate])
                    print("processed: [%.4f]" % (float(step * 1.0 / batch_num)))
                    print("[Matching Model Optimisation] - step: %d , lr: %f , c_loss: [%.4f] m_loss: [%.4f]" %(m_g_step, m_lr, (m_average_c_loss / conf["print_step"]), (m_average_m_loss / conf["print_step"])))
                    print("[Calibration Model Optimisation] - step: %d , lr: %f , c_loss: [%.4f] m_loss: [%.4f]" %(c_g_step, c_lr, (c_average_c_loss / (conf["print_step"]*validation_batch_num)), (c_average_m_loss / (conf["print_step"]*validation_batch_num))))
                    c_average_m_loss, c_average_c_loss, m_average_m_loss, m_average_c_loss, average_correction_rate = 0.0, 0.0, 0.0, 0.0, 0.0
                    #if m_summaries:
                    #    train_summary_writer.add_summary(m_summaries, step)
                    #if c_summaries:
                    #    train_summary_writer.add_summary(c_summaries, step)

                if step % conf["save_step"] == 0 and step > 0:
                    index = step / conf['save_step']
                    print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())) + ' - save step: %s' %index)
                    average_correction_rate = 0.0
                    label_list, _y_pred_list = [], []
                    for batch_index in xrange(dev_batch_num):

                        _feed = {
                            _model.is_pretrain_calibration: False,
                            _model.is_pretrain_matching: False,
                            _model.is_joint_learning: True,
                            _model.calibration_type: conf['calibration_type'],
                            _model._turns: dev_batches["turns"][batch_index],
                            _model._tt_turns_len: dev_batches["tt_turns_len"][batch_index],
                            _model._every_turn_len: dev_batches["every_turn_len"][batch_index],
                            _model._response: dev_batches["response"][batch_index],
                            _model._response_len: dev_batches["response_len"][batch_index],
                            _model._label: dev_batches["label"][batch_index]
                        }

                        c_logits, c_y_pred, m_logits, m_y_pred, c_gumbel_softmax = _sess.run(
                            [_model.c_logits, _model.c_y_pred, _model.m_logits, _model.m_y_pred, _model.c_gumbel_softmax], feed_dict=_feed)

                        calibrated_label = ['1' if scores[1] > scores[0] else '0' for scores in c_gumbel_softmax]
                        calibrated_rate = 1 - accuracy_score(calibrated_label, dev_batches["label"][batch_index])

                        average_correction_rate += calibrated_rate

                        label_list.extend(dev_batches["label"][batch_index])
                        _y_pred_list.extend(list(m_y_pred[:, -1]))
                        #batch_index = (batch_index + 1) % batch_num
                        # write evaluation result
                    print('Data Calibration Rate: %.4f' % (average_correction_rate/dev_batch_num))
                    result = eva.evaluate_auc(_y_pred_list, label_list)
                    print('Epoch %d - Accuracy: %.3f' %(epoch, result))
                    if result > best_result:
                        best_result = result
                        save_path = _model.saver.save(_sess, conf["save_path"] + "joint_learning_model.ckpt." + str(int((step / conf["save_step"]))))
                        print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())) + " - finish evaluation - success saving model in " + save_path)

                

