import sys
import os
import time
import random

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
    average_loss, step, best_result, avg_acc = 0.0, 0, 0.0, 0.0

    for epoch in xrange(conf['calibration_pretrain_epoch']):
        #starting shuffle train data
        shuffle_train = reader.unison_shuffle(train_data)
        train_batches = reader.build_batches(shuffle_train, conf)
        for batch_index in range(batch_num):
            _feed = {
                _model.is_pretrain_calibration: True,
                _model.is_pretrain_matching: False,
                _model.is_backprop_calibration: False,
                _model.is_backprop_matching: False,
                _model.calibration_type: conf['calibration_type'],
                _model._turns: train_batches["turns"][batch_index],
                _model._tt_turns_len: train_batches["tt_turns_len"][batch_index],
                _model._every_turn_len: train_batches["every_turn_len"][batch_index],
                _model._response: train_batches["response"][batch_index],
                _model._response_len: train_batches["response_len"][batch_index],
                _model._label: train_batches["label"][batch_index]
            }

            _, _curr_loss = _sess.run([_model.g_updates, _model.c_loss], feed_dict=_feed)
            average_loss += _curr_loss
            step += 1

            if step % conf["print_step"] == 0 and step > 0:
                _g_step, _lr = _sess.run([_model.global_step, _model.learning_rate])
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
                        _model.is_backprop_calibration: False,
                        _model.is_backprop_matching: False,
                        _model.calibration_type: conf['calibration_type'],
                        _model._turns: dev_batches["turns"][batch_index],
                        _model._tt_turns_len: dev_batches["tt_turns_len"][batch_index],
                        _model._every_turn_len: dev_batches["every_turn_len"][batch_index],
                        _model._response: dev_batches["response"][batch_index],
                        _model._response_len: dev_batches["response_len"][batch_index],
                        _model._label: dev_batches["label"][batch_index]
                    }

                    _, _y_pred = _sess.run([_model.c_logits, _model.c_y_pred], feed_dict=_feed)
                    #avg_acc += c_accuracy
                    label_list.extend(dev_batches["label"][batch_index])
                    _y_pred_list.extend(list(_y_pred[:, -1]))
                # write evaluation result
                #avg_acc = avg_acc / dev_batch_num
                result = eva.evaluate_auc(_y_pred_list, label_list)
                print('[Pretrain Calibration] Epoch %d - AUC: %.3f' % (epoch, result))
                if result > best_result:
                    best_result = result
                    save_path = _model.saver.save(_sess, conf["save_path"] + "pretrain_calibration_model.ckpt." + str(
                        step / conf["save_step"]))
                    _pretrain_update_model_save_name = "pretrain_calibration_model.ckpt." + str(int(step / conf["save_step"]))
                    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())) + " - success saving model - " + _pretrain_update_model_save_name + " - in " + save_path)
                if step >= conf["calibration_max_step"]:
                    break

    return _pretrain_update_model_save_name

def _pretrain_matching(_sess, _graph, _model, conf, train_data, dev_batches):
    _pretrain_update_model_save_name = None
    #conf
    batch_num = int(len(train_data['y']) / conf["batch_size"])
    dev_batch_num = len(dev_batches["response"])
    conf["train_steps"] = conf['matching_pretrain_epoch'] * batch_num
    conf["save_step"] = int(max(1, batch_num / 10))
    conf["print_step"] = int(max(1, batch_num / 100))

    # Indicators
    average_loss, step, best_result, avg_accuracy = 0.0, 0, 0.0, 0.0

    for epoch in xrange(conf['matching_pretrain_epoch']):
        #starting shuffle train data
        shuffle_train = reader.unison_shuffle(train_data)
        train_batches = reader.build_batches(shuffle_train, conf)
        for batch_index in range(batch_num):

            _feed = {
                _model.is_pretrain_calibration: False,
                _model.is_pretrain_matching: True,
                _model.is_backprop_calibration: False,
                _model.is_backprop_matching: False,
                _model.calibration_type: conf['calibration_type'],
                _model._turns: train_batches["turns"][batch_index],
                _model._tt_turns_len: train_batches["tt_turns_len"][batch_index],
                _model._every_turn_len: train_batches["every_turn_len"][batch_index],
                _model._response: train_batches["response"][batch_index],
                _model._response_len: train_batches["response_len"][batch_index],
                _model._label: train_batches["label"][batch_index]
            }

            _, _curr_loss = _sess.run([ _model.g_updates, _model.m_loss], feed_dict=_feed)

            #print(refine_label)
            #print(_label)
            average_loss += _curr_loss

            step += 1

            if step % conf["print_step"] == 0 and step > 0:
                _g_step, _lr = _sess.run([_model.global_step, _model.learning_rate])
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
                        _model.is_backprop_calibration: False,
                        _model.is_backprop_matching: False,
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
                print('[Pretrain Matching] Epoch %d - AUC: %.3f' % (epoch, result))
                if result > best_result:
                    best_result = result
                    save_path = _model.saver.save(_sess, conf["save_path"] + "pretrain_matching_model.ckpt." + str(
                        step / conf["save_step"]))
                    _pretrain_update_model_save_name = "pretrain_matching_model.ckpt." + str(int(step / conf["save_step"]))
                    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())) + " - success saving model - " + _pretrain_update_model_save_name + " - in " + save_path)
                if step >= conf["matching_max_step"]:
                    break
    return _pretrain_update_model_save_name

def train(conf, _model):
    final_model_save_name = None
    if conf['rand_seed'] is not None:
        np.random.seed(conf['rand_seed'])

    if not os.path.exists(conf['save_path']):
        os.makedirs(conf['save_path'])

    # load vocab
    id_to_word = {}
    word_to_id = {}
    with open(conf['word_to_id'], 'r') as files:
        word2id = files.readlines()
        for line in word2id:
            id_to_word[line.strip().split('\t')[1]] = line.strip().split('\t')[0]
            word_to_id[line.strip().split('\t')[0]] = line.strip().split('\t')[1]

    # load data
    print(str(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))) + ' - start loading data')
    train_data, test_data, validation_data = pickle.load(open(conf["data_path"], 'rb'))
    dev_batches = reader.build_batches(validation_data, conf)
    test_batches = reader.build_batches(test_data, conf)
    print(str(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))) + ' - Finish Data Pre-processing')

    #Print configuration setting
    print('configurations: %s' %conf)

    _graph = _model.build_graph()

    print('=' * 60 + '\n' + 'Calibration Network Pre-training' + '\n' + '=' * 60)
    with tf.Session(graph=_graph) as pre_c_sess:
        tf.global_variables_initializer().run()

        if conf["init_model"] is not None:
            _model.saver.restore(pre_c_sess, os.path.join(conf["save_path"], conf["init_model"]))
            print("success init model %s" % str(os.path.join(conf["save_path"], conf["init_model"])))

        _pretrain_model_name = _pretrain_calibration(pre_c_sess, _graph, _model, conf, train_data, dev_batches)
        if _pretrain_model_name != '' or None: conf["init_model"] = str(_pretrain_model_name)
        print('Pretrained Model Save Name (Calibration): %s' % (str(conf["init_model"])))

    tf.reset_default_graph()

    print('=' * 60 + '\n' + 'Matching Network Pre-training' + '\n' + '=' * 60)
    with tf.Session(graph=_graph) as pre_m_sess:
        tf.global_variables_initializer().run()

        if conf["init_model"] is not None:
            _model.saver.restore(pre_m_sess, os.path.join(conf["save_path"], conf["init_model"]))
            print("success init model %s" % str(os.path.join(conf["save_path"], conf["init_model"])))

        _pretrain_model_name = _pretrain_matching(pre_m_sess, _graph, _model, conf, train_data, dev_batches)
        if _pretrain_model_name != '' or None: conf["init_model"] = str(_pretrain_model_name)
        print('Pretrained Model Save Name (Matching): %s' % (str(conf["init_model"])))

    tf.reset_default_graph()

    print('=' * 60 + '\n' + 'Joint Training' + '\n' + '=' * 60)
    with tf.Session(graph=_graph) as _sess:
        tf.global_variables_initializer().run()

        if conf["init_model"] is not None:
            _model.saver.restore(_sess, os.path.join(conf["save_path"], conf["init_model"]))
            print("success init model %s" % str(os.path.join(conf["save_path"], conf["init_model"])))

        # refine conf
        batch_num = int(len(train_data['y']) / conf["batch_size"])
        dev_batch_num = len(dev_batches["response"])
        test_batch_num = len(test_batches["response"])

        conf["train_steps"] = conf["num_scan_data"] * batch_num
        conf["save_step"] = int(max(1, batch_num / 10))
        conf["print_step"] = int(max(1, batch_num / 100))

        # Indicators
        matching_loss, calibration_loss = 0.0, 0.0
        step, best_result = 0, 0.0
        #c_summaries, m_summaries = None, None
        for epoch in xrange(conf["num_scan_data"]):
            data_calibration_log = []
            print('starting shuffle train data')
            shuffle_train = reader.unison_shuffle(train_data)
            train_batches = reader.build_batches(shuffle_train, conf)
            print('finish building train data')

            for batch_index in range(batch_num):

                # -------------------- Matching Model Optimisation ------------------- #
                _feed = {
                    _model.is_pretrain_calibration: False,
                    _model.is_pretrain_matching: False,
                    _model.is_backprop_calibration: False,
                    _model.is_backprop_matching: True,
                    _model.calibration_type: conf['calibration_type'],
                    _model._turns: train_batches["turns"][batch_index],
                    _model._tt_turns_len: train_batches["tt_turns_len"][batch_index],
                    _model._every_turn_len: train_batches["every_turn_len"][batch_index],
                    _model._response: train_batches["response"][batch_index],
                    _model._response_len: train_batches["response_len"][batch_index],
                    _model._label: train_batches["label"][batch_index]
                }

                #m_loss, g_updates = _sess.run([_model.m_loss, _model.g_updates], feed_dict=_feed)
                c_y_pred, refine_label, m_loss, g_updates = _sess.run([_model.c_y_pred, _model.refine_label, _model.m_loss, _model.g_updates], feed_dict=_feed)
                for j in range(conf['batch_size']):
                    q = ' '.join([id_to_word[i]for i in train_batches["turns"][batch_index][j][0] if i != 0 and i != 1 and i in word2id])
                    r = ' '.join([id_to_word[i]for i in list(train_batches["response"][batch_index][j]) if i != 0 and i != 1 and i in word2id])
                    o_l = train_batches["label"][batch_index][j]
                    r_l = list(refine_label)[j]
                    label_1_prob = '%.3f'%(c_y_pred[j][-1])
                    #isSame = str(o_l == r_l)
                    #data_calibration_log.append('\t'.join([q, r, o_l, r_l, label_1_prob, isSame]))
                    data_calibration_log.append([q, r, o_l, r_l, label_1_prob])
                matching_loss += m_loss

                # -------------------- Calibration Model Optimisation ------------------- #
                if step % conf['validation_step'] == 0:
                    shuffle_validation_data = reader.unison_shuffle(validation_data)
                    validation_batches = reader.build_batches(shuffle_validation_data, conf)
                    validation_batch_num = len(validation_batches["response"])
                    if conf['validation_update_batch_percentage'] != 1.0:
                        sample_num = int(validation_batch_num * conf['validation_update_batch_percentage'])
                        scalar = [i for i in range(validation_batch_num)]
                        sample_batch_number = random.sample(scalar, sample_num)
                    else:
                        scalar = [i for i in range(validation_batch_num)]
                        sample_batch_number = scalar
                    for validation_batch_index in sample_batch_number:
                    #for validation_batch_index in xrange(validation_batch_num):
                        _feed = {
                            _model.is_pretrain_calibration: False,
                            _model.is_pretrain_matching: False,
                            _model.is_backprop_calibration: True,
                            _model.is_backprop_matching: False,
                            _model.calibration_type: conf['calibration_type'],
                            _model._turns: validation_batches["turns"][validation_batch_index],
                            _model._tt_turns_len: validation_batches["tt_turns_len"][validation_batch_index],
                            _model._every_turn_len: validation_batches["every_turn_len"][validation_batch_index],
                            _model._response: validation_batches["response"][validation_batch_index],
                            _model._response_len: validation_batches["response_len"][validation_batch_index],
                            _model._label: validation_batches["label"][validation_batch_index]
                        }

                        total_loss, g_updates = _sess.run([_model.total_loss, _model.g_updates], feed_dict=_feed)

                        calibration_loss += total_loss

                step += 1

                if step % conf["print_step"] == 0 and step > 0:
                    g_step, lr = _sess.run([_model.global_step, _model.learning_rate])
                    print("processed: [%.4f]" % (float(step * 1.0 / batch_num)))
                    print("[Joint Model] - step: %d , lr: %f , m_loss: [%.6f], c_loss: [%.6f]" %(
                            g_step, lr, (matching_loss / conf["print_step"]), (calibration_loss / conf["print_step"])))
                    matching_loss, calibration_loss = 0.0, 0.0

                if step % conf["save_step"] == 0 and step > 0:
                    index = step / conf['save_step']
                    print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())) + ' - save step: %s' %index)
                    #average_calibrate_rate, average_calibrated_correctness = 0.0, 0.0
                    m_label_list, c_label_list, m_y_pred_list, c_y_pred_list = [], [], [], []
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
                            _model._label: test_batches["label"][batch_index]
                        }

                        _, m_y_pred = _sess.run([_model.m_logits, _model.m_y_pred], feed_dict=_feed)
                        #if conf['calibration_loss_type'] == 'hinge':
                        #    calibrated_label = [str(int(l)) for l in c_y_pred]
                        #elif conf['calibration_loss_type'] == 'cross_entropy':
                        #    calibrated_label = ['1' if scores[1] > scores[0] else '0' for scores in c_y_pred]
                        #origin_label = ['1', '0'] * int(len(calibrated_label) / 2)
                        #calibrated_rate = 1 - accuracy_score(calibrated_label, origin_label)
                        #calibrated_correctness = accuracy_score(calibrated_label, dev_batches["label"][batch_index])
                        #average_calibrate_rate += calibrated_rate
                        #average_calibrated_correctness += calibrated_correctness
                        m_label_list.extend(dev_batches["label"][batch_index])
                        m_y_pred_list.extend(list(m_y_pred[:, -1]))

                    #print('Data Calibration Rate: %.4f' % (average_correction_rate/dev_batch_num))
                    result = eva.evaluate_auc(m_y_pred_list, m_label_list)
                    #loss_str = "%.6f" %(m_loss)
                    #print('Epoch %d - Calibrate rate: %.4f; correctness: %.4f, Matching Auc: %.4f' %(epoch, (average_calibrate_rate/dev_batch_num), (average_calibrated_correctness/dev_batch_num), result))
                    print('Epoch %d - Matching Auc: %.4f' %(epoch, result))
                    if result > best_result:
                        best_result = result
                        save_path = _model.saver.save(_sess, conf["save_path"] + "joint_learning_model.ckpt." + str(int((step / conf["save_step"]))))
                        print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())) + " - finish evaluation - success saving model in " + save_path)
                        final_model_save_name = save_path
            fname = 'data.calibration.epoch.{}.txt'.format(epoch)
            with open(os.path.join(conf['save_path'], fname), 'w') as o_f:
                for log in data_calibration_log:
                    o_f.write(log[0] + '\t' + log[1] + '\t' + str(log[2]) + '\t' + str(int(log[3])) + '\t' + log[4] +'\n')

            # Update the Matching model variables to the calibration model when one epoch end
            t_vars = tf.trainable_variables()
            var_in_m_model = [var for var in t_vars if 'm_' in var.name]
            var_in_c_model = [var for var in t_vars if 'c_' in var.name]
            #print(len(var_in_m_model))
            #print(len(var_in_c_model))

            #for index, var in enumerate(var_in_m_model):
            #    print(index, var)
            #for index, var in enumerate(var_in_c_model):
            #    print(index, var)
            if conf['update_cmodel_epoch_end']:
                for var_idx, var in enumerate(var_in_m_model):
                    #print(var_idx)
                    #print(var)
                    #print(var_in_c_model[var_idx])
                    #print(var_in_m_model[var_idx].eval()[0])
                    #print(var_in_c_model[var_idx].eval()[0])
                    var_in_c_model[var_idx].load(var_in_m_model[var_idx].eval(), _sess)
                    #print(var_in_c_model[var_idx].eval()[0])
                    #print('-' * 50)
    return final_model_save_name

