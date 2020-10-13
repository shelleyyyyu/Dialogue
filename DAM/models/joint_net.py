import tensorflow as tf
import numpy as np
import cPickle as pickle
from utils.gumbel_softmax import gumbel_softmax
import utils.layers as layers
import utils.operations as op
import sys

class Net(object):
    '''Add positional encoding(initializer lambda is 0),
       cross-attention, cnn integrated and grad clip by value.

    Attributes:
        conf: a configuration parameters dict
        word_embedding_init: a 2-d array with shape [vocab_size+1, emb_size]
    '''

    def __init__(self, conf):
        self._graph = tf.Graph()
        self._conf = conf

        if self._conf['word_emb_init'] is not None:
            print('loading word emb init')
            self.word_embedding_init = pickle.load(open(self._conf['word_emb_init'], 'rb'))
        else:
            self.word_embedding_init = None

    def build_graph(self):
        with self._graph.as_default():

            # --- Share Parameter: rand_seed ---
            if self._conf['rand_seed'] is not None:
                rand_seed = self._conf['rand_seed']
                tf.set_random_seed(rand_seed)
                print('set tf random seed: %s' % self._conf['rand_seed'])

            # --- Share Parameter: word_embedding_init ---
            # word embedding
            if self.word_embedding_init is not None:
                word_embedding_initializer = tf.constant_initializer(self.word_embedding_init)
            else:
                word_embedding_initializer = tf.random_normal_initializer(stddev=0.1)

            self.c_word_embedding = tf.get_variable(
                name='c_word_embedding',
                shape=[self._conf['vocab_size'] + 1, self._conf['emb_size']],
                dtype=tf.float32,
                initializer=word_embedding_initializer)

            self.m_word_embedding = tf.get_variable(
                name='m_word_embedding',
                shape=[self._conf['vocab_size'] + 1, self._conf['emb_size']],
                dtype=tf.float32,
                initializer=word_embedding_initializer)

            self.min_threshold = tf.get_variable(
                name='min_threshold',
                shape=[],
                initializer=tf.constant_initializer(0.1))

            self.max_threshold = tf.get_variable(
                name='max_threshold',
                shape=[],
                initializer=tf.constant_initializer(0.9))

            # define placehloders
            self._turns = tf.placeholder(
                tf.int32,
                shape=[self._conf["batch_size"], self._conf["max_turn_num"], self._conf["max_turn_len"]])

            self._tt_turns_len = tf.placeholder(
                tf.int32,
                shape=[self._conf["batch_size"]])

            self._every_turn_len = tf.placeholder(
                tf.int32,
                shape=[self._conf["batch_size"], self._conf["max_turn_num"]])

            self._response = tf.placeholder(
                tf.int32,
                shape=[self._conf["batch_size"], self._conf["max_turn_len"]])

            self._response_len = tf.placeholder(
                tf.int32,
                shape=[self._conf["batch_size"]])

            self._label = tf.placeholder(
                tf.float32,
                shape=[self._conf["batch_size"]])

            self.calibration_type = tf.placeholder(
                tf.int32,
                shape=[])

            self.is_pretrain_calibration = tf.placeholder(tf.bool)

            self.is_pretrain_matching = tf.placeholder(tf.bool)

            self.is_backprop_calibration= tf.placeholder(tf.bool)

            self.is_backprop_matching= tf.placeholder(tf.bool)


            # ========== Calibration Network ==========

            # define operations
            # response part
            c_Hr = tf.nn.embedding_lookup(self.c_word_embedding, self._response)

            if self._conf['is_positional'] and self._conf['c_stack_num'] > 0:
                with tf.variable_scope('c_positional'):
                    c_Hr = op.positional_encoding_vector(c_Hr, max_timescale=10)
            c_Hr_stack = [c_Hr]

            for index in range(self._conf['c_stack_num']):
                with tf.variable_scope('c_self_stack_' + str(index)):
                    c_Hr = layers.block(
                        c_Hr, c_Hr, c_Hr,
                        Q_lengths=self._response_len, K_lengths=self._response_len)
                    c_Hr_stack.append(c_Hr)

            # context part
            # a list of length max_turn_num, every element is a tensor with shape [batch, max_turn_len]
            c_list_turn_t = tf.unstack(self._turns, axis=1)
            c_list_turn_length = tf.unstack(self._every_turn_len, axis=1)

            c_sim_turns = []
            # for every turn_t calculate matching vector
            for c_turn_t, c_t_turn_length in zip(c_list_turn_t, c_list_turn_length):
                c_Hu = tf.nn.embedding_lookup(self.c_word_embedding, c_turn_t)  # [batch, max_turn_len, emb_size]

                if self._conf['is_positional'] and self._conf['c_stack_num'] > 0:
                    with tf.variable_scope('c_positional', reuse=True):
                        c_Hu = op.positional_encoding_vector(c_Hu, max_timescale=10)
                c_Hu_stack = [c_Hu]

                for index in range(self._conf['c_stack_num']):
                    with tf.variable_scope('c_self_stack_' + str(index), reuse=True):
                        c_Hu = layers.block(
                            c_Hu, c_Hu, c_Hu,
                            Q_lengths=c_t_turn_length, K_lengths=c_t_turn_length)

                        c_Hu_stack.append(c_Hu)

                c_r_a_t_stack = []
                c_t_a_r_stack = []
                for index in range(self._conf['c_stack_num'] + 1):

                    with tf.variable_scope('c_t_attend_r_' + str(index)):
                        try:
                            c_t_a_r = layers.block(
                                c_Hu_stack[index], c_Hr_stack[index], c_Hr_stack[index],
                                Q_lengths=c_t_turn_length, K_lengths=self._response_len)
                        except ValueError:
                            tf.get_variable_scope().reuse_variables()
                            c_t_a_r = layers.block(
                                c_Hu_stack[index], c_Hr_stack[index], c_Hr_stack[index],
                                Q_lengths=c_t_turn_length, K_lengths=self._response_len)

                    with tf.variable_scope('c_r_attend_t_' + str(index)):
                        try:
                            c_r_a_t = layers.block(
                                c_Hr_stack[index], c_Hu_stack[index], c_Hu_stack[index],
                                Q_lengths=self._response_len, K_lengths=c_t_turn_length)
                        except ValueError:
                            tf.get_variable_scope().reuse_variables()
                            c_r_a_t = layers.block(
                                c_Hr_stack[index], c_Hu_stack[index], c_Hu_stack[index],
                                Q_lengths=self._response_len, K_lengths=c_t_turn_length)

                    c_t_a_r_stack.append(c_t_a_r)
                    c_r_a_t_stack.append(c_r_a_t)

                c_t_a_r_stack.extend(c_Hu_stack)
                c_r_a_t_stack.extend(c_Hr_stack)

                c_t_a_r = tf.stack(c_t_a_r_stack, axis=-1)
                c_r_a_t = tf.stack(c_r_a_t_stack, axis=-1)

                # calculate similarity matrix
                with tf.variable_scope('c_similarity'):
                    # sim shape [batch, max_turn_len, max_turn_len, 2*c_stack_num+1]
                    # divide sqrt(200) to prevent gradient explosion
                    c_sim = tf.einsum('biks,bjks->bijs', c_t_a_r, c_r_a_t) #/ tf.sqrt(200.0)

                c_sim_turns.append(c_sim)

            # cnn and aggregation
            c_sim = tf.stack(c_sim_turns, axis=1)
            print('sim shape: %s' % c_sim.shape)
            with tf.variable_scope('c_cnn_aggregation'):
                # for douban
                c_final_info = layers.CNN_3d(c_sim, 16, 16)

            # ========== Matching Network ==========

            # define operations
            # response part
            m_Hr = tf.nn.embedding_lookup(self.m_word_embedding, self._response)

            if self._conf['is_positional'] and self._conf['stack_num'] > 0:
                with tf.variable_scope('m_positional'):
                    m_Hr = op.positional_encoding_vector(m_Hr, max_timescale=10)
            m_Hr_stack = [m_Hr]

            for index in range(self._conf['stack_num']):
                with tf.variable_scope('m_self_stack_' + str(index)):
                    m_Hr = layers.block(
                        m_Hr, m_Hr, m_Hr,
                        Q_lengths=self._response_len, K_lengths=self._response_len)
                    m_Hr_stack.append(m_Hr)

            # context part
            # a list of length max_turn_num, every element is a tensor with shape [batch, max_turn_len]
            m_list_turn_t = tf.unstack(self._turns, axis=1)
            m_list_turn_length = tf.unstack(self._every_turn_len, axis=1)

            m_sim_turns = []
            # for every turn_t calculate matching vector
            for m_turn_t, m_t_turn_length in zip(m_list_turn_t, m_list_turn_length):
                m_Hu = tf.nn.embedding_lookup(self.m_word_embedding, m_turn_t)  # [batch, max_turn_len, emb_size]

                if self._conf['is_positional'] and self._conf['stack_num'] > 0:
                    with tf.variable_scope('m_positional', reuse=True):
                        m_Hu = op.positional_encoding_vector(m_Hu, max_timescale=10)
                m_Hu_stack = [m_Hu]

                for index in range(self._conf['stack_num']):
                    with tf.variable_scope('m_self_stack_' + str(index), reuse=True):
                        m_Hu = layers.block(
                            m_Hu, m_Hu, m_Hu,
                            Q_lengths=m_t_turn_length, K_lengths=m_t_turn_length)

                        m_Hu_stack.append(m_Hu)

                m_r_a_t_stack = []
                m_t_a_r_stack = []
                for index in range(self._conf['stack_num'] + 1):

                    with tf.variable_scope('m_t_attend_r_' + str(index)):
                        try:
                            m_t_a_r = layers.block(
                                m_Hu_stack[index], m_Hr_stack[index], m_Hr_stack[index],
                                Q_lengths=m_t_turn_length, K_lengths=self._response_len)
                        except ValueError:
                            tf.get_variable_scope().reuse_variables()
                            m_t_a_r = layers.block(
                                m_Hu_stack[index], m_Hr_stack[index], m_Hr_stack[index],
                                Q_lengths=m_t_turn_length, K_lengths=self._response_len)
                    with tf.variable_scope('m_r_attend_t_' + str(index)):
                        try:
                            m_r_a_t = layers.block(
                                m_Hr_stack[index], m_Hu_stack[index], m_Hu_stack[index],
                                Q_lengths=self._response_len, K_lengths=m_t_turn_length)
                        except ValueError:
                            tf.get_variable_scope().reuse_variables()
                            m_r_a_t = layers.block(
                                m_Hr_stack[index], m_Hu_stack[index], m_Hu_stack[index],
                                Q_lengths=self._response_len, K_lengths=m_t_turn_length)

                    m_t_a_r_stack.append(m_t_a_r)
                    m_r_a_t_stack.append(m_r_a_t)

                m_t_a_r_stack.extend(m_Hu_stack)
                m_r_a_t_stack.extend(m_Hr_stack)

                m_t_a_r = tf.stack(m_t_a_r_stack, axis=-1)
                m_r_a_t = tf.stack(m_r_a_t_stack, axis=-1)

                # 2 network representation fusion
                '''with tf.variable_scope('m_merge_rep'):

                    def f1():
                        return m_t_a_r, m_t_a_r, m_r_a_t, m_r_a_t
                    def f2():
                        return c_t_a_r, m_t_a_r, c_r_a_t, m_r_a_t

                    tar1, tar2, rat1, rat2 = tf.case({tf.equal(self.is_pretrain_calibration, tf.constant(True)): f1,
                                            tf.equal(self.is_pretrain_matching, tf.constant(True)): f1,
                                            tf.equal(self.is_backprop_matching, tf.constant(True)): f2,
                                            tf.equal(self.is_backprop_calibration, tf.constant(True)): f2},
                                default=f2, exclusive=False)


                    #Combine with the calibration infos
                    m_t_a_r = tf.reduce_mean(tf.concat([tf.expand_dims(tar1, 0), tf.expand_dims(tar2, 0)], axis=0), axis=0)
                    m_r_a_t = tf.reduce_mean(tf.concat([tf.expand_dims(rat1, 0), tf.expand_dims(rat2, 0)], axis=0), axis=0)'''

                # calculate similarity matrix
                with tf.variable_scope('m_similarity'):
                    # sim shape [batch, max_turn_len, max_turn_len, 2*stack_num+1]
                    # divide sqrt(200) to prevent gradient explosion
                    m_sim = tf.einsum('biks,bjks->bijs', m_t_a_r, m_r_a_t) #/ tf.sqrt(200.0)

                m_sim_turns.append(m_sim)
            # cnn and aggregation
            m_sim = tf.stack(m_sim_turns, axis=1)
            print('sim shape: %s' % m_sim.shape)
            with tf.variable_scope('m_cnn_aggregation'):
                # for douban
                m_final_info = layers.CNN_3d(m_sim, 16, 16)

            # loss and train
            with tf.variable_scope('loss'):
                # pass to linear transformation and softmax to get the logits and softmax-ed value y_pred
                self.c_loss, self.c_logits, self.c_y_pred = layers.calibration_loss(c_final_info, self._label, loss_type=self._conf['calibration_loss_type'])
                #self.c_correct = tf.equal(tf.cast(tf.argmax(self.c_y_pred, axis=1), tf.int32), tf.to_int32(self._label))
                #self.c_accuracy = tf.reduce_mean(tf.cast(self.c_correct, 'float'))

                # Use the c_y_pred abd define the calibrated label for the matching model (classifier)
                def calibrate_label(c_y_pred, true_labels):
                    target_label = []
                    for i in range(c_y_pred.shape[0]):
                        true = true_labels[i]

                        def fn1():
                            #return tf.cond(tf.greater(c_y_pred[i, -1], tf.cast(self._conf['positive_sample_threshold'], tf.float32)), lambda:tf.cast(tf.constant(1), tf.float32), lambda: tf.cast(tf.constant(0), tf.float32))
                            return tf.cast(tf.constant(1), tf.float32)

                        def fn2():
                            #return tf.cond(tf.greater(tf.cast(self._conf['negative_sample_threshold'], tf.float32), c_y_pred[i, -1]), lambda:tf.cast(tf.constant(0), tf.float32), lambda: tf.cast(tf.constant(1), tf.float32))
                            return tf.cast(tf.constant(0), tf.float32)

                        def fn3():
                            return true

                        #refine_label = \
                        #    tf.case({tf.greater(c_y_pred[i, -1], tf.cast(self._conf['positive_sample_threshold'], tf.float32)): fn1,
                        #             tf.greater(tf.cast(self._conf['negative_sample_threshold'], tf.float32), c_y_pred[i, -1]): fn2},
                        #            default=fn3, exclusive=False)

                        refine_label = \
                            tf.case({tf.greater(c_y_pred[i, -1], self.max_threshold): fn1,
                                     tf.greater(tf.cast(self.min_threshold, tf.float32),
                                                c_y_pred[i, -1]): fn2},
                                    default=fn3, exclusive=False)

                        target_label.append(refine_label)
                    return target_label

                def f_pretrain_matching():
                    #print(self._label)
                    return tf.cast(self._label, tf.int32)#, tf.constant(-1)
                def f_calibration_type_0():
                    target_label = calibrate_label(self.c_y_pred, self._label)
                    stacked_target_label = tf.stack(target_label)
                    #print(stacked_target_label)
                    return tf.cast(stacked_target_label, tf.int32)#, tf.constant(0)
                    #return tf.cast(tf.argmax(self.c_y_pred, axis=1), tf.int32)
                #def f_calibration_type_1():
                #    return self.c_y_pred[:, -1], tf.constant(1)
                #def f_calibration_type_2():
                #    return self.c_logits[:, -1], tf.constant(2)

                #target_label, self.shelly_test = tf.case({tf.equal(self.is_pretrain_matching, tf.constant(True)): f_pretrain_matching,
                #                        tf.equal(self.calibration_type, tf.constant(0)): f_calibration_type_0,
                #                        tf.equal(self.calibration_type, tf.constant(1)): f_calibration_type_1,
                #                        tf.equal(self.calibration_type, tf.constant(2)): f_calibration_type_2},
                #            default=f_pretrain_matching, exclusive=False)

                self.refine_label = tf.cond(tf.equal(self.is_pretrain_matching, tf.constant(True)), f_pretrain_matching, f_calibration_type_0)
                self.m_loss, self.m_logits, self.m_y_pred = layers.matching_loss(m_final_info, self.refine_label, loss_type=self._conf['matching_loss_type'])
                #self.m_correct = tf.equal(tf.cast(tf.argmax(self.m_y_pred, axis=1), tf.int32), tf.to_int32(self.refine_label))
                #self.m_accuracy = tf.reduce_mean(tf.cast(self.m_correct, 'float'))
                self.total_loss = self.m_loss+self.c_loss

                # Start update the network variable
                self.saver = tf.train.Saver(max_to_keep=self._conf["max_to_keep"])
                self.global_step = tf.Variable(0, trainable=False)
                initial_learning_rate = self._conf['learning_rate']
                self.learning_rate = tf.train.exponential_decay(
                    initial_learning_rate,
                    global_step=self.global_step,
                    decay_steps=self._conf['decay_steps'],
                    decay_rate=self._conf['decay_rate'],
                    staircase=True)

                Optimizer = tf.train.AdamOptimizer(self.learning_rate)

                def c_loss_fn():
                    self.optimizer = Optimizer.minimize(self.c_loss, global_step=self.global_step)
                    grads_and_vars = Optimizer.compute_gradients(self.c_loss)
                    target_grads_and_vars = []
                    for grad, var in grads_and_vars:
                        if grad is not None and ('c_' in var.name):
                            target_grads_and_vars.append((grad, var))
                    print(len(target_grads_and_vars))
                    capped_gvs = [(tf.clip_by_value(grad, -1, 1), var) for grad, var in target_grads_and_vars]
                    g_updates = Optimizer.apply_gradients(
                        capped_gvs,
                        global_step=self.global_step)
                    return g_updates, tf.constant(111)
                def m_loss_fn():
                    self.optimizer = Optimizer.minimize(self.m_loss, global_step=self.global_step)
                    grads_and_vars = Optimizer.compute_gradients(self.m_loss)
                    target_grads_and_vars = []
                    for grad, var in grads_and_vars:
                        if grad is not None and ('m_' in var.name):
                            target_grads_and_vars.append((grad, var))
                    print(len(target_grads_and_vars))
                    capped_gvs = [(tf.clip_by_value(grad, -1, 1), var) for grad, var in target_grads_and_vars]
                    g_updates = Optimizer.apply_gradients(
                        capped_gvs,
                        global_step=self.global_step)
                    return g_updates, tf.constant(222)
                def c_m_loss_fn():
                    self.optimizer = Optimizer.minimize(self.c_loss, global_step=self.global_step)
                    grads_and_vars = Optimizer.compute_gradients(self.c_loss)
                    target_grads_and_vars = []
                    for grad, var in grads_and_vars:
                        if grad is not None and ('c_' in var.name):
                            target_grads_and_vars.append((grad, var))
                    print(len(target_grads_and_vars))
                    capped_gvs = [(tf.clip_by_value(grad, -1, 1), var) for grad, var in target_grads_and_vars]
                    g_updates = Optimizer.apply_gradients(
                        capped_gvs,
                        global_step=self.global_step)
                    return g_updates, tf.constant(333)
                def c_m_loss_fn2():
                    self.optimizer = Optimizer.minimize(self.total_loss, global_step=self.global_step)
                    grads_and_vars = Optimizer.compute_gradients(self.total_loss)
                    target_grads_and_vars = []
                    for grad, var in grads_and_vars:
                        if grad is not None: #and ('m_' in var.name):
                            target_grads_and_vars.append((grad, var))
                    print(len(target_grads_and_vars))
                    capped_gvs = [(tf.clip_by_value(grad, -1, 1), var) for grad, var in target_grads_and_vars]
                    g_updates = Optimizer.apply_gradients(
                        capped_gvs,
                        global_step=self.global_step)
                    return g_updates, tf.constant(444)

                self.g_updates, test = tf.case({tf.equal(self.is_pretrain_calibration, tf.constant(True)): c_loss_fn,
                     tf.equal(self.is_pretrain_matching, tf.constant(True)): m_loss_fn,
                     tf.equal(self.is_backprop_calibration, tf.constant(True)): c_m_loss_fn,
                     tf.equal(self.is_backprop_matching, tf.constant(True)): c_m_loss_fn2},
                    default=c_m_loss_fn2, exclusive=False)

                '''t_vars = tf.trainable_variables()
                var_in_m_model = [var for var in t_vars if 'm_' in var.name]
                var_in_c_model = [var for var in t_vars if 'c_' in var.name]
                print(len(var_in_m_model))
                print(len(var_in_c_model))
                # for index, var in enumerate(var_in_m_model):
                #    print(index, var)
                # for index, var in enumerate(var_in_c_model):
                #    print(index, var)

                for var_idx, var in enumerate(var_in_m_model):
                    # print(var_idx)
                    # print(var)
                    # print(var_in_c_model[var_idx])
                    print('-' * 50)
                    tf.assign(var_in_c_model[var_idx], var)'''

        return self._graph

