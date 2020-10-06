import tensorflow as tf
import numpy as np
import cPickle as pickle
from utils.gumbel_softmax import gumbel_softmax
import utils.layers as layers
import utils.operations as op


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

            self.is_pretrain_calibration = tf.placeholder(tf.bool)

            self.is_pretrain_matching = tf.placeholder(tf.bool)

            self.is_joint_learning= tf.placeholder(tf.bool)

            # -------------------- Data Calibration Model ------------------- #
            self.word_embedding = tf.get_variable(
                name='word_embedding',
                shape=[self._conf['vocab_size'] + 1, self._conf['emb_size']],
                dtype=tf.float32,
                initializer=word_embedding_initializer)

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

            # define operations
            # response part
            c_Hr = tf.nn.embedding_lookup(self.word_embedding, self._response)

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
                c_Hu = tf.nn.embedding_lookup(self.word_embedding, c_turn_t)  # [batch, max_turn_len, emb_size]

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
                    c_sim = tf.einsum('biks,bjks->bijs', c_t_a_r, c_r_a_t) / tf.sqrt(200.0)

                c_sim_turns.append(c_sim)

            # cnn and aggregation
            c_sim = tf.stack(c_sim_turns, axis=1)
            print('sim shape: %s' % c_sim.shape)
            with tf.variable_scope('c_cnn_aggregation'):
                # for douban
                c_final_info = layers.CNN_3d(c_sim, 16, 16)

            # define operations
            # response part
            m_Hr = tf.nn.embedding_lookup(self.word_embedding, self._response)

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
                m_Hu = tf.nn.embedding_lookup(self.word_embedding, m_turn_t)  # [batch, max_turn_len, emb_size]

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

                # calculate similarity matrix
                with tf.variable_scope('m_similarity'):
                    # sim shape [batch, max_turn_len, max_turn_len, 2*stack_num+1]
                    # divide sqrt(200) to prevent gradient explosion
                    m_sim = tf.einsum('biks,bjks->bijs', m_t_a_r, m_r_a_t) / tf.sqrt(200.0)

            m_sim_turns.append(m_sim)

            # cnn and aggregation
            m_sim = tf.stack(m_sim_turns, axis=1)
            print('sim shape: %s' % m_sim.shape)
            with tf.variable_scope('m_cnn_aggregation'):
                # for douban
                m_final_info = layers.CNN_3d(m_sim, 16, 16)
                # loss and train

            # loss and train
            with tf.variable_scope('c_loss'):
                self.c_loss, self.c_logits, self.c_y_pred = layers.loss(c_final_info, self._label)
                self.c_gumbel_softmax = gumbel_softmax(self.c_logits, hard=False)
                self.c_gumbel_softmax_label = gumbel_softmax(self.c_logits, hard=True)
                self.c_global_step = tf.Variable(0, trainable=False)
                c_initial_learning_rate = self._conf['learning_rate']
                self.c_learning_rate = tf.train.exponential_decay(
                    c_initial_learning_rate,
                    global_step=self.c_global_step,
                    decay_steps=400,
                    decay_rate=0.9,
                    staircase=True)

                c_Optimizer = tf.train.AdamOptimizer(self.c_learning_rate)
                self.c_optimizer = c_Optimizer.minimize(
                    self.c_loss,
                    global_step=self.c_global_step)

                #self.c_saver = tf.train.Saver(max_to_keep=self._conf["max_to_keep"])
                self.c_all_variables = tf.global_variables()
                self.c_all_operations = self._graph.get_operations()
                self.c_grads_and_vars = c_Optimizer.compute_gradients(self.c_loss)
                self.c_target_grads_and_vars = []
                self.c_target_var_names = []
                self.c_trainable_variables = tf.trainable_variables()


                for grad, var in self.c_grads_and_vars:
                    if grad is not None:
                        print(var)
                        self.c_target_var_names.append(var)
                        self.c_target_grads_and_vars.append((grad, var))
                if len(self.c_target_var_names) != 0:
                    print('=' * 60 + '\n' + 'c target grads and vars' + '\n' + '=' * 60)

                self.c_capped_gvs = [(tf.clip_by_value(grad, -1, 1), var) for grad, var in
                                        self.c_target_grads_and_vars]
                self.c_g_updates = c_Optimizer.apply_gradients(
                    self.c_capped_gvs,
                    global_step=self.c_global_step)

            with tf.variable_scope('variables'):
                self.t_vars = tf.trainable_variables()
                self.c_vars = [var for var in self.t_vars if 'c_' in var.name]
                self.m_vars = [var for var in self.t_vars if 'm_' in var.name]

            with tf.variable_scope('m_loss'):
                #TODO - SHEllY
                if self.calibration_type == tf.constant(0):
                    c_label = tf.cast(tf.argmax(self.c_y_pred, axis=1), tf.float32)
                    target_label = tf.cond(tf.equal(self.is_pretrain_matching, tf.constant(True)), lambda: self._label, lambda: c_label)
                elif self.calibration_type == tf.constant(2):
                    target_label = tf.cond(tf.equal(self.is_pretrain_matching, tf.constant(True)), lambda: self._label,lambda: self.c_logits[:, -1])
                else:
                    target_label = tf.cond(tf.equal(self.is_pretrain_matching, tf.constant(True)), lambda: self._label,lambda: self.c_y_pred[:, -1])

                self.m_loss, self.m_logits, self.m_y_pred = layers.loss(m_final_info, target_label)
                self.m_gumbel_softmax = gumbel_softmax(self.m_logits, hard=False)
                self.m_gumbel_softmax_label = gumbel_softmax(self.m_logits, hard=True)
                self.m_global_step = tf.Variable(0, trainable=False)
                m_initial_learning_rate = self._conf['learning_rate']
                self.m_learning_rate = tf.train.exponential_decay(
                    m_initial_learning_rate,
                    global_step=self.m_global_step,
                    decay_steps=400,
                    decay_rate=0.9,
                    staircase=True)

                m_Optimizer = tf.train.AdamOptimizer(self.m_learning_rate)
                self.m_optimizer = m_Optimizer.minimize(
                    self.m_loss,
                    global_step=self.m_global_step)

                self.init = tf.global_variables_initializer()
                self.saver = tf.train.Saver(max_to_keep=self._conf["max_to_keep"])
                self.m_all_variables = tf.global_variables()
                self.m_all_operations = self._graph.get_operations()
                self.m_grads_and_vars = m_Optimizer.compute_gradients(self.m_loss)
                self.m_target_grads_and_vars = []
                self.m_target_var_names = []

                for grad, var in self.m_grads_and_vars:
                    if grad is not None:
                        print(var)
                        self.m_target_var_names.append(var)
                        self.m_target_grads_and_vars.append((grad, var))

                if len(self.m_target_var_names) != 0:
                    print('=' * 60 + '\n' + 'm target grads and vars' + '\n' + '=' * 60)

                self.m_capped_gvs = [(tf.clip_by_value(grad, -1, 1), var) for grad, var in self.m_target_grads_and_vars]
                self.m_g_updates = m_Optimizer.apply_gradients(
                    self.m_capped_gvs,
                    global_step=self.m_global_step)

        return self._graph

