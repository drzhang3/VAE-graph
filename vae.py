import functools
import tensorflow as tf
from tensorflow.python.layers.core import dense
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
import numpy as np


def lazy_scope(function):
    """Creates a decorator for methods that makes their return values load lazily.

    A method with this decorator will only compute the return value once when called
    for the first time. Afterwards, the value will be cached as an object attribute.
    Inspired by: https://danijar.com/structuring-your-tensorflow-models

    Args:
        function (func): Function to be decorated.

    Returns:
        decorator: Decorator for the function.
    """
    attribute = "_cache_" + function.__name__

    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            with tf.variable_scope(function.__name__):
                setattr(self, attribute, function(self))
        return getattr(self, attribute)

    return decorator
def compute_similarity(node1, node2):
    temp = np.multiply(node1, node2)
    return np.sum(temp)/(np.sum(node1)*np.sum(node2))


def node_similarity(node_state):
    similarity = tf.matmul(node_state, tf.transpose(node_state))/tf.norm(node_state, axis=-1)
    return similarity


def get_z_e(value, seq_len=3, step=1):
    z_e_x = []
    z_e_y = []
    for i in range(0, value.shape[0] - seq_len, step):
        x = value[i:i + seq_len + 1]
        y = value[i:i + seq_len + 1]
        z_e_x.append(x[:-1])
        z_e_y.append(y[-1])
    return z_e_x, z_e_y, i+1


def init_adj_prob(n):
    mat = np.random.random([n, n])
    for i in range(n):
        mat[i][i] = 1
    return mat


def global_state():
    pass


def cid_ce(ts):
    x = np.diff(ts)
    res = np.sqrt(np.dot(x, x))
    return res


def compute_ts_complexity(ts):
    return cid_ce(ts)

class VAE():
    def __init__(self, batch_size, z_dim, seq_len, input_dim, hidden_dim, alpha, beta, gamma, eta, kappa, theta):
        self.batch_size = batch_size
        self.z_dim = z_dim
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.multihead_num = 3
        self.embedding_size = self.z_dim
        self.keep_prob = 0.9
        self.dropout = 0.5
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.eta = eta
        self.kappa = kappa
        self.theta = theta
        self.input_x = tf.placeholder(shape=[self.batch_size, input_dim, seq_len], dtype=tf.float32, name="input_x")
        self.global_step = tf.Variable(0, trainable=False, name="global_step")
        # self.batch_size = tf.shape(self.input_x)[0]
        # self.z_e, self.mu, self.sigma_2 = self.encoder()
        self.r_t_1, self.r_t_2, self.mu_, self.sigma_2_ = self.spike_vae()
        self.c_t = self.bernoulli()
        self.spike_z = self.spike_z()
        self.z_e = self.spike_z
        self.z_graph = self._graph()
        self.position = self.position_embedding()
        self.attn = self.attention()
        # self.z_graph = self.z_e
        self.x_hat_e = self.decoder_ze()
        self.x_hat_q = self.decoder_zq()
        self.x_hat = self.decoder_z()
        self.loss = self.get_loss()
        self.train_vae, self.train_op = self.optimize()

    def global_state(self):
        embeddings = tf.get_variable("embeddings", [self.batch_size, self.batch_size]+[self.z_dim],
                                     initializer=tf.truncated_normal_initializer(stddev=0.05))
        tf.summary.tensor_summary("embeddings", embeddings)
        return embeddings

    def encoder(self):
        with tf.variable_scope("encoder"):
            h_1 = Dense(2 * self.z_dim, activation=tf.nn.leaky_relu)(self.input_x)
            h_1 = Dropout(rate=self.dropout)(h_1)
            h_1 = BatchNormalization()(h_1)
            h_2 = Dense(self.z_dim, activation=tf.nn.leaky_relu)(h_1)
            h_2 = Dropout(rate=self.dropout)(h_2)
            h_2 = BatchNormalization()(h_2)
            with tf.name_scope("mu"):
                mu = Dense(self.z_dim)(h_2)
                mu = tf.reshape(mu, shape=[-1, self.z_dim])
            with tf.name_scope("sigma"):
                log_sigma_2 = Dense(self.z_dim)(h_2)
                sigma_2 = tf.exp(log_sigma_2)
                sigma_2 = tf.reshape(sigma_2, shape=[-1, self.z_dim])
                sigma = tf.sqrt(sigma_2)
            epsilon = tf.random_normal(shape=tf.shape(sigma), name="epsilon")
            with tf.name_scope("z_e"):
                z_e = mu + sigma * epsilon
        return z_e, mu, sigma_2

    def spike_vae(self):
        with tf.variable_scope("spike_encoder", reuse=tf.AUTO_REUSE):
            h_1 = Dense(2 * self.z_dim, activation=tf.nn.leaky_relu)(self.input_x)
            h_1 = Dropout(rate=self.dropout)(h_1)
            h_1 = BatchNormalization()(h_1)
            h_2 = Dense(self.z_dim, activation=tf.nn.leaky_relu)(h_1)
            h_2 = Dropout(rate=self.dropout)(h_2)
            h_2 = BatchNormalization()(h_2)
            with tf.name_scope("mu"):
                mu = Dense(self.z_dim)(h_2)
                mu = tf.reshape(mu, shape=[-1, self.z_dim])
            with tf.name_scope("sigma"):
                log_sigma_2 = Dense(self.z_dim)(h_2)
                sigma_2 = tf.exp(log_sigma_2)
                sigma_2 = tf.reshape(sigma_2, shape=[-1, self.z_dim])
                sigma = tf.sqrt(sigma_2)
            epsilon_1 = tf.random_normal(shape=tf.shape(sigma), name="epsilon1")
            epsilon_2 = tf.random_normal(shape=tf.shape(sigma), name="epsilon2")
            with tf.name_scope("r_t_1"):
                r_t_1 = sigma * epsilon_1 / 10
            with tf.name_scope("r_t_1"):
                r_t_2 = mu + sigma * epsilon_2
        return r_t_1, r_t_2, mu, sigma_2

    def bernoulli(self):
        with tf.variable_scope("bernoulli", reuse=tf.AUTO_REUSE):
            h_1 = Dense(2 * self.z_dim, activation=tf.nn.leaky_relu)(self.input_x)
            h_1 = Dropout(rate=self.dropout)(h_1)
            h_1 = BatchNormalization()(h_1)
            h_2 = Dense(self.z_dim, activation=tf.nn.leaky_relu)(h_1)
            h_2 = Dropout(rate=self.dropout)(h_2)
            # h_2 = BatchNormalization()(h_2)
            c_t = tf.reshape(h_2, shape=[-1, self.z_dim], name="mu")
            epsilon = tf.random.uniform(shape=tf.shape(c_t), minval=0, maxval=1, name="epsilon")
            with tf.name_scope("c_t"):
                c_t = tf.nn.sigmoid(tf.log(epsilon + 1e-10) - tf.log(1 - epsilon + 1e-10) + tf.log(c_t + 1e-10)
                                    + tf.log(1 - c_t + 1e-10))
        return c_t

    def spike_z(self):
        with tf.variable_scope("s_t", reuse=tf.AUTO_REUSE):
            s_t = tf.multiply(1-self.c_t, self.r_t_1) + tf.multiply(self.c_t, self.r_t_2)
        return s_t

    def spike_kl_loss(self):

        return loss_kl

    def decoder_ze(self):
        with tf.variable_scope("decoder_ze", reuse=tf.AUTO_REUSE):
            h_1 = Dense(2 * self.z_dim, activation=tf.nn.leaky_relu)(self.attn)
            h_2 = Dense(self.seq_len)(h_1)
            x_hat = tf.expand_dims(h_2, 1, name="result")
        return x_hat

    def decoder_zq(self):
        with tf.variable_scope("decoder_zq", reuse=tf.AUTO_REUSE):
            h_1 = Dense(2 * self.z_dim, activation=tf.nn.leaky_relu)(self.z_graph)
            h_2 = Dense(self.seq_len)(h_1)
            x_hat = tf.expand_dims(h_2, 1, name='result')
        return x_hat

    def decoder_z(self):
        with tf.variable_scope("decoder_z", reuse=tf.AUTO_REUSE):
            h_1 = Dense(2 * self.z_dim, activation=tf.nn.leaky_relu)(self.spike_z)
            h_2 = Dense(self.seq_len)(h_1)
            # h_2 = BatchNormalization()(h_2)
            x_hat = tf.expand_dims(h_2, 1, name='result')
        return x_hat

    def run_lstm_ze(self):
        with tf.variable_scope("run_lstm_ze", reuse=tf.AUTO_REUSE):
            z_e_x, z_e_y, num = get_z_e(self.z_graph)
            loss = []
            for i in range(num):
                outputs = self.lstm_vae(z_e_x[i])
                loss_ = tf.losses.mean_squared_error(outputs, z_e_y[i])
                loss.append(loss_)
            loss_res = tf.reduce_mean(loss, name="lstm_loss")
        return loss_res

    def lstm_vae(self, x_data):
        with tf.variable_scope('lstm_vae', reuse=tf.AUTO_REUSE):
            cell = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_dim)
            input_lstm = tf.expand_dims(x_data, 0, name="input_lstm")
            # cell = tf.contrib.rnn.AttentionCellWrapper(cell, attn_length = self.attention_len)
            outputs, _ = tf.nn.dynamic_rnn(cell, input_lstm, dtype=tf.float32)    # (batch_size, seq_len, z_dim)
            outputs = tf.transpose(outputs, [1, 0, 2])[-1]     # (1 , z_dim)
            outputs = dense(outputs, self.z_dim)     # (1, hidden_dim)
            outputs = tf.reshape(outputs, [self.z_dim])
        return outputs

    def attention(self):
        with tf.variable_scope("multi_head_attention", reuse=tf.AUTO_REUSE):
            final_embedding = self.position + self.z_e
            # final_embedding = self.z_e
            query = tf.expand_dims(final_embedding, 0)
            key_value = tf.expand_dims(final_embedding, 0)
            # (bs = 1, seq_len = batch_size, z_dim)
            # compute Q、K、V
            V = dense(key_value, units=self.embedding_size, use_bias=False, name='V')
            K = dense(key_value, units=self.embedding_size, use_bias=False, name='K')
            Q = dense(query, units=self.embedding_size, use_bias=False, name='Q')
            # (bs, seq_len, embedding_size)

            # multi-heads
            V = tf.concat(tf.split(V, self.multihead_num, axis=-1), axis=0)
            K = tf.concat(tf.split(K, self.multihead_num, axis=-1), axis=0)
            Q = tf.concat(tf.split(Q, self.multihead_num, axis=-1), axis=0)
            # (bs*head_num, seq_len, embedding_size/head_num)
            score = tf.matmul(Q, tf.transpose(K, [0, 2, 1])) / tf.sqrt(self.embedding_size / self.multihead_num)
            # (bs*head_num, seq_len, seq_len)
            # TODO : add mask
            # if score_mask is not None:
            #     score *= score_mask
            #     score += ((score_mask - 1) * 1e+9)
            # softmax
            softmax = tf.nn.softmax(score, axis=2)
            # (bs*head_num, seq_len, seq_len)
            # attention
            attention = tf.matmul(softmax, V)
            # (bs*head_num, seq_len, embedding_size/head_num)
            concat = tf.concat(tf.split(attention, self.multihead_num, axis=0), axis=-1)
            # (bs, seq_len, embedding_size)
            Multihead = dense(concat, units=self.embedding_size, use_bias=False, name='linear')
            # output mask
            # TODO : add mask for output
            # if output_mask is not None:
            #     Multihead *= output_mask

            Multihead = tf.nn.dropout(Multihead, keep_prob=self.keep_prob)
            Multihead += query
            # Layer Norm
            Multihead = tf.contrib.layers.layer_norm(Multihead, begin_norm_axis=2)
            Multihead = tf.reshape(Multihead, [-1, self.embedding_size])
            tf.add_to_collection("multi-head", Multihead)
        return Multihead

    def position_embedding(self):
        with tf.variable_scope("position_embedding"):
            position_size = self.embedding_size
            seq_len = tf.shape(self.z_e)[0]
            position_j = 1. / tf.pow(10000., 2 * tf.range(position_size / 2, dtype=tf.float32) / position_size)
            position_j = tf.expand_dims(position_j, 0)
            position_i = tf.range(tf.cast(seq_len, tf.float32), dtype=tf.float32)
            position_i = tf.expand_dims(position_i, 1)
            position_ij = tf.matmul(position_i, position_j)
            position_ij = tf.concat([tf.cos(position_ij), tf.sin(position_ij)], 1)
            # (seq_len, embedding_size)
            tf.add_to_collection("position", position_ij)
        return position_ij

    def _graph(self):
        with tf.variable_scope("gcn", reuse=tf.AUTO_REUSE):
            node_num = self.batch_size    # node_num = batch_size
            node_dim = self.z_dim
            node_state = self.z_e       # (node_num, node_dim)
            tf.add_to_collection("z_e", node_state)
            self.init_prob = tf.get_variable("prob", [node_num, node_num], initializer=tf.truncated_normal_initializer())
            weight = tf.Variable(tf.random_normal([node_dim, node_dim], stddev=0.35), dtype=tf.float32, name="weight")
            with tf.name_scope("similarity"):
                similarity = node_similarity(node_state)
                tf.add_to_collection("similarity", similarity)
            with tf.name_scope("final_output"):
                h_0 = tf.nn.sigmoid(tf.matmul(tf.matmul(self.init_prob, node_state), weight))     # (node_num, node_dim)
                h_1 = tf.matmul(tf.matmul(self.init_prob, h_0), weight)
                tf.add_to_collection("node_state", h_1)
            # loss = 0.00005*tf.reduce_sum(tf.square(self.init_prob - node_similarity(node_state)), name="w_loss")
        return h_1    # (batch_size, z_dim)
        # # GNN  second stage
        # weight_adj = tf.Variable(tf.random_normal([node_num, node_num], stddev=0.35), dtype=tf.float32)
        # similarity = node_similarity(node_num, h_0)     # (node_num, node_num)
        # tf.add_to_collection("similarity", similarity)
        # P_0 = weight_adj
        # for i in range(max_iteration):
        #     P_0 = tf.nn.sigmoid(tf.matmul(tf.matmul(tf.matmul(P_0, similarity), node_state)), learning_matrix)
        # return P_0

    def loss_reconstruction(self):
        with tf.variable_scope("loss_reconstruction"):
            with tf.name_scope("zq_loss"):
                loss_rec_mse_zq = tf.reduce_sum(tf.square(self.input_x-self.x_hat_q))
            with tf.name_scope("vae_loss"):
                loss_rec_mse_z = tf.reduce_sum(tf.square(self.input_x-self.x_hat), axis=-1)
                elbo_c = -tf.reduce_sum(-tf.log(2.0)-self.c_t*tf.log(self.c_t+1e-20)-(1.0-self.c_t)*tf.log(
                            1.0-self.c_t+1e-20), axis=-1)
                elbo_r1 = -0.5*tf.reduce_sum(1.0-100*tf.square(self.mu_)-self.sigma_2_+tf.log(self.sigma_2_), axis=-1)
                elbo_r2 = -0.5*tf.reduce_sum(1.0-tf.square(self.mu_)-self.sigma_2_+tf.log(self.sigma_2_), axis=-1)
                loss_kl = elbo_c + elbo_r1 + elbo_r2
                # loss_kl = - 0.5*tf.reduce_sum(1.0-tf.square(self.mu_)-self.sigma_2_+tf.log(self.sigma_2_), axis=-1)
                vae_loss = tf.reduce_mean(loss_rec_mse_z, name="vae_loss")
            with tf.name_scope("ze_loss"):
                loss_rec_mse_ze = tf.reduce_sum(tf.square(self.input_x-self.x_hat_e))
        loss_rec_mse = vae_loss + loss_rec_mse_zq + loss_rec_mse_ze
        tf.summary.scalar("loss_reconstruction", loss_rec_mse)
        return loss_rec_mse

    @lazy_scope
    def loss_vae(self):
        with tf.name_scope("loss_vae"):
            loss_rec_mse_z = tf.reduce_sum(tf.square(self.input_x - self.x_hat), axis=-1)
            elbo_c = -tf.reduce_sum(-tf.log(2.0) - self.c_t * tf.log(self.c_t + 1e-20) - (1.0 - self.c_t) * tf.log(
                1.0 - self.c_t + 1e-20), axis=-1)
            elbo_r1 = -0.5 * tf.reduce_sum(1.0 - 100 * tf.square(self.mu_) - self.sigma_2_ + tf.log(self.sigma_2_),
                                           axis=-1)
            elbo_r2 = -0.5 * tf.reduce_sum(1.0 - tf.square(self.mu_) - self.sigma_2_ + tf.log(self.sigma_2_), axis=-1)
            loss_kl = elbo_c + elbo_r1 + elbo_r2
            # loss_kl = - 0.5*tf.reduce_sum(1.0-tf.square(self.mu_)-self.sigma_2_+tf.log(self.sigma_2_), axis=-1)
            vae_loss = tf.reduce_mean(loss_rec_mse_z, name="vae_loss")
            tf.summary.scalar("loss_vae", vae_loss)
        return vae_loss

    @lazy_scope
    def loss_ze(self):
        with tf.name_scope("loss_ze"):
            loss_rec_mse_ze = tf.reduce_sum(tf.square(self.input_x - self.x_hat_e))
            tf.summary.scalar("loss_ze", loss_rec_mse_ze)
        return loss_rec_mse_ze

    @lazy_scope
    def loss_zq(self):
        with tf.name_scope("loss_zq"):
            loss_rec_mse_zq = tf.reduce_sum(tf.square(self.input_x - self.x_hat_q))
            tf.summary.scalar("loss_zq", loss_rec_mse_zq)
        return loss_rec_mse_zq

    @lazy_scope
    def loss_commitment(self):
        with tf.variable_scope("loss_commit"):
            # loss_commit = tf.reduce_mean(tf.squared_difference(self.attn, self.z_graph))
            loss_commit1 = tf.reduce_mean(tf.squared_difference(self.z_e, self.z_graph))
            loss_commit2 = tf.reduce_mean(tf.squared_difference(self.z_e, self.attn))
            loss_commit = loss_commit1 + loss_commit2
        tf.summary.scalar("loss_commit", loss_commit)
        return loss_commit

    @lazy_scope
    def loss_graph(self):
        with tf.variable_scope("loss_graph"):
            loss_graph = self.run_lstm_ze()
        tf.summary.scalar("loss_graph", loss_graph)
        return loss_graph

    def get_loss(self):
        with tf.variable_scope("total_loss"):
            # loss = self.loss_reconstruction + self.loss_commitment()
            loss = self.alpha * self.loss_vae + self.beta * self.loss_ze + self.gamma * self.loss_zq +\
                   self.eta * self.loss_commitment + self.kappa * self.loss_graph
            # loss = tf.reduce_mean(loss, name='loss')
        tf.summary.scalar("loss", loss)
        return loss

    def optimize(self):
        with tf.variable_scope("optimize"):
            starter_learning_rate = 0.005
            learning_rate = tf.train.exponential_decay(starter_learning_rate, self.global_step,
                                                       20, 0.9, staircase=True)
            optimizer = tf.train.AdamOptimizer(learning_rate)
            # grads, variables = zip(*optimizer.compute_gradients(self.loss))
            # grads, global_norm = tf.clip_by_global_norm(grads, 5)
            # train_op = optimizer.apply_gradients(zip(grads, variables), global_step=self.global_step)
            train_vae = optimizer.minimize(self.loss_reconstruction, self.global_step)
            train_op = optimizer.minimize(self.loss, self.global_step)
        return train_vae, train_op
