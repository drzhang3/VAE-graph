import tensorflow as tf
from tensorflow.python.layers.core import dense
import numpy as np


def z_loss(x, y):
    return tf.losses.mean_squared_error(x, y)


def compute_similarity(node1, node2):
    temp = np.multiply(node1, node2)
    return np.sum(temp)/(np.sum(node1)*np.sum(node2))


def node_similarity(node_num, node_state):
    similarity = np.zeros([node_num, node_num])
    # similarity = tf.convert_to_tensor(similarity)
    for i in range(node_num):
        for j in range(node_num):
            temp = np.dot(node_state[i], node_state[j])
            print(temp)
            # print(tf.reduce_sum(node_state[i]))
            # res = np.sum(temp) / (tf.reduce_sum(node_state[i]) * tf.reduce_sum(node_state[j]))
            similarity[i, j] = temp
    return similarity


def get_z_e(data, seq_len=3, step=1):
    z_e_x = []
    z_e_y = []
    for i in range(0, data.shape[0] - seq_len, step):
        x = data[i:i + seq_len + 1]
        y = data[i:i + seq_len + 1]
        z_e_x.append(x[:-1])
        z_e_y.append(y[-1])
    return z_e_x, z_e_y, i+1


def init_adj_prob(n):
    mat = np.random.random([n, n])
    for i in range(n):
        mat[i][i] = 1
    return mat


class VAE():
    def __init__(self, z_dim, seq_len, input_dim, hidden_dim):
        self.z_dim = z_dim
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.input_x = tf.placeholder(shape=[8, input_dim, seq_len], dtype=tf.float32, name="input_x")
        self.global_step = tf.Variable(0, trainable=False, name="global_step")
        self.batch_size = tf.shape(self.input_x)[0]
        self.z_e, self.mu, self.sigma_2 = self.encoder()
        self.z_graph = self._graph()
        # self.z_graph = self.z_e
        self.x_hat_e = self.decoder_ze()
        self.x_hat_q = self.decoder_zq()
        self.loss = self.get_loss()
        self.train_op = self.optimize().minimize(self.loss, self.global_step)

    def encoder(self):
        with tf.variable_scope("encoder"):
            h_1 = dense(self.input_x, 2*self.z_dim, activation=tf.nn.relu, name="h_1")
            h_2 = dense(h_1, self.z_dim, activation=tf.nn.relu, name="h_2")
            mu = dense(h_2, self.z_dim, name="mu")
            log_sigma_2 = dense(h_2, self.z_dim, name="sigma")
            sigma_2 = tf.exp(log_sigma_2)
            mu = tf.reshape(mu, shape=[-1, self.z_dim])
            sigma_2 = tf.reshape(sigma_2, shape=[-1, self.z_dim])
            sigma = tf.sqrt(sigma_2)
            epsilon = tf.random_normal(shape=tf.shape(sigma))
            z_e = mu + sigma * epsilon
        return z_e, mu, tf.square(sigma)

    def decoder_ze(self):
        with tf.variable_scope("decoder_ze", reuse=tf.AUTO_REUSE):
            h_3 = dense(self.z_e, 2*self.z_dim, activation=tf.nn.relu)
            x_hat = dense(h_3, self.seq_len)
            x_hat = tf.expand_dims(x_hat, 1, name="result")
        return x_hat

    def decoder_zq(self):
        with tf.variable_scope("decoder_zq", reuse=tf.AUTO_REUSE):
            h_3 = dense(self.z_graph, 2*self.z_dim, tf.nn.relu)
            x_hat = dense(h_3, self.seq_len)
            x_hat = tf.expand_dims(x_hat, 1, name='result')
        return x_hat

    def run_lstm_ze(self):
        z_e_x, z_e_y, num = get_z_e(self.z_graph)
        loss = []
        for i in range(num):
            outputs = self.lstm_vae(z_e_x[i])
            loss.append(z_loss(outputs, z_e_y[i]))
        return tf.reduce_mean(loss)

    def lstm_vae(self, x_data):
        with tf.variable_scope('lstm_vae', reuse=tf.AUTO_REUSE):
            cell = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_dim)
            input_lstm = tf.expand_dims(x_data, 0, name="input_lstm")
            # cell = tf.contrib.rnn.AttentionCellWrapper(cell, attn_length = self.attention_len)
            outputs, _ = tf.nn.dynamic_rnn(cell, input_lstm, dtype=tf.float32)    # (batch_size, seq_len, z_dim)
            outputs = tf.transpose(outputs, [1, 0, 2])[-1]     # (1 , z_dim)
            outputs = dense(outputs, self.z_dim)     # (1, hidden_dim)
            outputs =tf.reshape(outputs, [self.z_dim])
        return outputs

    def _graph(self):
        # with tf.variable_scope("gcn", reuse=tf.AUTO_REUSE):
        max_iteration = 3
        node_num = 8    # node_num = batch_size
        node_dim = self.z_dim
        node_state = self.z_e       # (node_num, node_dim)
        tf.add_to_collection("z_e", node_state)
        init_prob = tf.get_variable("prob", [node_num, node_num],
                                    initializer=tf.truncated_normal_initializer(stddev=0.05),
                                    constraint=lambda x: tf.clip_by_value(x, -1, 1))
        # init_prob = tf.get_variable("prob", [node_num, node_num],
        #                             initializer=tf.orthogonal_initializer(),
        #                             constraint=lambda x: tf.clip_by_value(x, -1, 1))
        weight = tf.Variable(tf.random_normal([node_dim, node_dim], stddev=0.35), dtype=tf.float32)
        learning_matrix = tf.Variable(tf.random_normal([node_dim, node_num], stddev=0.35), dtype=tf.float32)
        print(weight.get_shape())
        # GCN  first stage
        h_0 = node_state
        for i in range(max_iteration):
            h_0 = tf.nn.sigmoid(tf.matmul(tf.matmul(init_prob, h_0), weight))     # (node_num, node_dim)
        tf.add_to_collection("node_state", h_0)
        return h_0       # (batch_size, z_dim)
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
            loss_rec_mse_zq = tf.losses.mean_squared_error(self.input_x, self.x_hat_q)
            loss_rec_mse_ze = tf.reduce_sum(tf.square(self.input_x-self.x_hat_e), axis=-1)
            loss_kl = - 0.5 * tf.reduce_sum(1-tf.square(self.mu)-self.sigma_2+tf.log(self.sigma_2), axis=-1)
            vae_loss = tf.reduce_mean(loss_rec_mse_ze + loss_kl)
            loss_rec_mse = vae_loss + loss_rec_mse_zq
        tf.summary.scalar("loss_reconstruction", loss_rec_mse)
        return loss_rec_mse_ze

    def loss_commitment(self):
        with tf.variable_scope("loss_commit"):
            loss_commit = tf.reduce_mean(tf.squared_difference(self.z_e, self.z_graph))
        tf.summary.scalar("loss_commit", loss_commit)
        return loss_commit

    def loss_graph(self):
        with tf.variable_scope("loss_graph"):
            loss_graph = self.run_lstm_ze()
        tf.summary.scalar("loss_graph", loss_graph)
        return loss_graph

    def get_loss(self):
        with tf.variable_scope("loss"):
            loss = self.loss_reconstruction() + self.loss_commitment() + self.loss_graph()
        tf.summary.scalar("loss", loss)
        return loss

    def optimize(self):
        with tf.variable_scope("optimize"):
            starter_learning_rate = 0.005
            learning_rate = tf.train.exponential_decay(starter_learning_rate, self.global_step,
                                                       50, 0.9, staircase=True)
            optimizer = tf.train.AdamOptimizer(learning_rate)
            # optimizer = tf.train.AdadeltaOptimizer()
        return optimizer
