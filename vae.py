import tensorflow as tf
from tensorflow.python.layers.core import dense
import numpy as np


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
    def __init__(self, batch_size, z_dim, seq_len, input_dim, hidden_dim):
        self.batch_size = batch_size
        self.z_dim = z_dim
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.input_x = tf.placeholder(shape=[self.batch_size, input_dim, seq_len], dtype=tf.float32, name="input_x")
        self.global_step = tf.Variable(0, trainable=False, name="global_step")
        # self.batch_size = tf.shape(self.input_x)[0]
        self.z_e, self.mu, self.sigma_2 = self.encoder()
        self.z_graph = self._graph()
        # self.z_graph = self.z_e
        self.x_hat_e = self.decoder_ze()
        self.x_hat_q = self.decoder_zq()
        self.loss = self.get_loss()
        self.train_op = self.optimize().minimize(self.loss, self.global_step)

    def global_state(self):
        embeddings = tf.get_variable("embeddings", [self.batch_size, self.batch_size]+[self.z_dim],
                                     initializer=tf.truncated_normal_initializer(stddev=0.05))
        tf.summary.tensor_summary("embeddings", embeddings)
        return embeddings

    def encoder(self):
        with tf.variable_scope("encoder"):
            h_1 = dense(self.input_x, 2*self.z_dim, activation=tf.nn.relu, name="h_1")
            h_2 = dense(h_1, self.z_dim, activation=tf.nn.relu, name="h_2")
            mu = dense(h_2, self.z_dim, name="mu")
            log_sigma_2 = dense(h_2, self.z_dim, name="log_sigma_2")
            sigma_2 = tf.exp(log_sigma_2)
            mu = tf.reshape(mu, shape=[-1, self.z_dim], name="mu")
            sigma_2 = tf.reshape(sigma_2, shape=[-1, self.z_dim], name="sigma_2")
            sigma = tf.sqrt(sigma_2, name="sigma")
            epsilon = tf.random_normal(shape=tf.shape(sigma), name="epsilon")
            z_e = tf.add(mu, sigma * epsilon, name="z_e")
        return z_e, mu, sigma_2

    def decoder_ze(self):
        with tf.variable_scope("decoder_ze", reuse=tf.AUTO_REUSE):
            h_3 = dense(self.z_e, 2*self.z_dim, activation=tf.nn.relu, name="h_3")
            h_4 = dense(h_3, self.seq_len, name="h_4")
            x_hat = tf.expand_dims(h_4, 1, name="result")
        return x_hat

    def decoder_zq(self):
        with tf.variable_scope("decoder_zq", reuse=tf.AUTO_REUSE):
            h_3 = dense(self.z_graph, 2*self.z_dim, tf.nn.relu, name="h_3")
            h_4 = dense(h_3, self.seq_len, name="h_4")
            x_hat = tf.expand_dims(h_4, 1, name='result')
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
            outputs =tf.reshape(outputs, [self.z_dim])
        return outputs

    def _graph(self):
        with tf.variable_scope("gcn", reuse=tf.AUTO_REUSE):
            node_num = self.batch_size    # node_num = batch_size
            node_dim = self.z_dim
            node_state = self.z_e       # (node_num, node_dim)
            tf.add_to_collection("z_e", node_state)
            self.init_prob = tf.get_variable("prob", [node_num, node_num],
                                             initializer=tf.orthogonal_initializer())
            weight = tf.Variable(tf.random_normal([node_dim, node_dim], stddev=0.35), dtype=tf.float32, name="weight")
            with tf.name_scope("similarity"):
                similarity = node_similarity(node_state)
                tf.add_to_collection("similarity", similarity)
            with tf.name_scope("final_output"):
                h_0 = tf.nn.sigmoid(tf.matmul(tf.matmul(self.init_prob, node_state), weight))     # (node_num, node_dim)
                h_1 = tf.nn.sigmoid(tf.matmul(tf.matmul(self.init_prob, h_0), weight))
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
                loss_rec_mse_ze = tf.reduce_sum(tf.square(self.input_x-self.x_hat_e), axis=-1)
                loss_kl = - 0.5 * tf.reduce_sum(1-tf.square(self.mu)-self.sigma_2+tf.log(self.sigma_2), axis=-1)
                vae_loss = tf.reduce_mean(loss_rec_mse_ze + loss_kl, name="vae_loss")
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
            loss = tf.reduce_mean(loss, name='loss')
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
