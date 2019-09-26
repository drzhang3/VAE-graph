import tensorflow as tf
from tensorflow.python.layers.core import dense
import numpy as np
import copy


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
        self.multihead_num = 3
        self.embedding_size = self.z_dim
        self.keep_prob = 0.9
        self.input_x = tf.placeholder(shape=[self.batch_size, input_dim, seq_len], dtype=tf.float32, name="input_x")
        self.global_step = tf.Variable(0, trainable=False, name="global_step")
        # self.batch_size = tf.shape(self.input_x)[0]
        self.z_e, self.mu, self.sigma_2 = self.encoder()
        self.z_graph = self._graph()
        self.position = self.position_embedding()
        self.attn = self.attention()
        # self.z_graph = self.z_e
        self.x_hat_e = self.decoder_ze()
        self.x_hat_q = self.decoder_zq()
        self.x_hat = self.decoder_z()
        self.loss = self.get_loss()
        self.train_op = self.optimize()

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
            h_3 = dense(self.attn, 2*self.z_dim, activation=tf.nn.relu, name="h_3")
            h_4 = dense(h_3, self.seq_len, name="h_4")
            x_hat = tf.expand_dims(h_4, 1, name="result")
        return x_hat

    def decoder_zq(self):
        with tf.variable_scope("decoder_zq", reuse=tf.AUTO_REUSE):
            h_3 = dense(self.z_graph, 2*self.z_dim, tf.nn.relu, name="h_3")
            h_4 = dense(h_3, self.seq_len, name="h_4")
            x_hat = tf.expand_dims(h_4, 1, name='result')
        return x_hat

    def decoder_z(self):
        with tf.variable_scope("decoder_z", reuse=tf.AUTO_REUSE):
            h_3 = dense(self.z_e, 2*self.z_dim, tf.nn.relu, name="h_3")
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
            self.init_prob = tf.get_variable("prob", [node_num, node_num],
                                             initializer=tf.orthogonal_initializer())
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
                loss_kl = - 0.5 * tf.reduce_sum(1-tf.square(self.mu)-self.sigma_2+tf.log(self.sigma_2), axis=-1)
                vae_loss = tf.reduce_mean(loss_rec_mse_z + loss_kl, name="vae_loss")
            with tf.name_scope("ze_loss"):
                loss_rec_mse_ze = tf.reduce_sum(tf.square(self.input_x-self.x_hat_e))
        loss_rec_mse = vae_loss + loss_rec_mse_zq + loss_rec_mse_ze
        tf.summary.scalar("loss_reconstruction", loss_rec_mse)
        return loss_rec_mse_ze

    def loss_commitment(self):
        with tf.variable_scope("loss_commit"):
            # loss_commit = tf.reduce_mean(tf.squared_difference(self.attn, self.z_graph))
            loss_commit1 = tf.reduce_mean(tf.squared_difference(self.z_e, self.z_graph))
            loss_commit2 = tf.reduce_mean(tf.squared_difference(self.z_e, self.attn))
            loss_commit = loss_commit1 + loss_commit2
        tf.summary.scalar("loss_commit", loss_commit)
        return loss_commit

    def loss_graph(self):
        with tf.variable_scope("loss_graph"):
            loss_graph = self.run_lstm_ze()
        tf.summary.scalar("loss_graph", loss_graph)
        return loss_graph

    def get_loss(self):
        with tf.variable_scope("total_loss"):
            with tf.name_scope("loss"):
                loss = self.loss_reconstruction() + self.loss_commitment()
                loss = tf.reduce_mean(loss, name='loss')
        tf.summary.scalar("loss", loss)
        return loss

    def optimize(self):
        with tf.variable_scope("optimize"):
            starter_learning_rate = 0.005
            learning_rate = tf.train.exponential_decay(starter_learning_rate, self.global_step,
                                                       50, 0.9, staircase=True)
            optimizer = tf.train.AdamOptimizer(learning_rate)
            grads, variables = zip(*optimizer.compute_gradients(self.loss))
            grads, global_norm = tf.clip_by_global_norm(grads, 5)
            train_op = optimizer.apply_gradients(zip(grads, variables), global_step=self.global_step)
        return train_op
