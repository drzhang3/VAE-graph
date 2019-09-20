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


def init_adj_prob(n):
    mat = np.random.random([n, n])
    for i in range(n):
        mat[i][i] = 1
    return mat

class VAE():
    def __init__(self, batch_size, z_dim, seq_len, input_dim, hidden_dim):
        self.batch_size = batch_size
        self.z_dim = z_dim
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.input_x = tf.placeholder(shape=[self.batch_size, input_dim, seq_len], dtype=tf.float32, name="input_x")
        self.global_step = tf.Variable(0, trainable=False, name="global_step")
        self.decay_steps = 500
        # self.batch_size = tf.shape(self.input_x)[0]
        self.som_dim = [8,8]
        self.embeddings = self.embeddings()
        self.transition_probabilities = self.transition_probabilities()
        self.z_e, self.mu, self.sigma_2 = self.encoder()
        self.z_graph = self._graph()
        # self.z_graph = self.z_e
        self.z_e_x, self.z_e_y = self.get_z_e(self.z_graph)
        self.z_e_old = self.z_e_old()
        self.k = self.k()
        self.z_q = self.z_q()
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
            # return z_e

    def z_e_old(self):
        """Aggregates the encodings of the respective previous time steps."""
        z_e_old = tf.concat([self.z_e[0:1], self.z_e[:-1]], axis=0)
        return z_e_old

    def embeddings(self):
        embeddings = tf.get_variable("embeddings", self.som_dim+[self.z_dim],
                                     initializer=tf.truncated_normal_initializer(stddev=0.05))
        tf.summary.tensor_summary("embeddings", embeddings)
        return embeddings

    def transition_probabilities(self):
        """Creates tensor for the transition probabilities."""
        with tf.variable_scope("probabilities"):
            probabilities_raw = tf.Variable(tf.zeros(self.som_dim+self.som_dim), name="probabilities_raw")
            probabilities_positive = tf.exp(probabilities_raw)
            probabilities_summed = tf.reduce_sum(probabilities_positive, axis=[-1,-2], keepdims=True)
            probabilities_normalized = probabilities_positive / probabilities_summed
            return probabilities_normalized         # (8,8,8,8)

    def z_dist_flat(self):
        z_dist = tf.squared_difference(tf.expand_dims(tf.expand_dims(self.z_e, 1), 1), tf.expand_dims(self.embeddings, 0))
        # (batch_size, 8, 8, z_dim)
        z_dist_red = tf.reduce_sum(z_dist, axis=-1)    # (batch_size, 8, 8)
        z_dist_flat = tf.reshape(z_dist_red, [self.batch_size, -1])     # (batch_size, 64)
        return z_dist_flat

    def z_q(self):
        k_1 = self.k // self.som_dim[1]
        k_2 = self.k % self.som_dim[1]
        k_stacked = tf.stack([k_1, k_2], axis=1)
        z_q = tf.gather_nd(self.embeddings, k_stacked)   # (batch_size, z_dim)
        return z_q

    def z_q_neighbors(self):
        """Aggregates the respective neighbors in the SOM for every embedding in z_q."""
        k_1 = self.k // self.som_dim[1]
        k_2 = self.k % self.som_dim[1]
        # k_stacked = tf.stack([k_1, k_2], axis=1)

        k1_not_top = tf.less(k_1, tf.constant(self.som_dim[0]-1, dtype=tf.int64))
        k1_not_bottom = tf.greater(k_1, tf.constant(0, dtype=tf.int64))
        k2_not_right = tf.less(k_2, tf.constant(self.som_dim[1]-1, dtype=tf.int64))
        k2_not_left = tf.greater(k_2, tf.constant(0, dtype=tf.int64))

        k1_up = tf.where(k1_not_top, tf.add(k_1, 1), k_1)
        k1_down = tf.where(k1_not_bottom, tf.subtract(k_1, 1), k_1)
        k2_right = tf.where(k2_not_right, tf.add(k_2, 1), k_2)
        k2_left = tf.where(k2_not_left, tf.subtract(k_2, 1), k_2)

        z_q_up = tf.where(k1_not_top, tf.gather_nd(self.embeddings, tf.stack([k1_up, k_2], axis=1)),
                          tf.zeros([self.batch_size, self.z_dim]))
        z_q_down = tf.where(k1_not_bottom, tf.gather_nd(self.embeddings, tf.stack([k1_down, k_2], axis=1)),
                          tf.zeros([self.batch_size, self.z_dim]))
        z_q_right = tf.where(k2_not_right, tf.gather_nd(self.embeddings, tf.stack([k_1, k2_right], axis=1)),
                          tf.zeros([self.batch_size, self.z_dim]))
        z_q_left = tf.where(k2_not_left, tf.gather_nd(self.embeddings, tf.stack([k_1, k2_left], axis=1)),
                          tf.zeros([self.batch_size, self.z_dim]))

        z_q_neighbors = tf.stack([self.z_q, z_q_up, z_q_down, z_q_right, z_q_left], axis=1)

        return z_q_neighbors

    def k(self):
        k = tf.argmax(-self.z_dist_flat(), axis=-1, name="k")     # (batch_size,)
        tf.summary.histogram("clusters", k)
        return k

    def decoder_ze(self):
        with tf.variable_scope("decoder_ze", reuse=tf.AUTO_REUSE):
            h_3 = dense(self.z_graph, 2*self.z_dim, activation=tf.nn.relu)
            x_hat = dense(h_3, self.seq_len)
            x_hat = tf.expand_dims(x_hat, 1, name="result")
            return x_hat

    def decoder_zq(self):
        with tf.variable_scope("decoder_zq", reuse=tf.AUTO_REUSE):
            h_3 = dense(self.z_q, 2*self.z_dim, tf.nn.relu)
            x_hat = dense(h_3, self.seq_len)
            x_hat = tf.expand_dims(x_hat, 1, name='result')
            return x_hat

    def get_z_e(self, value, seq_len=3, step=1):
        z_e_x = []
        z_e_y = []
        for i in range(0, value.shape[0] - seq_len, step):
            x = value[i:i + seq_len + 1]
            y = value[i:i + seq_len + 1]
            z_e_x.append(x[:-1])
            z_e_y.append(y[-1])
        return z_e_x, z_e_y

    def run_lstm_ze(self):
        loss = []
        for i in range(5):
            outputs = self.lstm_vae(self.z_e_x[i])
            loss.append(z_loss(outputs, self.z_e_y[i]))
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
        #with tf.variable_scope("gcn", reuse=tf.AUTO_REUSE):

        max_iteration = 2
        node_num = self.batch_size    # node_num = batch_size
        node_dim = self.z_dim
        node_state = self.z_e       # (node_num, node_dim)
        tf.add_to_collection("z_e", node_state)
        # init_prob = tf.get_variable("prob", [node_num, node_num],
        #                             initializer=tf.truncated_normal_initializer(stddev=0.05))
        init_prob = tf.get_variable("prob", [node_num, node_num],
                                    initializer=tf.orthogonal_initializer())

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
        loss_rec_mse_zq = tf.losses.mean_squared_error(self.input_x, self.x_hat_q)
        loss_rec_mse_ze = tf.losses.mean_squared_error(self.input_x, self.x_hat_e)
        loss_rec_mse = loss_rec_mse_ze
        tf.summary.scalar("loss_reconstruction", loss_rec_mse)
        return loss_rec_mse_ze

    def loss_commitment(self):
        loss_commit = tf.reduce_mean(tf.squared_difference(self.z_e, self.z_q))
        tf.summary.scalar("loss_commit", loss_commit)
        return loss_commit

    def loss_som(self):
        """Computes the SOM loss."""
        loss_som = tf.reduce_mean(tf.squared_difference(tf.expand_dims(tf.stop_gradient(self.z_e), axis=1), self.z_q_neighbors()))
        tf.summary.scalar("loss_som", loss_som)
        return loss_som

    def loss_probabilities(self):
        """Computes the negative log likelihood loss for the transition probabilities."""
        k_1 = self.k // self.som_dim[1]
        k_2 = self.k % self.som_dim[1]
        k_1_old = tf.concat([k_1[0:1], k_1[:-1]], axis=0)
        k_2_old = tf.concat([k_2[0:1], k_2[:-1]], axis=0)
        k_stacked = tf.stack([k_1_old, k_2_old, k_1, k_2], axis=1)         # (batch_size, 4)
        transitions_all = tf.gather_nd(self.transition_probabilities, k_stacked)    # (batch_size, )
        loss_probabilities = - tf.reduce_mean(tf.log(transitions_all))
        return loss_probabilities

    def loss_z_prob(self):
        """Computes the smoothness loss for the transitions given their probabilities."""
        k_1 = self.k // self.som_dim[1]
        k_2 = self.k % self.som_dim[1]
        k_1_old = tf.concat([k_1[0:1], k_1[:-1]], axis=0)
        k_2_old = tf.concat([k_2[0:1], k_2[:-1]], axis=0)
        k_stacked_old = tf.stack([k_1_old, k_2_old], axis=1)
        out_probabilities_old = tf.gather_nd(self.transition_probabilities, k_stacked_old)
        out_probabilities_flat = tf.reshape(out_probabilities_old, [self.batch_size, -1])
        weighted_z_dist_prob = tf.multiply(self.z_dist_flat(), out_probabilities_flat)
        loss_z_prob = tf.reduce_mean(weighted_z_dist_prob)
        return loss_z_prob

    def get_loss(self):
        with tf.variable_scope("loss"):
            # loss_rec = tf.losses.mean_squared_error(self.input_x, self.x_hat)
            loss_kl = - 0.5 * tf.reduce_sum(1-tf.square(self.mu)-self.sigma_2+tf.log(self.sigma_2))
            # loss = self.loss_reconstruction() + loss_kl
            # loss = self.loss_reconstruction()+self.loss_commitment()+self.loss_som()+\
            #      self.loss_probabilities() + self.loss_z_prob()
            loss = self.loss_reconstruction()+self.run_lstm_ze()
            return loss

    def optimize(self):
        with tf.variable_scope("optimize"):
            starter_learning_rate = 0.005
            learning_rate = tf.train.exponential_decay(starter_learning_rate, self.global_step,
                                                       50, 0.9, staircase=True)
            optimizer = tf.train.AdamOptimizer(learning_rate)
            # optimizer = tf.train.AdadeltaOptimizer()
            return optimizer

