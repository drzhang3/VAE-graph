import tensorflow as tf
from tensorflow.python.layers.core import dense


class VAE():
    def __init__(self, z_dim, seq_len, input_dim):
        self.z_dim = z_dim
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.hidden_size = z_dim
        self.input_x = tf.placeholder(shape=[None, input_dim, seq_len], dtype=tf.float32, name="input_x")
        self.global_step = tf.Variable(0, trainable=False, name="global_step")
        self.decay_steps = 500
        self.batch_size = tf.shape(self.input_x)[0]
        self.som_dim = [8,8]
        self.z_e, self.mu, self.sigma_2 = self.encoder()
        self.x_hat_e = self.decoder_ze()
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

    def decoder_ze(self):
        with tf.variable_scope("decoder_ze", reuse=tf.AUTO_REUSE):
            h_3 = dense(self.z_e, 2*self.z_dim, activation=tf.nn.relu)
            x_hat = dense(h_3, self.seq_len)
            x_hat = tf.expand_dims(x_hat, 1, name="result")
            return x_hat

    def lstm_vae(self):
        cell = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_size)
        input_lstm = tf.expand_dims(self.z_e[:-1], 0, name="input_lstm")
        # cell = tf.contrib.rnn.AttentionCellWrapper(cell, attn_length = self.attention_len)
        outputs, _ = tf.nn.dynamic_rnn(cell, input_lstm, dtype=tf.float32)    # (batch_size, seq_len, z_dim)
        outputs = tf.transpose(outputs, [1, 0, 2])[-1]     # (1 , z_dim)
        return outputs

    def z_loss(self):
        return tf.losses.mean_squared_error(self.z_e[-1:], self.lstm_vae())

    def get_loss(self):
        with tf.variable_scope("loss"):
            loss_rec = tf.losses.mean_squared_error(self.input_x, self.x_hat_e)
            loss_kl = - 0.5 * tf.reduce_sum(1-tf.square(self.mu)-self.sigma_2+tf.log(self.sigma_2))
            loss = loss_rec + self.z_loss()
            return loss

    def optimize(self):
        with tf.variable_scope("optimize"):
            starter_learning_rate = 0.0005
            learning_rate = tf.train.exponential_decay(starter_learning_rate, self.global_step,
                                                200, 0.9, staircase=True)
            optimizer = tf.train.AdamOptimizer(learning_rate)
            return optimizer

