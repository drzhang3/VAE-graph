import tensorflow as tf
import tensorflow_probability as tfp


class StochasticRnn:
    def __init__(self, step, x_dim, h_dim, z_dim, batch_size, use_planarNF=True):
        self.time_step = step
        self.x_dim = x_dim
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.batch_size = batch_size
        self.use_planarNF = use_planarNF
        self.input_x = tf.placeholder(shape=[None, self.time_step, self.x_dim], dtype=tf.float32, name="input_x")

    def omni_net(self):
        cell1 = tf.keras.layers.GRU(self.h_dim)
        cell2 = tf.keras.layers.GRU(self.h_dim)
        init_z = tf.get_variable("z", [self.batch_size, self.z_dim],
                                 initializer=tf.truncated_normal_initializer(stddev=0.05))
        init_state1 = cell1.get_initial_state(self.input_x)
        init_state2 = cell2.get_initial_state(init_z)
        state1 = init_state1
        state2 = init_state2
        z = init_z
        for t in range(self.time_step):
            input_x = self.input_x[:,t,:]
            lstm_output1, state_h1, state_c1 = cell1(input_x, initial_state=state1)
            state1 = tf.identity([state_h1, state_c1])
            new_input = tf.concat(z, state1)
            prev_z = z
            z = self.convert(new_input, self.z_dim)
            z = self.planarNF(z) if self.use_planarNF else z
            self.linear_gaussian_state_model(z, prev_z)
            lstm_output2, state_h2, state_c2 = cell1(z, initial_state=state2)
            state2 = tf.identity([state_h2, state_c2])
            x = self.convert(lstm_output2, self.x_dim)

    def planarNF(self, x):
        # TODO : add planarNF for z
        return x

    def linear_gaussian_state_model(self, z_t_prev, z_t):
        pass

    def convert(self, x, dim):
        h_1 = tf.keras.layers.Dense(2 * dim, activation=tf.nn.leaky_relu)(x)
        # h_1 = Dropout(rate=self.dropout)(h_1)
        # h_1 = BatchNormalization()(h_1)
        h_2 = tf.keras.layers.Dense(dim, activation=tf.nn.leaky_relu)(h_1)
        # h_2 = Dropout(rate=self.dropout)(h_2)
        # h_2 = BatchNormalization()(h_2)
        with tf.name_scope("mu"):
            mu = tf.keras.layers.Dense(dim)(h_2)
            mu = tf.reshape(mu, shape=[-1, dim])
        with tf.name_scope("sigma"):
            log_sigma_2 = tf.keras.layers.Dense(dim)(h_2)
            sigma_2 = tf.exp(log_sigma_2)
            sigma_2 = tf.reshape(sigma_2, shape=[-1, dim])
            sigma = tf.sqrt(sigma_2)
        epsilon = tf.random_normal(shape=tf.shape(sigma), name="epsilon")
        with tf.name_scope("z"):
            z = mu + sigma * epsilon
        return z

    def loss(self):
        pass






