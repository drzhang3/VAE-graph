import tensorflow as tf
import tensorflow_probability as tfp
from functools import partial
from utils import wrap_params_net, wrap_params_net_srnn, softplus_std, rnn, RecurrentDistribution
from tensorflow_probability.python.distributions import LinearGaussianStateSpaceModel, MultivariateNormalDiag
from tensorflow.python.ops.linalg.linear_operator_identity import LinearOperatorIdentity


class OmniAnomaly:
    def __init__(self, config, window_length, x_dim, h_dim, z_dim, batch_size, use_planarNF=True):
        self.window_length = config.window_length
        self.x_dim = x_dim
        self.h_dim = config.rnn_num_hidden
        self.z_dim = config.z_dim
        self.batch_size = config.batch_size
        self.use_planarNF = config.use_planarNF
        self.input_x = tf.placeholder(shape=[None, self.window_length, self.x_dim],
                                      dtype=tf.float32, name="input_x")
        normal = tfp.distributions.MultivariateNormalDiag(loc=tf.zeros(self.latent_dim),
                                                              scale_diag=tf.ones(self.latent_dim))
        self.p_x_given_z = normal
        self.q_z_given_x = partial(RecurrentDistribution,
                                    mean_q_mlp=partial(tf.layers.dense, units=config.z_dim, name='z_mean', reuse=tf.AUTO_REUSE),
                                    std_q_mlp=partial(softplus_std, units=config.z_dim, epsilon=config.std_epsilon,
                                                      name='z_std'),
                                    z_dim=config.z_dim) if config.use_connected_z_q else normal
        self.p_z = LinearGaussianStateSpaceModel(
                        num_timesteps=config.window_length,
                        transition_matrix=LinearOperatorIdentity(config.z_dim),
                        transition_noise=MultivariateNormalDiag(
                            scale_diag=tf.ones([config.z_dim])),
                        observation_matrix=LinearOperatorIdentity(config.z_dim),
                        observation_noise=MultivariateNormalDiag(
                            scale_diag=tf.ones([config.z_dim])),
                        initial_state_prior=MultivariateNormalDiag(
                            scale_diag=tf.ones([config.z_dim]))
                    )

    def encoder(self, config):
        # h for q_z
        h_for_q_z = lambda x: rnn(x=x,
                                   window_length=config.window_length,
                                   rnn_num_hidden=config.rnn_num_hidden,
                                   hidden_dense=2,
                                   dense_dim=config.dense_dim,
                                   name='rnn_q_z')
        with tf.variable_scope('h_for_q_z'):
            z_params = h_for_q_z(self.input_x)
        with tf.variable_scope('q_z_given_x'):
            q_z_given_x = self.q_z_given_x(**z_params)
        z = tf.identity(q_z_given_x)

        self.z_e = q_z_given_x

    def decoder(self, config):
        z = tf.identity(self.p_z)
        h_for_dist = lambda x: rnn(x=x,
                                   window_length=config.window_length,
                                   rnn_num_hidden=config.rnn_num_hidden,
                                   hidden_dense=2,
                                   dense_dim=config.dense_dim,
                                   name='rnn_p_x')
        h = h_for_dist(z)
        with tf.variable_scope('x_mean', reuse=tf.AUTO_REUSE):
            mean = tf.keras.layers.Dense(config.x_dim)(h)
        with tf.variable_scope('x_std', reuse=tf.AUTO_REUSE):
            std = tf.nn.softplus(tf.keras.layers.Dense(config.x_dim)(h)) + config.std_epsilon
        with tf.variable_scope('p_x_given_z'):
            p_x_given_z = tfp.distributions.MultivariateNormalDiag(loc=mean,scale_diag=std)
        return p_x_given_z

    def loss(self):
        """
        kl_loss = tf.reduce_mean(self.q_z_given_x.kl_divergence(self.p_z))
        log_like_loss = -tf.reduce_mean(self.decoder.log_prob(self.input_x)

        :return:
        """
        pass


    def planarNF(self, x):
        # TODO : add planarNF for z
        return x

    def linear_gaussian_state_model(self, z_t_prev, z_t):
        pass






