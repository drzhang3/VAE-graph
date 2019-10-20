import tensorflow as tf
import tensorflow_probability as tfp
from functools import partial
from utils import wrap_params_net, wrap_params_net_srnn, softplus_std, rnn, RecurrentDistribution, lazy_scope
from tensorflow_probability.python.distributions import LinearGaussianStateSpaceModel, MultivariateNormalDiag
from tensorflow.python.ops.linalg.linear_operator_identity import LinearOperatorIdentity


class OmniAnomaly:
    def __init__(self, config):
        self.config = config
        self.window_length = config.window_length
        self.x_dim = config.x_dim
        self.h_dim = config.rnn_num_hidden
        self.z_dim = config.z_dim
        self.batch_size = config.batch_size
        self.use_planarNF = config.posterior_flow_type
        self.input_x = tf.placeholder(shape=[None, self.window_length, self.x_dim],
                                      dtype=tf.float32, name="input_x")
        normal = tfp.distributions.MultivariateNormalDiag(loc=tf.zeros(config.z_dim),
                                                              scale_diag=tf.ones(config.z_dim))
        self.p_x_given_z = normal
        self.q_z_given_x = partial(RecurrentDistribution,
                            mean_q_mlp=partial(tf.layers.dense, units=config.z_dim, name='z_mean', reuse=tf.AUTO_REUSE),
                            std_q_mlp=partial(softplus_std, units=config.z_dim, epsilon=config.std_epsilon, name='z_std'),
                            z_dim=config.z_dim, window_length=config.window_length) if config.use_connected_z_q else normal
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

    @lazy_scope
    def encoder(self):
        config = self.config       # h for q_z
        h_for_q_z = lambda x: rnn(x=x,
                                   window_length=config.window_length,
                                   rnn_num_hidden=config.rnn_num_hidden,
                                   hidden_dense=2,
                                   dense_dim=config.dense_dim,
                                   name='rnn_q_z')
        with tf.variable_scope('h_for_q_z'):
            z_params = h_for_q_z(self.input_x)
            # (batch_size, window_length, dense_dim)
        with tf.variable_scope('q_z_given_x'):
            q_z_given_x = self.q_z_given_x(input_q=z_params)
        # z = tf.identity(q_z_given_x)
        return q_z_given_x

    @lazy_scope
    def decoder(self):
        config = self.config
        z_e = self.encoder.sample()
        h_for_dist = lambda x: rnn(x=x,
                                   window_length=config.window_length,
                                   rnn_num_hidden=config.rnn_num_hidden,
                                   hidden_dense=2,
                                   dense_dim=config.dense_dim,
                                   name='rnn_p_x')
        h = h_for_dist(z_e)
        with tf.variable_scope('x_mean', reuse=tf.AUTO_REUSE):
            mean = tf.keras.layers.Dense(config.x_dim)(h)
        with tf.variable_scope('x_std', reuse=tf.AUTO_REUSE):
            std = tf.nn.softplus(tf.keras.layers.Dense(config.x_dim)(h)) + config.std_epsilon
        with tf.variable_scope('p_x_given_z'):
            p_x_given_z = tfp.distributions.MultivariateNormalDiag(loc=mean,scale_diag=std)
        return p_x_given_z


    @lazy_scope
    def loss(self):
        loss_recon = tf.reduce_sum(self.decoder.log_prob(self.input_x), axis=-1)
        kl_loss = tf.reduce_mean(self.q_z_given_x.kl_divergence(self.p_z),axis=-1)
        # loss = tf.recude_mean(loss_recon, kl_loss)
        return loss_recon







