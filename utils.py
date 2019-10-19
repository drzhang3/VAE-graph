import tensorflow as tf
import tensorflow_probability as tfp
import logging
import functools


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

# TODO : add necessary code
def wrap_params_net(inputs, h_for_dist, mean_layer, std_layer):
    with tf.variable_scope('hidden', reuse=tf.AUTO_REUSE):
        h = h_for_dist(inputs)
    return {
        'mean': mean_layer(h),
        'std': std_layer(h),
    }


def wrap_params_net_srnn(inputs, h_for_dist):
    with tf.variable_scope('hidden', reuse=tf.AUTO_REUSE):
        h = h_for_dist(inputs)
    return {
        'input_q': h
    }

def rnn(x,
        window_length,
        rnn_num_hidden,
        rnn_cell='GRU',
        hidden_dense=2,
        dense_dim=200,
        time_axis=1,
        name='rnn'):
    from tensorflow.contrib import rnn
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        if len(x.shape) == 4:
            x = tf.reduce_mean(x, axis=0)
        elif len(x.shape) != 3:
            logging.error("rnn input shape error")
        x = tf.unstack(x, window_length, time_axis)

        if rnn_cell == 'LSTM':
            # Define lstm cells with TensorFlow
            # Forward direction cell
            fw_cell = rnn.BasicLSTMCell(rnn_num_hidden,
                                        forget_bias=1.0)
        elif rnn_cell == "GRU":
            fw_cell = tf.nn.rnn_cell.GRUCell(rnn_num_hidden)
        elif rnn_cell == 'Basic':
            fw_cell = tf.nn.rnn_cell.BasicRNNCell(rnn_num_hidden)
        else:
            raise ValueError("rnn_cell must be LSTM or GRU")

        # Get lstm cell output

        try:
            outputs, _ = rnn.static_rnn(fw_cell, x, dtype=tf.float32)
        except Exception:  # Old TensorFlow version only returns outputs not states
            outputs = rnn.static_rnn(fw_cell, x, dtype=tf.float32)
        outputs = tf.stack(outputs, axis=time_axis)
        for i in range(hidden_dense):
            outputs = tf.layers.dense(outputs, dense_dim)
        return outputs
    # return size: (batch_size, window_length, rnn_num_hidden)


def softplus_std(inputs, units, epsilon, name):
    return tf.nn.softplus(tf.layers.dense(inputs, units, name=name, reuse=tf.AUTO_REUSE)) + epsilon


class RecurrentDistribution:
    """
    A multi-variable distribution integrated with recurrent structure.
    """
    def sample_step(self, a, t):
        z_previous, mu_q_previous, std_q_previous = a
        noise_n, input_q_n = t
        input_q_n = tf.broadcast_to(input_q_n,
                                    [tf.shape(z_previous)[0], tf.shape(input_q_n)[0], input_q_n.shape[1]])
        input_q = tf.concat([input_q_n, z_previous], axis=-1)
        mu_q = self.mean_q_mlp(input_q, reuse=tf.AUTO_REUSE)  # n_sample * batch_size * z_dim

        std_q = self.std_q_mlp(input_q)  # n_sample * batch_size * z_dim

        temp = tf.einsum('ik,ijk->ijk', noise_n, std_q)  # n_sample * batch_size * z_dim
        mu_q = tf.broadcast_to(mu_q, tf.shape(temp))
        std_q = tf.broadcast_to(std_q, tf.shape(temp))
        z_n = temp + mu_q

        return z_n, mu_q, std_q

    # @global_reuse
    def log_prob_step(self, _, t):

        given_n, input_q_n = t
        if len(given_n.shape) > 2:
            input_q_n = tf.broadcast_to(input_q_n,
                                        [tf.shape(given_n)[0], tf.shape(input_q_n)[0], input_q_n.shape[1]])
        input_q = tf.concat([given_n, input_q_n], axis=-1)
        mu_q = self.mean_q_mlp(input_q, reuse=tf.AUTO_REUSE)

        std_q = self.std_q_mlp(input_q)
        logstd_q = tf.log(std_q)
        precision = tf.exp(-2 * logstd_q)
        if self._check_numerics:
            precision = tf.check_numerics(precision, "precision")
        log_prob_n = - 0.9189385332046727 - logstd_q - 0.5 * precision * tf.square(tf.minimum(tf.abs(given_n - mu_q),
                                                                                              1e8))
        return log_prob_n

    def __init__(self, input_q, mean_q_mlp, std_q_mlp, z_dim, window_length, is_reparameterized=True,
                 check_numerics=True):
        self.normal = tfp.distributions.MultivariateNormalDiag(loc=tf.zeros([window_length, z_dim]),
                                                          scale_diag=tf.ones([window_length, z_dim]))
        self.std_q_mlp = std_q_mlp
        self.mean_q_mlp = mean_q_mlp
        self._check_numerics = check_numerics
        self.input_q = tf.transpose(input_q, [1, 0, 2])
        self._dtype = input_q.dtype
        self._is_reparameterized = is_reparameterized
        self._is_continuous = True
        self.z_dim = z_dim
        self.window_length = window_length
        self.time_first_shape = tf.convert_to_tensor([self.window_length, tf.shape(input_q)[0], self.z_dim])

    def sample(self, n_samples=1, name=None):
        with tf.name_scope(name=name, default_name='sample'):
            noise = self.normal.sample(n_samples)

            noise = tf.transpose(noise, [1, 0, 2])  # window_length * n_samples * z_dim
            noise = tf.truncated_normal(tf.shape(noise))

            time_indices_shape = tf.convert_to_tensor([n_samples, tf.shape(self.input_q)[1], self.z_dim])

            samples = tf.scan(fn=self.sample_step,
                              elems=(noise, self.input_q),
                              initializer=(tf.zeros(time_indices_shape),
                                           tf.zeros(time_indices_shape),
                                           tf.ones(time_indices_shape)),
                              back_prop=False
                              )[0]  # time_step * n_samples * batch_size * z_dim

            samples = tf.transpose(samples, [1, 2, 0, 3])  # n_samples * batch_size * time_step *  z_dim
            t = tf.reduce_mean(samples, axis=0)
            return t

    def log_prob(self, given, group_ndims=0, name=None):
        with tf.name_scope(name=name, default_name='log_prob'):
            if len(given.shape) > 3:
                time_indices_shape = tf.convert_to_tensor([tf.shape(given)[0], tf.shape(self.input_q)[1], self.z_dim])
                given = tf.transpose(given, [2, 0, 1, 3])
            else:
                time_indices_shape = tf.convert_to_tensor([tf.shape(self.input_q)[1], self.z_dim])
                given = tf.transpose(given, [1, 0, 2])
            log_prob = tf.scan(fn=self.log_prob_step,
                               elems=(given, self.input_q),
                               initializer=tf.zeros(time_indices_shape),
                               back_prop=False
                               )
            if len(given.shape) > 3:
                log_prob = tf.transpose(log_prob, [1, 2, 0, 3])
            else:
                log_prob = tf.transpose(log_prob, [1, 0, 2])

            if group_ndims == 1:
                log_prob = tf.reduce_sum(log_prob, axis=-1)
            return log_prob

    def prob(self, given, group_ndims=0, name=None):
        with tf.name_scope(name=name, default_name='prob'):
            log_prob = self.log_prob(given, group_ndims, name)
            return tf.exp(log_prob)
