from vae import VAE
from preprocess import data_mu_scaler, get_train_data
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error,median_absolute_error
tf.reset_default_graph()


def evaluation(x, y):
    r2 = r2_score(x, y)
    mse = mean_squared_error(x, y)
    print("The r2 is %f"%r2)
    print("The mse if %f"%mse)


def shuffle_data(x_train):
    indices = np.arange(x_train.shape[0])
    np.random.shuffle(indices)
    x_train_res = x_train[indices]
    return x_train_res


def train(model, x_train, epochs, batch_size):
    # x_train = shuffle_data(x_train)
    batch_num = x_train.shape[0] // batch_size
    print('start training...')
    loss_list = []
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        saver = tf.train.Saver()
        writer = tf.summary.FileWriter('.\logs', sess.graph)
        for epoch in range(epochs):
            temp_loss = []
            for j in range(batch_num):
                _, loss_ = sess.run([model.train_op, model.loss],
                                    feed_dict={model.input_x: x_train[j * batch_size:(j + 1) * batch_size]})

                # print('Epoch: ', epoch + 1, '| Batch: ', j + 1, '| Loss: ', loss_)
                temp_loss.append(loss_)
            loss_list.append(np.mean(temp_loss))
        saver.save(sess, 'models\ckp')
        summaries = tf.summary.merge_all()
    return loss_list


def test(x_test):
    batch_num = x_test.shape[0]
    print('start predicting...')
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph('models\ckp.meta')
        saver.restore(sess, tf.train.latest_checkpoint('models'))
        graph = tf.get_default_graph()
        input_x = graph.get_tensor_by_name("input_x:0")
        x_hat_e = graph.get_tensor_by_name("decoder_ze/result:0")

        for i in range(batch_num):
            x_hat_e = sess.run(x_hat_e, feed_dict={input_x: x_test[i:i + 1]})
            # print(np.shape(x_test[0:1][0,0,:]))
            evaluation(x_hat_e[0, 0, :], x_test[0:1][0, 0, :])
            # plt.plot(x_hat_e[0, 0, :])
            # plt.plot(x_test[0:1][0, 0, :])
            # plt.show()
            if i == 0:
                break


seq_len = 128
step = 8
z_dim = 16
epochs = 400
batch_size = 32
decay_factor = 0.9
# data1 = [np.sin(np.pi*i*0.03125) for i in range(5000)]
data2 = [np.sin(np.pi * i * 0.04)+0.1*np.random.random() for i in range(10000)]
data = data2
data = list(pd.read_csv("latency_15_min.csv").Latency)[1:-1]
data = [(i-np.min(data))/(np.max(data)-np.min(data)) for i in data]
x_train, y_train = get_train_data(data, seq_len, step)
x_train = np.reshape(x_train, [x_train.shape[0], 1, x_train.shape[1]])
print("ok")
# data = [(i-np.mean(data)/np.std(data)) for i in data]
print(x_train.shape)
model = VAE(z_dim=z_dim, seq_len=seq_len, input_dim=1)
loss = train(model, x_train, epochs, batch_size)
plt.plot(loss)
plt.show()
test(x_train)
