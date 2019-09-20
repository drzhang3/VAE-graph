from vae import VAE
from preprocess import data_mu_scaler, get_train_data, binarization, drawDAG, sigmoid, mask_data
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
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        saver = tf.train.Saver()
        writer = tf.summary.FileWriter('.\logs', sess.graph)
        for epoch in range(epochs):
            temp_loss = []
            for j in range(batch_num):
                _, loss_ = sess.run([model.train_op, model.loss],
                                    feed_dict={model.input_x: x_train[j * batch_size:(j + 1) * batch_size]})
                temp_loss.append(loss_)
            print('Epoch: ', epoch + 1, '| Loss: ', np.mean(temp_loss))
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
        prob_ = graph.get_tensor_by_name("prob:0")
        node_state_ = tf.get_collection("node_state")[0]
        z_e_ = tf.get_collection("z_e")[0]
        # similarity = graph.get_tensor_by_name("similarity:0")
        print("variable is ready")

        for i in range(batch_num//batch_size):
            node_state = sess.run(node_state_, feed_dict={input_x: x_test[i * batch_size:(i + 1) * batch_size]})
            z_e = sess.run(z_e_, feed_dict={input_x: x_test[i * batch_size:(i + 1) * batch_size]})
            print(node_state)
            # print(z_e)
            plt.subplot(1, 2, 1)
            plt.title("state after GCN")
            plt.imshow(node_state, cmap="RdYlBu", vmin=0, vmax=1, origin='low')
            plt.colorbar()
            plt.subplot(1, 2, 2)
            plt.title("state before GCN")
            plt.imshow(z_e, cmap="RdYlBu", vmin=0, vmax=1, origin='low')
            plt.colorbar()
            plt.show()
            if i == 0:
                break

        # plt.ion()
        # for i in range(batch_num):
        #     plt.cla()
        #     embedding = sess.run(embeddings, feed_dict={input_x: x_test[i:i+1]})
        #     print(type(embedding))
        #     print(np.shape(embedding))
        #     temp = np.reshape(embedding, [8, 8])
        #     plt.imshow(temp, cmap="RdYlBu", vmin=0, vmax=1)
        #     # plt.colorbar()
        #     plt.pause(0.1)
        # plt.ioff()
        # plt.show()
        # if i == 0:
        #     break

        for i in range(batch_num):
            x_hat_e = sess.run(x_hat_e, feed_dict={input_x: x_test[i * batch_size:(i + 1) * batch_size]})
            # print(np.shape(x_test[0:1][0,0,:]))
            evaluation(x_hat_e[0, 0, :], x_test[0:1][0, 0, :])
            plt.plot(x_hat_e[0, 0, :], label='reconstruction')
            plt.plot(x_test[0:1][0, 0, :], label = "raw data")
            plt.legend(labels=['reconstruction', 'raw data'], loc='best')
            plt.show()
            if i == 0:
                break

        probs = []
        for i in range(batch_num//batch_size):
            prob = sess.run(prob_, feed_dict={input_x: x_test[i * batch_size:(i + 1) * batch_size]})
            probs.append(prob)
        #print(probs[0])
        plt.subplot(1, 3, 1)
        plt.imshow(probs[0], cmap="Greys", vmin=0, vmax=1, origin='low')
        plt.subplot(1, 3, 2)
        plt.imshow(binarization(probs[0]), cmap="Greys", vmin=0, vmax=1, origin='low')
        plt.subplot(1, 3, 3)
        plt.imshow(sigmoid(probs[0]), cmap="Greys", vmin=0, vmax=1, origin='low')
        plt.show()
        # drawDAG(binarization(probs[0]))
        # print("k is ok")
        # fig = plt.figure()
        # ims = []
        # for i in range(batch_num):
        #     k_temp = np.zeros([8, 8])
        #     k_temp[k_index[i] // 8, k_index[i] % 8] = 1
        #     ims.append([plt.imshow(k_temp, cmap="Greys", vmin=0, vmax=1)])
        # ani = animation.ArtistAnimation(fig, ims, interval=200, repeat_delay=1000)
        # print('ok')
        # plt.show()
        # ani.save("test.mp4", writer='imagemagick')


seq_len = 32
step = 2
z_dim = 4     # VAE hidden_state size
hidden_dim = 4     # LSTM cell state size
epochs = 20
batch_size = 8
decay_factor = 0.9
data1 = [np.sin(np.pi*i*0.04) for i in range(5000)]
data2 = [np.sin(np.pi*i*0.02) for i in range(5000)]
raw_data = [np.sin(np.pi * i * 0.04) for i in range(5000)]
# raw_data = data1 + data2
# raw_data = list(pd.read_csv("latency_15_min.csv").Latency)[1:-1]
# raw_data = [(i-np.min(raw_data))/(np.max(raw_data)-np.min(raw_data)) for i in raw_data]
# mask = mask_data(raw_data)
mask = raw_data
x_train, y_train = get_train_data(mask, seq_len, step)
x_train = np.reshape(x_train, [x_train.shape[0], 1, x_train.shape[1]])
print("ok")
# data = [(i-np.mean(data)/np.std(data)) for i in data]
print(x_train.shape)
model = VAE(z_dim=z_dim, seq_len=seq_len, input_dim=1, hidden_dim=hidden_dim)
loss = train(model, x_train, epochs, batch_size)
plt.plot(loss)
plt.savefig("./fig/loss.png")
plt.show()
test(x_train)
