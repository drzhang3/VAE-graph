from vae import VAE
from preprocess import data_mu_scaler, get_train_data, binarization, drawDAG, sigmoid, mask_data
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, median_absolute_error
tf.reset_default_graph()


def evaluation(x, y):
    r2 = r2_score(x, y)
    mse = mean_squared_error(x, y)
    print("The r2 is %f" % r2)
    print("The mse if %f" % mse)


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
        summaries = tf.summary.merge_all()
        writer = tf.summary.FileWriter('.\logs', sess.graph)
        try:
            print('VAE pre_training stage1 ...')
            for epoch in range(30):
                temp_loss = []
                for j in range(batch_num):
                    _, loss_, train_summaries = sess.run([model.train_vae, model.loss_vae, summaries],
                                    feed_dict={model.input_x: x_train[j * batch_size:(j + 1) * batch_size]})
                    temp_loss.append(loss_)
                    writer.add_summary(train_summaries, tf.train.global_step(sess, model.global_step))
                if epoch % 10 == 0:
                    print('Epoch: ', epoch + 1, '| Loss: ', np.mean(temp_loss))
                loss_list.append(np.mean(temp_loss))

            print('VAE pre_training stage2 ...')
            for epoch in range(30):
                temp_loss = []
                for j in range(batch_num):
                    _, loss_, train_summaries = sess.run([model.train_vae, model.loss_vae, summaries],
                                    feed_dict={model.input_x: x_train[j * batch_size:(j + 1) * batch_size]})
                    temp_loss.append(loss_)
                    writer.add_summary(train_summaries, tf.train.global_step(sess, model.global_step))
                if epoch % 10 == 0:
                    print('Epoch: ', epoch + 1, '| Loss: ', np.mean(temp_loss))
                loss_list.append(np.mean(temp_loss))

            print('training ...')
            for epoch in range(epochs):
                temp_loss = []
                for j in range(batch_num):
                    _, loss_, train_summaries = sess.run([model.train_op, model.loss, summaries],
                                    feed_dict={model.input_x: x_train[j * batch_size:(j + 1) * batch_size]})
                    temp_loss.append(loss_)
                    writer.add_summary(train_summaries, tf.train.global_step(sess, model.global_step))
                if epoch % 10 == 0:
                    print('Epoch: ', epoch + 1, '| Loss: ', np.mean(temp_loss))
                loss_list.append(np.mean(temp_loss))

        except KeyboardInterrupt:
            pass
        finally:
            saver.save(sess, 'models\ckp')

    return loss_list


def test(x_test):
    batch_num = x_test.shape[0]
    print('start predicting...')
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph('models\ckp.meta')
        saver.restore(sess, tf.train.latest_checkpoint('models'))
        graph = tf.get_default_graph()
        input_x = graph.get_tensor_by_name("input_x:0")
        # x_hat_e = graph.get_tensor_by_name("decoder_ze/result:0")
        x_hat_ = graph.get_tensor_by_name("x_hat/result:0")
        k_ = graph.get_tensor_by_name("k/index:0")
        # prob_ = graph.get_tensor_by_name("gcn/prob:0")
        # similarity_ = tf.get_collection("similarity")[0]
        # node_state_ = tf.get_collection("node_state")[0]
        # multi_head_ = tf.get_collection("multi-head")[0]
        # position_ = tf.get_collection("position")[0]
        # z_e_ = tf.get_collection("z_e")[0]
        # similarity = graph.get_tensor_by_name("similarity:0")
        print("variable is ready")

        # for i in range(batch_num//batch_size):
        #     node_state = sess.run(node_state_, feed_dict={input_x: x_test[i * batch_size:(i + 1) * batch_size]})
        #     z_e = sess.run(z_e_, feed_dict={input_x: x_test[i * batch_size:(i + 1) * batch_size]})
        #     multi_head = sess.run(multi_head_, feed_dict={input_x: x_test[i * batch_size:(i + 1) * batch_size]})
        #     position = sess.run(position_, feed_dict={input_x: x_test[i * batch_size:(i + 1) * batch_size]})
        #
        #     # print(z_e)
        #     plt.subplot(2, 2, 1)
        #     plt.title("raw_state:z_e")
        #     plt.imshow(z_e, cmap="RdYlBu", aspect='auto', vmin=0, vmax=1, origin='low')
        #     plt.colorbar()
        #     plt.subplot(2, 2, 2)
        #     plt.title("GCN state")
        #     plt.imshow(node_state, cmap="RdYlBu", aspect='auto', vmin=0, vmax=1, origin='low')
        #     plt.colorbar()
        #     plt.subplot(2, 2, 3)
        #     plt.title("attention state")
        #     plt.imshow(multi_head, cmap="RdYlBu", aspect='auto', vmin=0, vmax=1, origin='low')
        #     plt.colorbar()
        #     plt.subplot(2, 2, 4)
        #     plt.title("position state")
        #     plt.imshow(position, cmap="RdYlBu", aspect='auto', vmin=0, vmax=1, origin='low')
        #     plt.colorbar()
        #     plt.show()
        #     if i == 0:
        #         break

        # similarity_list = []
        # for i in range(batch_num//batch_size):
        #     similarity = sess.run(similarity_, feed_dict={input_x: x_test[i * batch_size:(i + 1) * batch_size]})
        #     similarity_list.append(similarity)

        # for i in range(4):
        #     # print(similarity_list[0]==similarity_list[i])
        #     plt.subplot(2, 2, i+1)
        #     plt.imshow(similarity_list[i], cmap="RdYlBu", vmin=-1, vmax=1, origin='low')
        #     plt.colorbar()
        # plt.show()

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
            x_hat_e = sess.run(x_hat_, feed_dict={input_x: x_test[i * batch_size:(i + 1) * batch_size]})
            # print(np.shape(x_test[0:1][0,0,:]))
            evaluation(x_hat_e[0, 0, :], x_test[0:1][0, 0, :])
            plt.plot(x_hat_e[0, 0, :], label='reconstruction')
            plt.plot(x_test[0:1][0, 0, :], label = "raw data")
            plt.legend(labels=['reconstruction', 'raw data'], loc='best')
            plt.show()
            if i == 0:
                break

        index = []
        for i in range(batch_num):
            k = sess.run(k_, feed_dict={input_x: x_test[i * batch_size:(i + 1) * batch_size]})
            # print(np.shape(x_test[0:1][0,0,:]))
            index.extend(k)
            plt.plot(index)
            plt.show()

        # probs = []
        # for i in range(batch_num//batch_size):
        #     prob = sess.run(prob_, feed_dict={input_x: x_test[i * batch_size:(i + 1) * batch_size]})
        #     probs.append(prob)
        #     # print(probs[i])
        #     plt.subplot(1, 3, 1)
        #     plt.imshow(probs[i], cmap="RdYlBu", vmin=-1, vmax=1, origin='low')
        #     plt.colorbar()
        #     plt.subplot(1, 3, 2)
        #     plt.imshow(binarization(probs[i]), cmap="RdYlBu", vmin=-1, vmax=1, origin='low')
        #     plt.colorbar()
        #     plt.subplot(1, 3, 3)
        #     plt.imshow(sigmoid(probs[i]), cmap="RdYlBu", vmin=-1, vmax=1, origin='low')
        #     plt.colorbar()
        #     plt.show()
        #     # print(probs[0]==probs[i])
        #     if i == 0:
        #         break

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


seq_len = 64
step = 4
z_dim = 18     # VAE hidden_state size
hidden_dim = 10     # LSTM cell state size
epochs = 1000
batch_size = 50
decay_factor = 0.9
alpha = 1
beta = 1
gamma = 1
eta = 1
kappa = 0
theta = 1
is_spike = False
data1 = [np.sin(np.pi*i*0.04) for i in range(5000)]
data2 = [np.sin(np.pi*i*0.02) for i in range(5000)]
# raw_data = [np.sin(np.pi * i * 0.04) for i in range(5000)]
raw_data = data1
# data1 = [np.sin(np.pi*i*0.04) for i in range(100)]
# data2 = [np.sin(np.pi*i*0.02) for i in range(100)]
# data = data1 + data2
# raw_data = []
# for i in range(50):
#     raw_data.extend(data)
# raw_data = list(pd.read_csv("latency_15_min.csv").Latency)[1:-1]
# raw_data = [(i-np.min(raw_data))/(np.max(raw_data)-np.min(raw_data)) for i in raw_data]
mask = mask_data(raw_data)
# mask = raw_data
x_train, y_train = get_train_data(mask, seq_len, step)
x_train = np.reshape(x_train, [x_train.shape[0], 1, x_train.shape[1]])
print("ok")
# data = [(i-np.mean(data)/np.std(data)) for i in data]
print(x_train.shape)
# model = VAE(batch_size=batch_size, z_dim=z_dim, seq_len=seq_len, input_dim=1, hidden_dim=hidden_dim,
#             alpha=alpha, beta=beta, gamma=gamma, eta=eta, kappa=kappa, theta=theta, is_spike=is_spike)
# loss = train(model, x_train, epochs, batch_size)
# plt.plot(loss)
# plt.savefig("./fig/loss.png")
# plt.show()
test(x_train)
