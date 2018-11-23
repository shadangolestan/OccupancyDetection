import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell
import numpy as np
import data_loader
from tqdm import tqdm

hm_epochs = 1000
batch_size = 6 * 24
rnn_size = 16
i = 4

SET = 1
DATASET = ["ml", "pcl"]
TRAINING_RATIO = 0.8

# import some data to play with
if SET == 0:
    data_library = data_loader.load_data()
elif SET == 1:
    data_library = data_loader.load_pcl_data()
else:
    data_library = None
x_total = data_library.data
y_total = data_library.label

chunk_size = x_total.shape[1]
n_chunks = 1
n_classes = 2

num_lines = int(x_total.shape[0] * (1 - TRAINING_RATIO))
start_pos = num_lines * i
end_pos = min(start_pos + num_lines, x_total.shape[0])

x_train = np.concatenate((x_total[:start_pos], x_total[end_pos:]), axis=0)
one_col = np.concatenate((y_total[:start_pos], y_total[end_pos:]), axis=0)
y_train = np.zeros((one_col.shape[0], 2))
y_train[np.arange(one_col.shape[0]), one_col] = 1

x_test = x_total[start_pos:end_pos]
one_col = y_total[start_pos:end_pos]
y_test = np.zeros((one_col.shape[0], 2))
y_test[np.arange(one_col.shape[0]), one_col] = 1

input_nodes = x_train.shape[1]
x = tf.placeholder('float', [None, n_chunks, chunk_size])
y = tf.placeholder('float')


def recurrent_neural_network_model(x):
    layer = {'weights': tf.Variable(tf.random_normal([rnn_size, n_classes])),
             'biases': tf.Variable(tf.random_normal([n_classes]))}

    x = tf.transpose(x, [1, 0, 2])
    x = tf.reshape(x, [-1, chunk_size])
    x = tf.split(x, n_chunks, 0)

    lstm_cell = rnn_cell.BasicLSTMCell(rnn_size, state_is_tuple=True)
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

    output = tf.matmul(outputs[-1], layer['weights']) + layer['biases']

    return output


def train_neural_network(x):
    prediction = recurrent_neural_network_model(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction, labels=y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in tqdm(range(hm_epochs)):
            epoch_loss = 0
            for i in range(int(x_train.shape[0] / batch_size)):
                epoch_x = x_train[i * batch_size:(i + 1) * batch_size]
                epoch_y = y_train[i * batch_size:(i + 1) * batch_size]
                epoch_x = epoch_x.reshape((batch_size, n_chunks, chunk_size))

                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c

            # print('Epoch', epoch + 1, 'completed out of', hm_epochs, 'loss:', epoch_loss)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

        print('Accuracy:', accuracy.eval({x: x_test.reshape((-1, n_chunks, chunk_size)), y: y_test}))
        # sess.run([prediction, y], feed_dict={x: x_test, y: song_test_y})


train_neural_network(x)

scores = [0.90882975, 0.91218156, 0.90844667, 0.9051752, 0.88412184]
