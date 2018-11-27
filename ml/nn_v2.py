import tensorflow as tf
import numpy as np
import data_loader
from tqdm import tqdm

batch_size = 60 * 24
hm_epochs = 1000
i = 4
TRAINING_RATIO = 0.8
layer_levels = 1
layer_nodes = [75] * layer_levels
SET = 1
DATASET = ["ml", "pcl"]

# import some data to play with
if SET == 0:
    data_library = data_loader.load_data()
elif SET == 1:
    data_library = data_loader.load_pcl_data()
else:
    data_library = None
x_total = data_library.data
y_total = data_library.label

n_classes = np.max(y_total) + 1

num_lines = int(x_total.shape[0] * (1 - TRAINING_RATIO))
start_pos = num_lines * i
end_pos = min(start_pos + num_lines, x_total.shape[0])

x_train = np.concatenate((x_total[:start_pos], x_total[end_pos:]), axis=0)
one_col = np.concatenate((y_total[:start_pos], y_total[end_pos:]), axis=0)
y_train = np.zeros((one_col.shape[0], n_classes))
y_train[np.arange(one_col.shape[0]), one_col] = 1

x_test = x_total[start_pos:end_pos]
one_col = y_total[start_pos:end_pos]
y_test = np.zeros((one_col.shape[0], n_classes))
y_test[np.arange(one_col.shape[0]), one_col] = 1

input_nodes = x_train.shape[1]
x = tf.placeholder('float', [None, input_nodes])
y = tf.placeholder('float')


def neural_network_model(data):
    hidden_layer = []
    layer_result = []
    layer_nodes.append(input_nodes)

    for hidden_layer_idx in range(layer_levels):
        hidden_layer.append({'weights': tf.Variable(
            tf.zeros([layer_nodes[hidden_layer_idx - 1], layer_nodes[hidden_layer_idx]])),
            'biases': tf.Variable(tf.random_normal([layer_nodes[hidden_layer_idx]]))})

    output_layer = {'weights': tf.Variable(tf.random_normal([layer_nodes[-2], n_classes])),
                    'biases': tf.Variable(tf.random_normal([n_classes]))}

    layer_result.append(tf.add(tf.matmul(data, hidden_layer[0]['weights']), hidden_layer[0]['biases']))

    layer_result[0] = tf.nn.relu(layer_result[0])

    for hidden_layer_idx in range(1, layer_levels):
        layer_result.append(
            tf.add(tf.matmul(layer_result[hidden_layer_idx - 1], hidden_layer[hidden_layer_idx]['weights']),
                   hidden_layer[hidden_layer_idx]['biases']))

        layer_result[hidden_layer_idx] = tf.nn.relu(layer_result[hidden_layer_idx])

    output = tf.matmul(layer_result[-1], output_layer['weights']) + output_layer['biases']

    return output


def train_neural_network(x):
    prediction = neural_network_model(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction, labels=y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in tqdm(range(hm_epochs)):
            epoch_loss = 0
            for i in range(int(x_train.shape[0] / batch_size)):
                epoch_x = x_train[i * batch_size:(i + 1) * batch_size]
                epoch_y = y_train[i * batch_size:(i + 1) * batch_size]
                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c

            # print('Epoch', epoch + 1, 'completed out of', hm_epochs, 'loss:', epoch_loss)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

        print('Accuracy:', accuracy.eval({x: x_test, y: y_test}))
        # sess.run([prediction, y], feed_dict={x: x_test, y: song_test_y})


train_neural_network(x)

scores = [0.91218156, 0.9100747, 0.90930855, 0.8851752, 0.88412184]