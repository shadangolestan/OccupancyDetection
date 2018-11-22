import data_loader
import numpy as np

TRAINING_RATIO = 0.8
EPOCH = 500
HIDDEN_NODE = 10

def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1.0 - x)


class NeuralNetwork:
    def __init__(self, x, y):
        self.input = x
        self.weights1 = np.random.rand(self.input.shape[1], HIDDEN_NODE)
        self.weights2 = np.random.rand(HIDDEN_NODE, 1)
        self.y = y
        self.output = np.zeros(self.y.shape)
        self.layer1 = None

    def feedforward(self):
        self.layer1 = sigmoid(np.dot(self.input, self.weights1))
        self.output = sigmoid(np.dot(self.layer1, self.weights2))

    def backprop(self):
        # application of the chain rule to find derivative of the loss function with respect to weights2 and weights1
        d_weights2 = np.dot(self.layer1.T, (2 * (self.y - self.output) * sigmoid_derivative(self.output)))
        d_weights1 = np.dot(self.input.T, (np.dot(2 * (self.y - self.output) * sigmoid_derivative(self.output),
                                                  self.weights2.T) * sigmoid_derivative(self.layer1)))

        # update the weights with the derivative (slope) of the loss function
        self.weights1 += d_weights1
        self.weights2 += d_weights2

    def compute(self, x_test):
        layer1 = sigmoid(np.dot(x_test, self.weights1))
        output = sigmoid(np.dot(layer1, self.weights2))
        return output


# import some data to play with
data_library = data_loader.load_data()
x = data_library.data
y = data_library.label

num_lines = int(x.shape[0] * (1 - TRAINING_RATIO))

scores = []

for i in range(int(1 // (1 - TRAINING_RATIO))):
    start_pos = num_lines * i
    end_pos = min(start_pos + num_lines, x.shape[0])

    x_train = np.concatenate((x[:start_pos], x[end_pos:]), axis=0)
    y_train = np.reshape(np.concatenate((y[:start_pos], y[end_pos:]), axis=0), (-1, 1))

    x_test = x[start_pos:end_pos]
    y_test = np.reshape(y[start_pos:end_pos], (-1, 1))

    nn = NeuralNetwork(x_train, y_train)

    for _ in range(EPOCH):
        nn.feedforward()
        nn.backprop()

    y_result = nn.compute(x_test)

    accuracy = sum(y_test == y_result) / y_test.shape[0]

    print("Hit accuracy:", accuracy)

    scores.append(accuracy)

print(scores)
score = np.asarray(scores, dtype=float)

print("Average accuracy:", np.mean(scores))
print("STD. DEV.:", np.std(scores))