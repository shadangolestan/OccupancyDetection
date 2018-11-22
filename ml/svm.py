import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.base import clone
import data_loader

EXACT_FIT_RATIO = 'auto'
PENALTY_ERROR = 1
KERNEL = 'linear'
TRAINING_RATIO = 0.8

# import some data to play with
data_library = data_loader.load_data()
x = data_library.data
y = data_library.label
svc = svm.SVC(kernel=KERNEL, C=PENALTY_ERROR, gamma=EXACT_FIT_RATIO)

num_lines = int(x.shape[0] * (1 - TRAINING_RATIO))

scores = []

for i in range(int(1 // (1 - TRAINING_RATIO))):
    start_pos = num_lines * i
    end_pos = min(start_pos + num_lines, x.shape[0])

    x_train = np.concatenate((x[:start_pos], x[end_pos:]), axis=0)
    y_train = np.concatenate((y[:start_pos], y[end_pos:]), axis=0)

    x_test = x[start_pos:end_pos]
    y_test = y[start_pos:end_pos]

    model = clone(svc)
    model.fit(x_train, y_train)
    print("Current model score:", model.score(x, y))

    y_result = model.predict(x_test)

    accuracy = sum(y_test == y_result) / y_test.shape[0]

    print("Hit accuracy:", accuracy)

    scores.append(accuracy)

print(scores)
score = np.asarray(scores, dtype=float)

print("Average accuracy:", np.mean(scores))
print("STD. DEV.:", np.std(scores))

