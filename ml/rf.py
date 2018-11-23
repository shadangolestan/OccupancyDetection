from sklearn.ensemble import RandomForestClassifier
from sklearn.base import clone
import pandas as pd
import numpy as np
import data_loader

TRAINING_RATIO = 0.8
SET = 1
DATASET = ["ml", "pcl"]

if SET == 0:
    data_library = data_loader.load_data()
elif SET == 1:
    data_library = data_loader.load_pcl_data()
else:
    data_library = None

x = data_library.data
y = data_library.label

# data = pd.DataFrame({
#     "Temperature": x[:, 0],
#     "Humidity": x[:, 1],
#     "Light": x[:, 2],
#     "CO2": x[:, 3],
#     "HumidityRatio": x[:, 4],
#     "Occupancy": y
# })
#
# data.head()

clf = RandomForestClassifier(n_estimators=200)

# feature_imp = pd.Series(clf.feature_importances_,index=iris.feature_names).sort_values(ascending=False)
# print(feature_imp)

num_lines = int(x.shape[0] * (1 - TRAINING_RATIO))

scores = []

for i in range(int(1 // (1 - TRAINING_RATIO))):
    start_pos = num_lines * i
    end_pos = min(start_pos + num_lines, x.shape[0])

    x_train = np.concatenate((x[:start_pos], x[end_pos:]), axis=0)
    y_train = np.concatenate((y[:start_pos], y[end_pos:]), axis=0)

    x_test = x[start_pos:end_pos]
    y_test = y[start_pos:end_pos]

    model = clone(clf)
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