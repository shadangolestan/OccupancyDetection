import csv
import numpy as np
from datetime import datetime


class Data:
    def __init__(self):
        self.data = []
        self.label = []
        self.date = []

    def to_np(self):
        self.data = np.asarray(self.data, dtype=float)
        self.label = np.asarray(self.label, dtype=int)
        self.date = np.asarray(self.date, dtype=int)


def load_data():
    data_list = Data()
    with open("ml-aggregate.csv", 'r') as input_file:
        reader = csv.reader(input_file, delimiter=',')
        next(reader)
        for row in reader:
            if len(row) == 0:
                continue
            data_list.data.append(row[2:7])
            data_list.label.append(row[7])
            curr_time = datetime.strptime(row[1], "%Y-%m-%d %H:%M:%S")
            minute = curr_time.minute + (curr_time.second >= 30)
            data_list.date.append((minute + 60 * curr_time.hour))

    data_list.to_np()
    return data_list


def load_pcl_data(binary=True, file_name="110107.csv"):
    data_list = Data()
    with open(file_name, 'r') as input_file:
        reader = csv.reader(input_file, delimiter=',')
        next(reader)
        for row in reader:
            if len(row) == 0:
                continue
            data_list.data.append(row[1:5] + row[-2:])
            occupant = 0
            if row[-3] != '0':
                if binary:
                    occupant = 1
                else:
                    occupant = int(row[-3])
            data_list.label.append(occupant)
            curr_time = datetime.strptime(row[0], "%Y/%m/%d %H:%M")
            minute = curr_time.minute
            data_list.date.append((minute + 60 * curr_time.hour))

    data_list.to_np()
    return data_list


if __name__ == '__main__':
    scores = [0.90882975, 0.91218156, 0.90844667, 0.9051752, 0.88412184]
    score = np.asarray(scores, dtype=float)

    print("Average accuracy:", np.mean(scores))
    print("STD. DEV.:", np.std(scores))
