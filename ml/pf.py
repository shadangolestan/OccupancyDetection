import numpy as np
import data_loader
import datetime
from scipy.stats import norm

TRAINING_RATIO = 0.8
TIME_INTERVAL = 1
PARTICLES = 1000


# Pr[t][a][b] -> Pr(b_t+1|a_t)
def transition_matrix(obs, states):
    unique_state = np.unique(states)
    pr_transition = []
    pr_count = []
    for t in range(6 * 24):
        pr_transition.append(dict())
        pr_count.append(dict())
        for a in unique_state:
            pr_transition[t][a] = dict()
            pr_count[t][a] = 0
            for b in unique_state:
                pr_transition[t][a][b] = 0

    temp_t = []
    pos_start = 0
    pos_end = 0
    total_num = 0
    first = True
    for i in range(states.shape[0]):
        temp_t.append([obs[i][5], states[i], None])
        total_num += states[i]
        if i != 0:
            if obs[i][5] != obs[i - 1][5] or i == states.shape[0] - 1:
                if first:
                    first = False
                    total_num = 0
                    pos_end = i
                    continue
                result = int((total_num / (pos_end - pos_start)) >= 0.5)
                for j in range(pos_start, pos_end):
                    temp_t[j][2] = result
                pos_start = pos_end - 1
                pos_end = i
                total_num = 0

    for i in temp_t:
        if i[2] == None:
            continue
        i = list(map(int, i))
        pr_count[i[0]][i[1]] += 1
        pr_transition[i[0]][i[1]][i[2]] += 1

    for t in range(6 * 24):
        for a in pr_transition[t].keys():
            for b in pr_transition[t][a].keys():
                if pr_count[t][a] == 0:
                    pr_transition[t][a][b] = 0.5
                else:
                    pr_transition[t][a][b] /= pr_count[t][a]

    return pr_transition


def state_observation_matrix(obs, states):
    unique_state = np.unique(states)
    pr = []
    for t in range(6 * 24):
        pr.append(dict())
        for a in unique_state:
            z = obs[states == a, :]
            z = z[z[:, 5] == t, :]
            pr[t][a] = dict()
            for i in range(z.shape[1] - 1):
                pr[t][a][i] = (np.mean(z[:, i]), np.var(z[:, i]))
                if np.isnan(pr[t][a][i][0]):
                    pr[t][a][i] = (np.mean(obs[:, i]), np.std(obs[:, i]))

    return pr


def evaluate(pr_t, pr_r, data):
    result = []
    prev = 0
    for i in range(data.shape[0]):
        all_result = []
        for _ in range(PARTICLES):
            sample_s = 0
            if np.random.random() < pr_t[(int(data[i][5]) - 1) % (6 * 24)][prev][1]:
                sample_s = 1

            weight_s = 1
            for pos in range(data.shape[1] - 1):
                weight_s *= norm.pdf(data[i][pos], loc=pr_r[int(data[i][5])][sample_s][pos][0],
                                     scale=pr_r[int(data[i][5])][sample_s][pos][1])

            all_result.append([sample_s, weight_s])

        counter = [0, 0]
        for j in range(PARTICLES):
            if np.random.random() < all_result[j][1]:
                counter[all_result[j][0]] += 1
        if counter[0] >= counter[1]:
            result.append(0)
        else:
            result.append(1)

        # print(i * 100 / data.shape[0])
    return result


# import some data to play with
data_library = data_loader.load_data()
x = data_library.data
y = data_library.label
x = np.concatenate((x, np.reshape(data_library.date, (-1, 1))), axis=1)

num_lines = int(x.shape[0] * (1 - TRAINING_RATIO))

scores = []

for i in range(int(1 // (1 - TRAINING_RATIO))):
    start_pos = num_lines * i
    end_pos = min(start_pos + num_lines, x.shape[0])

    x_train = np.concatenate((x[:start_pos], x[end_pos:]), axis=0)
    y_train = np.concatenate((y[:start_pos], y[end_pos:]), axis=0)

    x_test = x[start_pos:end_pos]
    y_test = y[start_pos:end_pos]

    pr_t = transition_matrix(x_train, y_train)
    pr_re = state_observation_matrix(x_train, y_train)

    y_result = evaluate(pr_t, pr_re, x_test)
    accuracy = sum(y_test == y_result) / y_test.shape[0]
    print("Hit accuracy:", accuracy)
    scores.append(accuracy)

    # y_result = model.predict(x_test)
    #
    # accuracy = sum(y_test == y_result) / y_test.shape[0]
    #
    # print("Hit accuracy:", accuracy)
    #
    # scores.append(accuracy)

print(scores)
score = np.asarray(scores, dtype=float)

print("Average accuracy:", np.mean(scores))
print("STD. DEV.:", np.std(scores))
