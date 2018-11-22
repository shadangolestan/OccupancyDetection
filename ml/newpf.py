import numpy as np
import data_loader
import matplotlib.pyplot as plt
from scipy.stats.kde import gaussian_kde
from scipy import stats
from tqdm import tqdm

TRAINING_RATIO = 0.8
TIME_INTERVAL = [0, 6 * 60, 18 * 60, 24 * 60 + 1]
PARTICLES = 100
PLOT_PDF = False
SET = 1
DATASET = ["ml", "pcl"]
LABELS = [["Temperature", "Humidity", "Light", "CO2", "HumidityRatio"],
          ["TL1", "TL3", "TL5", "TL6", "TL8", "temperature", "cloudCover"]]


# linear
def get_interval(value):
    for i in range(len(TIME_INTERVAL) - 1):
        if value < TIME_INTERVAL[i + 1]:
            return i


# Px
def motion_model(obs, occupants):
    # px[interval][occu] = poss
    px = []
    unique_occupant = np.unique(occupants)
    for i in range(len(TIME_INTERVAL) - 1):
        px.append(dict())
        for possible_value in unique_occupant:
            px[i][possible_value] = 0

    for i in range(len(TIME_INTERVAL) - 1):
        interval_indices = (obs[:, -1] < TIME_INTERVAL[i + 1]) & (obs[:, -1] >= TIME_INTERVAL[i])
        occupants_in_interval = occupants[interval_indices]
        total_row = occupants_in_interval.shape[0]
        for possible_value in unique_occupant:
            px[i][possible_value] = np.sum(occupants_in_interval == possible_value) / total_row

    return px


# P(Z|X, TI)
def sensor_model(obs, occupants):
    # gaussian[col of feature][interval][occupancy X] = [mean, var]
    gaussian = []
    unique_occupant = np.unique(occupants)

    for feature in range(obs.shape[1] - 1):

        gaussian.append([])

        for i in range(len(TIME_INTERVAL) - 1):

            gaussian[feature].append({})
            interval_indices = (obs[:, -1] < TIME_INTERVAL[i + 1]) & (obs[:, -1] >= TIME_INTERVAL[i])
            obs_in_interval = obs[interval_indices]
            occupants_in_interval = occupants[interval_indices]

            for possible_value in unique_occupant:

                occupant_interval_indices = occupants_in_interval == possible_value
                obs_in_interval_occupant = obs_in_interval[occupant_interval_indices]

                if obs_in_interval_occupant.shape[0] != 0:
                    # mean = np.mean(obs_in_interval_occupant[:, feature])
                    # var = np.var(obs_in_interval_occupant[:, feature])

                    if np.sum(obs_in_interval_occupant[:, feature]) != 0:
                        kde = gaussian_kde(obs_in_interval_occupant[:, feature])

                        if PLOT_PDF:
                            minimum = np.min(obs_in_interval_occupant[:, feature])
                            maximum = np.max(obs_in_interval_occupant[:, feature])

                            num_bar = 500

                            dist_space = np.linspace(minimum, maximum, num_bar)
                            title = "PDF for time interval ["
                            title += str(TIME_INTERVAL[i] // 60) + ":00"
                            title += ", "
                            title += str(TIME_INTERVAL[i + 1] // 60) + ":00"
                            title += ")"
                            title += " for "
                            title += LABELS[SET][feature]

                            file_name = DATASET[SET] + '-' + str(i) + '-' + LABELS[SET][feature] + ".png"

                            plt.hist(obs_in_interval_occupant[:, feature], density=1, alpha=0.5, facecolor="green")
                            plt.plot(dist_space, kde.pdf(dist_space), label=LABELS[SET][feature], color="blue")
                            plt.legend()
                            plt.title(title)
                            plt.savefig("./figures/" + DATASET[SET] + '/' + file_name)
                            plt.close()

                        gaussian[feature][i][possible_value] = kde
                    else:
                        gaussian[feature][i][possible_value] = 0
                else:
                    gaussian[feature][i][possible_value] = None

    return gaussian


def sampled_particle_filter(px, gaussian, obs):
    result = []
    for i in tqdm(range(obs.shape[0])):

        candidate_x = []
        candidate_w = []

        interval = get_interval(obs[i, -1])

        for x in px[interval].keys():

            prob = px[interval][x]
            # gaussian[col of feature][interval][occupancy X] = kde
            ws_t = 1
            for feature in range(obs.shape[1] - 1):
                ws_t *= gaussian[feature][interval][x].pdf(obs[i, feature])

            candidate_x.append(x)
            candidate_w.append(ws_t * prob)

        candidate_w = np.asarray(candidate_w)
        candidate_w /= np.sum(candidate_w)
        candidate_w = np.reshape(candidate_w, (-1))

        result.append(np.random.choice(candidate_x, p=candidate_w))
    return np.asarray(result)


def particle_filter(px, gaussian, obs):
    result = []
    for i in tqdm(range(obs.shape[0])):
        candidate_x = []
        candidate_w = []

        for s in range(PARTICLES):
            interval = get_interval(obs[i, -1])
            prob = px[interval]
            xs_t = np.random.choice(list(prob.keys()), p=list(prob.values()))
            # gaussian[col of feature][interval][occupancy X] = kde
            ws_t = 1
            for feature in range(obs.shape[1] - 1):
                ws_t *= gaussian[feature][interval][xs_t].pdf(obs[i, feature])

            candidate_x.append(xs_t)
            candidate_w.append(ws_t)

        index = 0
        while index < len(candidate_x):
            if np.random.random() < candidate_w[index]:
                index += 1
            else:
                candidate_x.pop(index)
                candidate_w.pop(index)

        candidate_x = np.asarray(candidate_x)
        result.append(stats.mode(candidate_x)[0])
    return np.asarray(result)


# import some data to play with
if SET == 0:
    data_library = data_loader.load_data()
elif SET == 1:
    data_library = data_loader.load_pcl_data()
else:
    data_library = None
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

    px = motion_model(x_train, y_train)
    gaussian = sensor_model(x, y)

    y_result = sampled_particle_filter(px, gaussian, x_test)

    accuracy = sum(y_test == y_result) / y_test.shape[0]
    print("Hit accuracy:", accuracy)
    scores.append(accuracy)

print(scores)
score = np.asarray(scores, dtype=float)

print("Average accuracy:", np.mean(scores))
print("STD. DEV.:", np.std(scores))
