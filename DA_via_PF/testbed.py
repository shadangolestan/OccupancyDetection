import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import sklearn.mixture as mix
import Cluster as clus

#################Index Variables############
DayId = 0
Hour = 1
Weekday = 2
Holiday = 3
EMI = 4
OCC = 5
############################################

def get_performance(actual, prediction):

    e = 1e-10
    
    tp = 0;
    tpr = 0.0
    
    fp = 0
    fpr = 0.0
    
    tn = 0
    tnr = 0.0
    
    fn = 0
    fnr = 0.0
    
    accuracy = 0.0
    rmse = 0.0
    
    precision  = 0.0
    recall = 0.0
    
    for i in range(len(actual)):
        if actual[i] == 1 and prediction[i] == 1:
            accuracy += 1.
            tp += 1
        if actual[i] == 1 and prediction[i] == 0:
            fn += 1
        if actual[i] == 0 and prediction[i] == 0:
            accuracy += 1.
            tn += 1
        if actual[i] == 0 and prediction[i] == 1:
            fp += 1
        rmse += (actual[i] - prediction[i])**2.0
    
    tpr = 100*tp/(1.*(tp+fn+e))
    fpr = 100*fp/(1.*(fp+tn+e))
    tnr = 100*tn/(1.*(tn+fp+e))
    fnr = 100*fn/(1.*(fn+tp+e))
    
    precision = 100*tp/(1.*(tp+fp+e))
    recall = tpr
    
    accuracy /= (1.*len(actual))
    accuracy *= 100.
    rmse /= (1.*len(actual))
    rmse = rmse**0.5
    
    return {'tp':tp, 'tpr':tpr, 'fp':fp, 'fpr':fpr, 'tn':tn, 'tnr':tnr, 'fn':fn, 'fnr':fnr, 'accuracy':accuracy, 'rmse':rmse, 'precision':precision, 'recall':recall}
    
def get_performance_seq(actual, prediction):
    result = {'tp':0, 'tpr':0., 'fp':0, 'fpr':0., 'tn':0, 'tnr':0., 'fn':0, 'fnr':0., 'accuracy':0., 'rmse':0., 'precision':0., 'recall':0.}
    for i in range(len(actual)):
        x = get_performance(actual[i], prediction[i])
        for key in result:
            result[key] += x[key]
    for key in result:
        result[key] /= (1.*len(actual))
    #print(prediction)
    return result
    
def get_actual(data, seqs, cluster):
    actual = []
    for s in seqs:
        x = []
        for i in range(s[0], s[1]):
            x.append(cluster.get_occu_index(data[i][OCC]))
        actual.append(x)
    #print(actual)
    return actual

def get_data(room, Name):

    occ_data = np.loadtxt(fname='data/occ_data_room_'+str(room)+'.csv', delimiter=',', dtype=object, usecols=(4), skiprows=1)
    emi_data = np.loadtxt(fname='data/'+Name+'_data_room_'+str(room)+'.csv', delimiter=',', dtype=object, usecols=(0,1,2,3,4), skiprows=1)
    room_data = np.zeros((emi_data.shape[0], emi_data.shape[1]+1))
    for i in range(room_data.shape[0]):
        room_data[i][DayId] = float(emi_data[i][0])
        room_data[i][Hour] = float(emi_data[i][1].split(':')[0])
        room_data[i][Weekday] = float(emi_data[i][2])
        room_data[i][Holiday] = float(emi_data[i][3])
        room_data[i][EMI] = float(emi_data[i][4])
        room_data[i][OCC] = float(occ_data[i][0])
    return room_data
    
def plot_occu(Actual, Prediction, Time_seq=[]):
    xLabel = 'Time (min)'
    yLabel = 'Occupancy Level'
    plt.title('Occupancy Level: Actual vs Prediction')
    plt.plot(Time_seq, Actual, 'g-', label='Actual')
    plt.plot(Time_seq, Prediction, 'r-', label='Prediction')
    plt.legend()
    plt.show()
###############################################Testing#############################
source = get_data(1, 'co2')
target = get_data(4, 'co2')

source_cluster = clus.Cluster(occu_intervals=[0.0,1.0,np.Inf])
target_cluster = clus.Cluster()

src_train_seqs = [[0, 16300]]
src_test_seqs = [[16300, 16900]]

source_cluster.learn(source, src_train_seqs)

trg_train_seqs = [[0, 1440]]
trg_test_seqs = [[16300, 16900]]

target_cluster.prior(source_cluster)
target_cluster.learn(target, trg_train_seqs)

actual = get_actual(target, trg_test_seqs, target_cluster)
prediction = target_cluster.predict(source, trg_test_seqs, pf={'NP':200}, cluster=source_cluster)


plot_occu(actual[0], prediction[0], Time_seq=list(range(trg_test_seqs[0][0], trg_test_seqs[0][1])))
print(get_performance_seq(actual, prediction))
