import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import sklearn.mixture as mix

#####################Condition Variable##########
room = 1
Name = 'co2'
#################################################

#################Index Variables############
DayId = 0
Hour = 1
Weekday = 2
Holiday = 3
CO2 = 4
OCC = 5
############################################

######################Loading Data####################
occ_data = np.loadtxt(fname='data/occ_data_room_'+str(room)+'.csv', delimiter=',', dtype=object, usecols=(4), skiprows=1)
co2_data = np.loadtxt(fname='data/'+Name+'_data_room_'+str(room)+'.csv', delimiter=',', dtype=object, usecols=(0,1,2,3,4), skiprows=1)
room_data = np.zeros((co2_data.shape[0], co2_data.shape[1]+1))
for i in range(room_data.shape[0]):
    room_data[i][DayId] = float(co2_data[i][0])
    room_data[i][Hour] = float(co2_data[i][1].split(':')[0])
    room_data[i][Weekday] = float(co2_data[i][2])
    room_data[i][Holiday] = float(co2_data[i][3])
    room_data[i][CO2] = float(co2_data[i][4])
    if float(occ_data[i][0]) > 0.0:
        room_data[i][OCC] = 1.0
######################################################

######################Loading Data4####################
room = 4
occ_data = np.loadtxt(fname='data/occ_data_room_'+str(room)+'.csv', delimiter=',', dtype=object, usecols=(4), skiprows=1)
co2_data = np.loadtxt(fname='data/'+Name+'_data_room_'+str(room)+'.csv', delimiter=',', dtype=object, usecols=(0,1,2,3,4), skiprows=1)
room_data4 = np.zeros((co2_data.shape[0], co2_data.shape[1]+1))
for i in range(room_data.shape[0]):
    room_data4[i][DayId] = float(co2_data[i][0])
    room_data4[i][Hour] = float(co2_data[i][1].split(':')[0])
    room_data4[i][Weekday] = float(co2_data[i][2])
    room_data4[i][Holiday] = float(co2_data[i][3])
    room_data4[i][CO2] = float(co2_data[i][4])
    if float(occ_data[i][0]) > 0.0:
        room_data4[i][OCC] = 1.0
######################################################

####################Sequence Generator################
def seq_gen(data, number_of_seq, length, from_begin=False):
    if from_begin==True:
        return [(0, length)]
    seq = []
    for i in range(number_of_seq):
        a = np.random.randint(data.shape[0])
        b = a + length
        if b >= data.shape[0]:
            b = data.shape[0]
        seq.append((a,b))
    return seq
######################################################

############################Gamma HMM Seq#############
class HMM_Gamma_Set:
    def __init__(self, time_slot):
        self.number_of_HMM = len(time_slot)-1
        self.time_slot = np.copy(time_slot)
        
        self.Acount = np.ones((self.number_of_HMM, 2, 2))
        self.PIcount = np.ones((self.number_of_HMM, 2))
        
        self.A = np.zeros((self.number_of_HMM, 2, 2))
        self.B = np.zeros((self.number_of_HMM, 2, 3))
        self.PI = np.zeros((self.number_of_HMM, 2))
        
    def get_time_slot(self, hour):
        for i in range(self.number_of_HMM):
            if hour >= self.time_slot[i] and hour < self.time_slot[i+1]:
                return i
        return -1
        
    def prior(self, A, PI):
        for i in range(self.number_of_HMM):
            self.Acount[i] = np.copy(A[i])
            self.PIcount[i] = np.copy(PI[i])
        
    def learn(self, data, seq):
        
        for i in range(self.number_of_HMM):
            co2_values0 = [0.0, 1.0]
            co2_values1 = [0.0, 1.0]
            for s in seq:
                hmm_index = self.get_time_slot(data[s[0]][Hour])
                if hmm_index == i:
                    self.PIcount[i][int(data[s[0]][OCC])] += 1
                
                for j in range(s[0], s[1]-1):
                    hmm_index = self.get_time_slot(data[j][Hour])
                    if hmm_index == i:
                        self.Acount[i][int(data[j][OCC])][int(data[j+1][OCC])] += 1
                        
                for j in range(s[0], s[1]):
                    hmm_index = self.get_time_slot(data[j][Hour])
                    if hmm_index == i:
                        if int(data[j][OCC]) == 1:
                            co2_values1.append(data[j][CO2])
                        else:
                            co2_values0.append(data[j][CO2])
                        
            #print(co2_values0)
            #print(co2_values1)
            Sum = np.sum(self.PIcount[i])
            self.PI[i][0] = self.PIcount[i][0]/Sum
            self.PI[i][1] = self.PIcount[i][1]/Sum
            
            Sum = np.sum(self.Acount[i][0])
            self.A[i][0][0] = self.Acount[i][0][0]/Sum
            self.A[i][0][1] = self.Acount[i][0][1]/Sum
            
            Sum = np.sum(self.Acount[i][1])
            self.A[i][1][0] = self.Acount[i][1][0]/Sum
            self.A[i][1][1] = self.Acount[i][1][1]/Sum
            
            fit_alpha, fit_loc, fit_beta = stats.gamma.fit(co2_values0)
            self.B[i][0][0] = fit_alpha
            self.B[i][0][1] = fit_loc
            self.B[i][0][2] = fit_beta
            
            fit_alpha, fit_loc, fit_beta = stats.gamma.fit(co2_values1)
            self.B[i][1][0] = fit_alpha
            self.B[i][1][1] = fit_loc
            self.B[i][1][2] = fit_beta
##############################################################

########################Get Index From Hour###############
def get_time_slot(time_slot, number_of_HMM, hour):
        for i in range(number_of_HMM):
            if hour >= time_slot[i] and hour < time_slot[i+1]:
                return i
        return -1
##########################################################

#########################PAFI###############################
def PaFi(data, seq, time_slot, NP, q, p_t, p_e):
    number_of_HMM = p_t.shape[0]
    particles = np.zeros((number_of_HMM, NP), dtype=int)
    for i in range(number_of_HMM):
        particles[i] = np.random.choice([0,1],size=NP,p=[0.5, 0.5])
    w = np.zeros(NP)
    
    predict = []
    for i in range(seq[0][0], seq[0][1]):
        index = get_time_slot(time_slot, number_of_HMM, data[i][Hour])
        for j in range(NP):
            prev = particles[index][j]
            particles[index][j] = np.random.choice([0,1],p=q[index][prev])
            new = particles[index][j]
            fit_alpha = p_e[index][new][0]
            fit_loc = p_e[index][new][1]
            fit_beta = p_e[index][new][2]
            w[j] = (stats.gamma.pdf(x=data[i][CO2], a=fit_alpha, loc=fit_loc, scale=fit_beta)*p_t[index][prev][new])/q[index][prev][new]
            
        w = w/np.sum(w)
        r = np.random.choice(particles[index],size=NP,p=w)
        avg = np.sum(r)/(1.0*NP)
        if avg >= 0.5:
            predict.append(1)
        else:
            predict.append(0)
    
    return predict
##############################################################################

time_slot = [0,8,12,13,17,24]
HMMs = HMM_Gamma_Set(time_slot)
HMMs.learn(room_data, [(0,7200)])
HMMt = HMM_Gamma_Set(time_slot)

for i in range(1, 21):
    test = [(1440, 1440+i*30)]

    HMMt.prior(HMMs.A, HMMs.PI)
    HMMt.learn(room_data4, [(0,1440)])

    predict = PaFi(room_data4, test, time_slot, 300, HMMs.A, HMMt.A, HMMs.B)

    accuracy = 0.0
    actual = []
    for i in range(test[0][0], test[0][1]):
        actual.append(int(room_data4[i][OCC]))
    #print(actual)
    for j in range(len(actual)):
        if predict[j] == actual[j]:
            accuracy += 1.0
    accuracy/=len(actual)
    accuracy*=100

    print(accuracy)

"""
DA = []
woDA = []
time = []

predict_hour = 2
for i in range(predict_hour):
    time.append((i+1)*60)
    
    HMMt = HMM_Gamma_Set(time_slot)
    #HMMt.prior(HMMs.A, HMMs.PI)
    HMMt.learn(room_data4, [(0,1440)])

    test = [(1440, 1440+(i+1)*60)]
    predict = PaFi(room_data4, test, time_slot, 300, HMMt.A, HMMt.A, HMMt.B)

    accuracy = 0.0
    actual = []
    for i in range(test[0][0], test[0][1]):
        actual.append(int(room_data4[i][OCC]))
    #print(actual)
    for j in range(len(actual)):
        if predict[j] == actual[j]:
            accuracy += 1.0
    accuracy/=len(actual)
    accuracy*=100
    
    woDA.append(accuracy)
    
    HMMt = HMM_Gamma_Set(time_slot)
    HMMt.prior(HMMs.A, HMMs.PI)
    HMMt.learn(room_data4, [(0,1440)])

    test = [(1440, 1440+(i+1)*60)]
    predict = PaFi(room_data4, test, time_slot, 300, HMMs.A, HMMt.A, HMMs.B)

    accuracy = 0.0
    actual = []
    for i in range(test[0][0], test[0][1]):
        actual.append(int(room_data4[i][OCC]))
    #print(actual)
    for j in range(len(actual)):
        if predict[j] == actual[j]:
            accuracy += 1.0
    accuracy/=len(actual)
    accuracy*=100
    
    DA.append(accuracy)
    print(accuracy)

plt.title('Accuracy With or Without Domain Adaptation')
plt.xlabel('Prediction Time (Minute)')
plt.ylabel('Accuracy')
plt.plot(waDA, time, 'r.', label='Without DA')
plt.plot(DA, time, 'g.', label='With DA')

plt.legend()
plt.show()
"""
#print(' Accuracy: '+str(accuracy))



#############################################################################
"""
time_slot = [0,8,12,13,17,24]
HMM = HMM_Gamma_Set(time_slot)
HMM.learn(room_data, seq_gen(room_data, 500, 100))
#test = seq_gen(room_data, 1, 100)
test = seq_gen(room_data, 1, 500)
predict = PaFi(room_data, test, time_slot, 300, HMM.A, HMM.A, HMM.B)
print(predict)
accuracy = 0.0
actual = []
for i in range(test[0][0], test[0][1]):
    actual.append(int(room_data[i][OCC]))
print(actual)
for j in range(len(actual)):
    if predict[j] == actual[j]:
        accuracy += 1.0
accuracy/=len(actual)
accuracy*=100

print(' Accuracy: '+str(accuracy))
"""
################################################################################
"""
###############################Cross Validation For Source#################
K = 5
alpha = 100.0/100
L = int(room_data.shape[0]*alpha)
#L = 5000
m = int(L/K)

time_slot = [0,8,12,17,24]

avg_accuracy = 0.0
a = 0
for i in range(K):
    b = a + m
    HMM = HMM_Gamma_Set(time_slot)
    
    HMM.learn(room_data, [(0,a), (b,L)])
    
    predict = PaFi(room_data, [(a,b)], time_slot, 300, HMM.A, HMM.A, HMM.B)
    
    accuracy = 0.0
    actual = room_data[a:b][OCC]
    for j in range(actual.shape[0]):
        if predict[j] == actual[j]:
            accuracy += 1.0
    accuracy/=actual.shape[0]
    accuracy*=100
    
    print('Round '+str(i)+' Accuracy: '+str(accuracy))
    
    avg_accuracy += accuracy
    a+=m

avg_accuracy/=K

print('Validation Accuracy: '+str(avg_accuracy))
"""
        
        
        
        
        
        
        
        
