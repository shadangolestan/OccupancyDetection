import numpy as np
import data_analyzer as da
import scipy.stats as stats

class HMM_Cluster():
    def __init__(self, time_intervals=None, occu_intervals=None, emi_intervals=None):
        if time_intervals == None:
            self.time_intervals = [0,8,17,24]
        else:
            self.time_intervals = time_intervals
            
        if occu_intervals == None:
            self.occu_intervals = [0.0,1.0,np.Inf]
        else:
            self.occu_intervals = occu_intervals
            
        if emi_intervals == None:
            self.emi_intervals = [0.0,450.0,np.Inf]
        else:
            self.emi_intervals = emi_intervals
        
        self.number_of_hmms = len(self.time_intervals) - 1
        self.number_of_hidd_states = len(self.occu_intervals) - 1
        self.number_of_emi_states = len(self.emi_intervals) - 1
        
        self.A_count = np.ones((self.number_of_hmms, self.number_of_hidd_states, self.number_of_hidd_states), dtype=int)
        self.B_count = np.ones((self.number_of_hmms, self.number_of_hidd_states, self.number_of_emi_states), dtype=int)
        self.PI_count = np.ones((self.number_of_hmms, self.number_of_hidd_states), dtype=int)
        
        self.A = np.zeros((self.number_of_hmms, self.number_of_hidd_states, self.number_of_hidd_states), dtype=float)
        self.B = np.zeros((self.number_of_hmms, self.number_of_hidd_states, self.number_of_emi_states), dtype=float)
        self.PI = np.zeros((self.number_of_hmms, self.number_of_hidd_states), dtype=float)
        
    def get_hmm_index(self, hour):
        for i in range(self.number_of_hmms):
            if hour >= self.time_intervals[i] and hour < self.time_intervals[i+1]:
                return i
        return -1
        
    def get_occu_index(self, occu):
        for i in range(self.number_of_hidd_states):
            if occu >= self.occu_intervals[i] and occu < self.occu_intervals[i+1]:
                return i
        return -1
    
    def get_emi_index(self, emi):
        for i in range(self.number_of_emi_states):
            if emi >= self.emi_intervals[i] and emi < self.emi_intervals[i+1]:
                return i
        return -1
        
    def emi_prob(self, hmm_, hidd_, emi_, type_=None):
        if type_ == None:
            return self.B[hmm_][hidd_][sefl.get_emi_index(emi)]
        if type_ == 'gamma':
            fit_alpha = self.gamma_params[hmm_][hidd_][0]
            fit_loc = self.gamma_params[hmm_][hidd_][1]
            fit_beta = self.gamma_params[hmm_][hidd_][2]
            return stats.gamma.pdf(x=emi, a=fit_alpha, loc=fit_loc, scale=fit_beta)
        return 0.0
        
    def supervised_learn(self, data, seq, prior=None):
        if prior != None:
            self.A_count = np.copy(prior[0])
            self.B_count = np.copy(prior[1])
            self.PI_count = np.copy(prior[2])
            
        
        data_points = []
        for i in range(self.number_of_hmms):
            x = []
            prev = -10
            y = []
            for j in range(seq[0], seq[1]):
                hmm_ = self.get_hmm_index(data[j][da.Hour])
                if hmm_ == i and prev != i:
                    y = [j]
                if hmm_ == i and prev == i:
                    y.append(j)
                if hmm_ != i and prev == i:
                    x.append(y)
                    y = []
                prev = hmm_
            if len(y) > 0:
                x.append(y)
            data_points.append(x)


        #############################Learning A####################
        for i in range(self.number_of_hmms):
            for s in data_points[i]:
                for j in s[:-1]:
                    from_ = self.get_occu_index(data[j][da.OCC])
                    to_ = self.get_occu_index(data[j+1][da.OCC])
                    self.A_count[i][from_][to_] += 1
            for j in range(self.number_of_hidd_states):
                sum_ = 1.*np.sum(self.A_count[i][j])
                self.A[i][j] = self.A_count[i][j]/sum_
        ###########################################################

        #############################Learning B####################
        for i in range(self.number_of_hmms):
            for s in data_points[i]:
                for j in s[:-1]:
                    emi_ = self.get_emi_index(data[j][da.EMI])
                    from_ = self.get_occu_index(data[j][da.OCC])
                    self.B_count[i][from_][emi_] += 1
            for j in range(self.number_of_hidd_states):
                sum_ = 1.*np.sum(self.B_count[i][j])
                self.B[i][j] = self.B_count[i][j]/sum_
        ###########################################################

        #############################Learning B Gamma##############
        self.gamma_params = np.zeros((self.number_of_hmms, self.number_of_hidd_states, 3))
        gamma_ = []
        for i in range(self.number_of_hmms):
            x = [] 
            for j in range(self.number_of_hidd_states):
                y = [0.0, 0.5]
                for s in data_points[i]:
                    for k in s:
                        hidd_ = self.get_occu_index(data[k][da.OCC])
                        if hidd_ == j:
                            y.append(data[i][da.EMI])
                x.append(y)
            gamma_.append(x)
        
        for i in range(self.number_of_hmms):
            for j in range(self.number_of_hidd_states):
                fit_alpha, fit_loc, fit_beta = stats.gamma.fit(gamma_[i][j])
                self.gamma_params[i][j][0] = fit_alpha
                self.gamma_params[i][j][1] = fit_loc
                self.gamma_params[i][j][2] = fit_beta
        ###########################################################

        #############################Learning PI####################
        for i in range(self.number_of_hmms):
            for s in data_points[i]:
                hidd_ = self.get_occu_index(data[s[0]][da.OCC])
                self.PI_count[i][hidd_] += 1
            sum_ = 1.*np.sum(self.PI_count[i])
            self.PI[i] = self.PI_count[i]/sum_
        ###########################################################

        print(self.gamma_params)
        print(self.PI_count)

HMM = HMM_Cluster()
HMM.supervised_learn(da.room_data, seq=[0, 2880])
