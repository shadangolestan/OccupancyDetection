import numpy as np
import data_analyzer as da

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
        
    def supervised_learn(self, data, seq, prior=None):
        if prior != None:
            self.A_count = np.copy(prior[0])
            self.B_count = np.copy(prior[1])
            self.PI_count = np.copy(prior[2])
            
        
        data_points = []
        for i in range(self.number_of_hmms):
            x = []
            for s in seq:
                for j in range(s[0], s[1]):
                    if self.get_hmm_index(data[j][da.Hour]) == i:
                        x.append(j)
            data_points.append(x)
            
