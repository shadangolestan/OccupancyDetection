import HMM as HMM
import data_analyzer as da
import numpy as np

class Cluster():
    def __init__(self, emission_type=None, time_intervals=None, occu_intervals=None, emi_intervals=None):
        if emission_type == None:
            self.emission_type = (HMM.Gamma, {})
        else:
            self.emission_type = emission_type
            
        self.emission_name = self.emission_type[0]().get_name()
        
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
        
        self.hmms = []
        for i in range(self.number_of_hmms):
            hmm = HMM.HMM(self.number_of_hidd_states, emission_type=self.emission_type)
            self.hmms.append(hmm)
        
            
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
        
    def get_data_seqs(self, data, seqs, hmm_index):
        hidd_seqs = []
        emi_seqs = []
        index_seqs = []
        for s in seqs:
            hidd = []
            emi = []
            index = []
            for j in range(s[0], s[1]):
                if self.get_hmm_index(data[j][da.Hour]) == hmm_index:
                    index.append(j)
                    hidd.append(self.get_occu_index(data[j][da.OCC]))
                    if self.emission_name == 'Categorical':
                        emi.append(self.get_emi_index(data[j][da.EMI]))
                    else:
                        emi.append(data[j][da.EMI])

            hidd_seqs.append(hidd)
            emi_seqs.append(emi)
            index_seqs.append(index)

        return (index_seqs, hidd_seqs, emi_seqs)
        
    def get_all_data(self, data, seqs):
        hidd_seqs = []
        emi_seqs = []
        index_seqs = []
        for s in seqs:
            hidd = []
            emi = []
            index = []
            for j in range(s[0], s[1]):
                index.append(j)
                hidd.append(self.get_occu_index(data[j][da.OCC]))
                if self.emission_name == 'Categorical':
                    emi.append(self.get_emi_index(data[j][da.EMI]))
                else:
                    emi.append(data[j][da.EMI])

            hidd_seqs.append(hidd)
            emi_seqs.append(emi)
            index_seqs.append(index)

        return (index_seqs, hidd_seqs, emi_seqs)
        
    def prior(self, prior_cluster):
        for i in range(self.number_of_hmms):
            self.hmms[i].prior((prior_cluster.hmms[i].A_count, prior_cluster.hmms[i].B, prior_cluster.hmms[i].PI_count))

    def learn(self, data, seqs, meta={'supervised':True}):
        all_index_seqs, all_hidd_seqs, all_emi_seqs = self.get_all_data(data, seqs)
        if meta['supervised'] == True:
            for i in range(self.number_of_hmms):
                index_seqs, hidd_seqs, emi_seqs = self.get_data_seqs(data, seqs, i)
                self.hmms[i].supervised_learn(hidd_seqs, emi_seqs)
                self.hmms[i].learn_emission(hidd_seqs, emi_seqs)
        else:
            for i in range(self.number_of_hmms):
                index_seqs, hidd_seqs, emi_seqs = self.get_data_seqs(data, seqs, i)
                self.hmms[i].unsupervised_learn(emi_seqs)
                
    def predict(self, data, seqs, pf=None, cluster=None):
        if cluster == None:
            cluster = self
        result = []
        for s in seqs:
            result.append(np.zeros(s[1]-s[0], dtype=int))
            
        if pf == None:
            for i in range(self.number_of_hmms):
                index_seqs, hidd_seqs, emi_seqs = self.get_data_seqs(data, seqs, i)
                x = self.hmms[i].viterbi_predict(emi_seqs)
                for j in range(len(seqs)):
                    for k in range(len(index_seqs[j])):
                        result[j][index_seqs[j][k]-seqs[j][0]] = x[j][k]
        else:
            for i in range(self.number_of_hmms):
                index_seqs, hidd_seqs, emi_seqs = self.get_data_seqs(data, seqs, i)
                x = self.hmms[i].pf_predict(emi_seqs, pf['NP'], cluster.hmms[i].A, None)
                for j in range(len(seqs)):
                    for k in range(len(index_seqs[j])):
                        result[j][index_seqs[j][k]-seqs[j][0]] = x[j][k]
            
        return result
if __name__ == '__main__':
    C = Cluster()
    C.learn(da.room_data, [[0, 2440]], {'supervised':True})
    print(C.predict(da.room_data, [[0, 2440]])[0])
