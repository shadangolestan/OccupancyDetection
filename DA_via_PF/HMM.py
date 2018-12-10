import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import sklearn.mixture as mix

class Emission_Model():

    def __init__(self, meta={}):
        self.meta_params = meta

    def learn(self, data):
        """Learn parameters"""

    def get_prob(self, value):
        """Get probability"""
        return 0.0

class Gamma(Emission_Model):
    
    def __init__(self, meta={}):
        self.meta_params = meta
        
    def learn(self, data):
        if len(data) == 0:
            data = [0.0]
        self.fit_alpha, self.fit_loc, self.fit_beta = stats.gamma.fit(data)

    def get_prob(self, value):
        p = stats.gamma.pdf(x=value, a=self.fit_alpha, loc=self.fit_loc, scale=self.fit_beta)
        if np.isnan(p) == True:
            return 0.0
        if p > 1.0:
            return 0.9999
        return p

class Categorical(Emission_Model):
    
    def __init__(self, meta={'K':2}):
        self.meta_params = meta

    def learn(self, data):
        self.probs = np.zeros(self.meta_params['K'])
        for i in range(len(data)):
            self.probs[int(data[i])] += 1.
        sum_ = np.sum(self.probs)
        if sum_ > 0:
            self.probs = self.probs/sum_

    def get_prob(self, value):
        return self.probs[int(value)]

class HMM():
    def __init__(self, number_of_hidd_states, emission_type):
        self.number_of_hidd_states = number_of_hidd_states
        
        self.A_count = np.ones((self.number_of_hidd_states, self.number_of_hidd_states))
        self.PI_count = np.ones(self.number_of_hidd_states)
        
        self.B = []
        for i in range(self.number_of_hidd_states):
            obj = emission_type[0](emission_type[1])
            self.B.append(obj)

    def supervised_learn(self, hidd_seqs, emi_seqs):
        ###################Learning A and PI##############
        for i in hidd_seqs:
            self.PI_count[i[0]] += 1
            k = len(i)-1
            for j in range(k):
                self.A_count[i[j]][i[j+1]] += 1.
                
        self.A = np.zeros((self.number_of_hidd_states, self.number_of_hidd_states))
        for i in range(self.number_of_hidd_states):
            self.A[i] = self.A_count[i] / np.sum(self.A_count[i])
        
        self.PI = np.zeros(self.number_of_hidd_states)
        self.PI = self.PI_count / np.sum(self.PI_count)
        #################################################
        
        ##############learning B#########################
        for i in range(self.number_of_hidd_states):
            data = []
            for j in range(len(hidd_seqs)):
                for k in range(len(hidd_seqs[j])):
                    if hidd_seqs[j][k] == i:
                        data.append(emi_seqs[j][k])
            self.B[i].learn(data)
        #################################################

    def viterbi_predict(self, emi_seqs):
        result = []
        for s in emi_seqs:
            T1 = np.zeros((self.number_of_hidd_states, len(s)))
            T2 = np.zeros((self.number_of_hidd_states, len(s)))
            
            for i in range(self.number_of_hidd_states):
                T1[i][0] = self.PI[i] * self.B[i].get_prob(s[0])
                T2[i][0] = 0
            
            for i in range(1, len(s)):
                for j in range(self.number_of_hidd_states):
                    T1[j][i] = np.amax(T1[:, i-1]*self.A[:, j]*self.B[j].get_prob(s[i]))
                    T2[j][i] = np.argmax(T1[:, i-1]*self.A[:, j]*self.B[j].get_prob(s[i]))
            
            z = np.zeros(len(s), dtype=int)
            x = np.zeros(len(s))
            
            T = len(s) - 1
            z[T] = np.argmax(T1[:, T])
            x[T] = z[T]
            for i in range(T, 0, -1):
                z[i-1] = T2[z[i]][i]
                x[i-1] = z[i-1]
            
            result.append(x)
        return result

    def pf_predict(self, emi_seqs, number_of_particles, q, p=None):
        result = []
        if p == None:
            p = (self.A, self.B)
        for s in emi_seqs:
            particles = np.random.choice(list(range(self.number_of_hidd_states)), size=number_of_particles)
            l = len(s)
            x = []
            for i in range(l):
                w = []
                for j in range(number_of_particles):
                    prev = particles[j]
                    particles[j] = np.random.choice(list(range(self.number_of_hidd_states)), p=q[prev])
                    w_ = p[1][particles[j]].get_prob(s[i])*p[0][prev][particles[j]] / q[prev][particles[j]]
                    w.append(w_)
                w = w / np.sum(w)
                new_particles = np.random.choice(particles, size=number_of_particles, p=w)
                particles = new_particles
                x.append(np.amax(particles))
            result.append(x)
        
        return result
            
hmm = HMM(2, (Categorical, {'K':2}))
hidd_seqs = [[0,0,0,1,1,0,1,1,1,0]]
emi_seqs =  [[1,1,1,0,0,0,0,0,0,1]]
hmm.supervised_learn(hidd_seqs=hidd_seqs, emi_seqs=emi_seqs)
#print(range(7))
print(hidd_seqs)
print(hmm.viterbi_predict(emi_seqs))
print(hmm.pf_predict(emi_seqs, 200, q=hmm.A))

    
    
    
    
