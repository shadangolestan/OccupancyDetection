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
        
    def update_prob(self, value, prob):
        """Update the probability"""
        
    def normalize(self):
        """normalize to make pmf"""
        
    def get_meta(self):
        """Return the meta paprameters"""
        return self.meta_params
        
    def get_name(self):
        """Return the name of the model"""
        return 'Emission_Model'

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
        
    def get_name(self):
        return 'Gamma'

class Categorical(Emission_Model):
    
    def __init__(self, meta={'K':2}):
        self.meta_params = meta
        self.probs = np.zeros(self.meta_params['K'])

    def learn(self, data):
        for i in range(len(data)):
            self.probs[int(data[i])] += 1.
        sum_ = np.sum(self.probs)
        if sum_ > 0:
            self.probs = self.probs/sum_

    def get_prob(self, value):
        return self.probs[int(value)]

    def update_prob(self, value, prob):
        self.probs[int(value)] = prob
    
    def normalize(self):
        self.probs = self.probs / np.sum(self.probs)

    def get_meta(self):
        return self.meta_params
        
    def get_name(self):
        return 'Categorical'

class HMM():
    def __init__(self, number_of_hidd_states, emission_type):
        self.number_of_hidd_states = number_of_hidd_states
        
        self.A_count = np.ones((self.number_of_hidd_states, self.number_of_hidd_states))
        self.PI_count = np.ones(self.number_of_hidd_states)
        
        self.A = np.zeros((self.number_of_hidd_states, self.number_of_hidd_states))
        self.PI = np.zeros(self.number_of_hidd_states)

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
                

        for i in range(self.number_of_hidd_states):
            self.A[i] = self.A_count[i] / np.sum(self.A_count[i])
        
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
        
    def unsupervised_learn(self, emi_seqs, prior=None):
        k = self.B[0].get_meta()['K'] ######Assumming B is Categoriacl#######
        
        if prior == None:
            for i in range(self.number_of_hidd_states):
                self.A[i] = np.random.rand(self.number_of_hidd_states)
                self.A[i] /= np.sum(self.A[i])
                p = np.random.rand(k)
                p /= np.sum(p)
                for j in range(k):
                    self.B[i].update_prob(j, p[j])
            self.PI = np.random.rand(self.number_of_hidd_states)
            self.PI /= np.sum(self.PI)
        else:
            self.A = np.copy(prior[0])
            self.PI = np.copy(prior[2])
            for i in range(self.number_of_hidd_states):
                for j in range(k):
                    self.B[i].update_prob(j, prior[1].get_prob(j))
        
        
        B_probs = np.zeros((self.number_of_hidd_states, k))

        for i in range(self.number_of_hidd_states):
            for j in range(k):
                B_probs[i][j] = self.B[i].get_prob(j)

        tolerance = 0.01
        max_iteration = 1000
        
        for s in emi_seqs:
            T = len(s)
            alpha = np.zeros((self.number_of_hidd_states, T))
            beta = np.zeros((self.number_of_hidd_states, T))
            
            for iteration in range(max_iteration):
                #################Forward Procedure##########
                for i in range(self.number_of_hidd_states):
                    alpha[i][0] = self.PI[i]*B_probs[i][s[0]]#self.B[i].get_prob(s[0])
                    
                for i in range(self.number_of_hidd_states):
                    for j in range(1, T):
                        sum_ = 0.0
                        for l in range(self.number_of_hidd_states):
                            sum_ += alpha[l][j-1]*self.A[l][i]
                        alpha[i][j] = B_probs[i][s[j]]*sum_#self.B[i].get_prob(s[j])*sum_
                #############################################
                #print(alpha)
                prob_old = 0.0
                for l in range(self.number_of_hidd_states):
                    prob_old += alpha[l][T-1]

                ###################Backward Procedure########
                for l in range(self.number_of_hidd_states):
                    beta[l][T-1] = 1.

                T = T-2
                for i in range(self.number_of_hidd_states):
                    for j in range(T, -1, -1):
                        for l in range(self.number_of_hidd_states):
                            beta[i][j] += beta[l][j+1]*self.A[i][l]*B_probs[l][s[j+1]]
                        #beta[i][j] = np.dot(beta[:, j+1], self.A[i, :]*B_probs[:, s[j+1]]) #self.B[:].get_prob(s[j+1]))
                ##############################################
                #print(beta)
                #########################gamma###############
                T += 2
                gamma = np.zeros((self.number_of_hidd_states, T))
                
                for t in range(T):
                    sum_ = 0.0
                    for i in range(self.number_of_hidd_states):
                        sum_ += alpha[i][t]*beta[i][t]
                    for i in range(self.number_of_hidd_states):
                        gamma[i][t] = alpha[i][t]*beta[i][t] / sum_
                #############################################
                #print(gamma)
                #######################zeta##################
                T -= 1
                zeta = np.zeros((self.number_of_hidd_states, self.number_of_hidd_states, T))
                for t in range(T):
                    sum_ = 0.0
                    for i in range(self.number_of_hidd_states):
                        for j in range(self.number_of_hidd_states):
                            sum_ += alpha[i][t]*self.A[i][j]*beta[j][t+1]*B_probs[j][s[t+1]]#self.B[j].get_prob(s[t+1])
                    zeta[i][j][t] = alpha[i][t]*self.A[i][j]*beta[j][t+1]*B_probs[j][s[t+1]] / sum_#self.B[j].get_prob(s[t+1]) / sum_
                ##############################################
                #print(zeta)
                ##################Updating PI#################
                for i in range(self.number_of_hidd_states):
                    self.PI[i] = gamma[i][0]
                self.PI = self.PI / np.sum(self.PI)
                ##############################################
                
                ####################Updating A################
                for i in range(self.number_of_hidd_states):
                    b = 0.0
                    for t in range(T):
                            b += gamma[i][t]
                    for j in range(self.number_of_hidd_states):
                        a = 0.0;
                        for t in range(T):
                            a += zeta[i][j][t]
                        #print('a'+str(a))
                        #print('b'+str(b))
                        self.A[i][j] = (a+1e-8) / b
                #print(self.A)
                for i in range(self.number_of_hidd_states):
                    self.A[i] = self.A[i] / np.sum(self.A[i])
                ##############################################
                
                #########################Updating B###########
                T += 1
                for i in range(self.number_of_hidd_states):
                    b = 0.0
                    for t in range(T):
                        b += gamma[i][t]
                    for j in range(k):
                        a = 0.0
                        for t in range(T):
                            if j == s[t]:
                                a += gamma[i][t]
                        self.B[i].update_prob(j, (a+1e-8) / b)
                for i in range(self.number_of_hidd_states):
                    self.B[i].normalize()
                ###############################################
                
                #################Forward Procedure Again##########
                for i in range(self.number_of_hidd_states):
                    alpha[i][0] = self.PI[i]*self.B[i].get_prob(s[0])
                    
                for i in range(self.number_of_hidd_states):
                    for j in range(1, T):
                        sum_ = 0.0
                        for l in range(self.number_of_hidd_states):
                            sum_ += alpha[l][j-1]*self.A[l][i]
                        alpha[i][j] = self.B[i].get_prob(s[j])*sum_
                #############################################
                
                prob_new = 0.0
                for l in range(self.number_of_hidd_states):
                    prob_new += alpha[l][T-1]
                print(str(iteration)+' '+str(prob_new - prob_old))
                if (prob_new >= prob_old) and (prob_new - prob_old <= tolerance):
                    break

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

if __name__ == '__main__':
    hmm = HMM(2, (Categorical, {'K':2}))
    hidd_seqs = [[1,1,1,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,1]]
    emi_seqs =  [[0,0,0,0,0,0,1,0,1,1,1,1,1,0,0,0,0,1,1,1,0,0,00,0,0,1]]
    #hmm.supervised_learn(hidd_seqs=hidd_seqs, emi_seqs=emi_seqs)
    hmm.unsupervised_learn(emi_seqs)
    print(hmm.PI)
    print(hmm.A)
    for i in range(2):
        for j in range(2):
            print(hmm.B[i].get_prob(j))
    #print(range(7))
    print(hidd_seqs)
    print(hmm.viterbi_predict(emi_seqs))
    print(hmm.pf_predict(emi_seqs, 500, q=hmm.A))

    
    
    
    
