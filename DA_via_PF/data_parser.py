import numpy as np
import matplotlib.pyplot as plt


co2_threshold = 800.0

def co2_gen(file_name):

	# Collecting data from dataset
	raw_data = np.loadtxt(fname=file_name, delimiter=',', skiprows=1, usecols=(2,3,4,5,6,7))
	
	x_occ = []
	y_occ = []

	x_unocc = []
	y_unocc = []

	j=0
	for i in raw_data:
		if i[5] >= 1.0:
			y_occ.append(i[3])
			x_occ.append(j)
		else:
			y_unocc.append(i[3])
			x_unocc.append(j)
		j+=1


	plt.plot(x_occ,y_occ,'r.',x_unocc,y_unocc,'b.')
	plt.show()
	

	# CO2 threshold value--equal or greater than this value denotes one emission/outcome state
	# and other value denotes another state
	#co2_threshold = 450.0

	# Converting CO2 raw data to outcome vs hidden state (occupied or not) data
	co2_data = np.zeros((raw_data.shape[0],2), dtype=int)

	j=0
	for i in raw_data:
		co2_data[j][1] = int(i[5])
		if i[3] >= co2_threshold:
			co2_data[j][0] = 1
		j+=1
	
	return co2_data

def co2_gen_2():
	raw_data_co2 = np.loadtxt(fname='co2_data_room_1.csv', delimiter=',', skiprows=1, usecols=(4))
	raw_data_occ = np.loadtxt(fname='occ_data_room_1.csv', delimiter=',', skiprows=1, usecols=(4))

	co2_data = np.zeros((raw_data_co2.shape[0],2), dtype=int)
	#co2_threshold = 450.0

	for i in range(co2_data.shape[0]):
		if raw_data_occ[i] > 0.0:
			co2_data[i][1] = 1
		if raw_data_co2[i] >= co2_threshold:
			co2_data[i][0] = 1
	return co2_data

def seq_gen(data,seq_count=10,max_seq_len=100,full=False):
	seqs = []
	if full==True:
		seqs.append(data)
		return seqs
	while(True):
		if(seq_count==0):
			return seqs
		i = np.random.randint(data.shape[0]-1)
		#j = i+np.random.randint(max_seq_len)+1
		j = i+max_seq_len+1
		if j > data.shape[0]:
			j = data.shape[0]
		seqs.append(data[i:j, :])
		seq_count-=1
		

class HMM:
	def __init__(self,NoHS,NoE):
		# NoHS = Number of Hidden States
		# NoE = Number of Emissions
		self.NoHS = NoHS
		self.NoE = NoE

		# Counting parameters
		self.Acount = np.ones((self.NoHS,self.NoHS),dtype=int)
		self.Bcount = np.ones((self.NoHS,self.NoE),dtype=int)
		self.PIcount = np.ones(self.NoHS,dtype=int)

		# Probability parameters
		self.A = np.zeros((self.NoHS,self.NoHS),dtype=float)
		self.B = np.zeros((self.NoHS,self.NoE),dtype=float)
		self.PI = np.zeros(self.NoHS,dtype=float)

	def prior(self,Acount,Bcount,PIcount):
		for i in range(self.NoHS):
			self.PIcount[i] = PIcount[i]
			for j in range(self.NoHS):
				self.Acount[i][j] = Acount[i][j]
			for j in range(self.NoE):
				self.Bcount[i][j] = Bcount[i][j]
		
	
	def learn(self,train_seqs):
		for s in train_seqs:
			self.PIcount[s[0][1]]+=1
			for i in range(s.shape[0]-1):
				self.Acount[s[i][1]][s[i+1][1]]+=1
			for i in range(s.shape[0]):
				self.Bcount[s[i][1]][s[i][0]]+=1

		sum = np.sum(self.PIcount)
		for i in range(self.NoHS):
			self.PI[i]=self.PIcount[i]/(1.0*sum)

		for i in range(self.NoHS):
			sum = np.sum(self.Acount[i])
			for j in range(self.NoHS):
				self.A[i][j]=self.Acount[i][j]/(1.0*sum)
		
		for i in range(self.NoHS):
			sum = np.sum(self.Bcount[i])
			for j in range(self.NoE):
				self.B[i][j]=self.Bcount[i][j]/(1.0*sum)
		
class DA_PaFi:
	def __init__(self,hmm_source,hmm_target):
		self.hmm_src = hmm_source
		self.hmm_trgt = hmm_target

	def learn(self,fut_emi_seq,NoP,DA=True):
		# NoP = Number of Particles
		# fut_emi_seq = future emission sequence	
		self.fut_hid_seq = np.zeros(fut_emi_seq.shape[0],dtype=int)
		x = np.random.choice([0,1],size=NoP,p=self.hmm_src.PI)
		w = np.ones(NoP)

		for i in range(fut_emi_seq.shape[0]):
			for j in range(NoP):
				p = x[j]
				if DA==True:
					x[j] = np.random.choice([0,1],p=self.hmm_src.A[p])
					w[j] = self.hmm_trgt.B[x[j]][fut_emi_seq[i]]*(self.hmm_trgt.A[p][x[j]]/self.hmm_src.A[p][x[j]])
				else:
					x[j] = np.random.choice([0,1],p=self.hmm_trgt.A[p])
					w[j] = self.hmm_trgt.B[x[j]][fut_emi_seq[i]]*(self.hmm_trgt.A[p][x[j]]/self.hmm_trgt.A[p][x[j]])
				w[j]/=NoP

			w = w/np.sum(w)
			r = np.random.choice(x,size=NoP,p=w)
			avg = np.sum(r)/(1.0*NoP)
			if avg >= 0.5:
				self.fut_hid_seq[i] = 1
			else:
				self.fut_hid_seq[i] = 0



avg_accu_DA = 0.0;
avg_accu = 0.0
Round = 3;
for R in range(Round):
	trgt_data = seq_gen(co2_gen('occupancy_data/datatest.txt'), seq_count=100, max_seq_len=100)
	#trgt_data = seq_gen(co2_gen_2(), seq_count=10, max_seq_len=20)
	hmm_trgt = HMM(2,2)
	print('ok')
	hmm_trgt.learn(trgt_data)
	print('done')

	src_data = seq_gen(co2_gen('occupancy_data/datatraining.txt'), seq_count=1000, max_seq_len=1000)
	hmm_src = HMM(2,2)
	#hmm_src.learn(src_data)

	test_data = seq_gen(co2_gen('occupancy_data/datatest.txt'), seq_count=1, max_seq_len=1000)
	#test_data = seq_gen(co2_gen_2(), seq_count=1, max_seq_len=1000)
	# Testing 
	emi_seq = test_data[0][:,0]
	hid_seq = test_data[0][:,1]

	PaFi = DA_PaFi(hmm_source=hmm_src,hmm_target=hmm_trgt)

	for case in range(2):
		if case==0:
			hmm_src.prior(Acount=hmm_trgt.Acount,Bcount=hmm_trgt.Bcount,PIcount=hmm_trgt.PIcount)
			#hmm_trgt.prior(Acount=hmm_src.Acount,Bcount=hmm_src.Bcount,PIcount=hmm_src.PIcount)
		print('ok')
		hmm_src.learn(src_data)
		print('done')
		#hmm_trgt.learn(trgt_data)
		
		if case==0:
			PaFi.learn(fut_emi_seq=emi_seq,NoP=200)
		else:
			PaFi.learn(fut_emi_seq=emi_seq,NoP=200,DA=False)

		sum = 0.0
		for i in range(emi_seq.shape[0]):
			if hid_seq[i] == PaFi.fut_hid_seq[i]:
				sum+=1.0
		if case==0:
			avg_accu_DA+=(sum/emi_seq.shape[0])*100.0
		else:
			avg_accu+=(sum/emi_seq.shape[0])*100.0
	print(str(R)+' Completed')

print('Avg Accuracy with DA: '+str(avg_accu_DA/Round))
#print('Avg Accuracy without DA: '+str(avg_accu/Round))

