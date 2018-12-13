import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import sklearn.mixture as mix

#########Conditional Variables###############
room = 1 # 1 to 4
occ = 1 # -1 to 1
weekday = 3 # -1 to 6
holiday = -1 # -1 to 2
start_hour = 0 # 00 to 23
end_hour = 24 # 01 to 24 (end hour is exclusive)
# Note: -1 = without condition
############################################

#################Some Parameters############
GMM_components = 25
Name = 'co2' # co2 or vav
Plot_GMM = False
Plot_Gamma = True
############################################

#################Index Variables############
DayId = 0
Hour = 1
Weekday = 2
Holiday = 3
EMI = 4
OCC = 5
############################################

occ_data = np.loadtxt(fname='data/occ_data_room_'+str(room)+'.csv', delimiter=',', dtype=object, usecols=(4), skiprows=1)
emi_data = np.loadtxt(fname='data/'+Name+'_data_room_'+str(room)+'.csv', delimiter=',', dtype=object, usecols=(0,1,2,3,4), skiprows=1)
room_data = np.zeros((emi_data.shape[0], emi_data.shape[1]+1))
for i in range(room_data.shape[0]):
    room_data[i][DayId] = float(emi_data[i][0])
    room_data[i][Hour] = float(emi_data[i][1].split(':')[0])
    room_data[i][Weekday] = float(emi_data[i][2])
    room_data[i][Holiday] = float(emi_data[i][3])
    room_data[i][EMI] = float(emi_data[i][4])
    if float(occ_data[i][0]) > 0.0:
        room_data[i][OCC] = 1.0

if __name__ == '__main__':

    emi_hist = []
    for i in range(room_data.shape[0]):
        if occ > -1 and occ != int(room_data[i][OCC]):
            continue
        if weekday > -1 and weekday != int(room_data[i][Weekday]):
        #if weekday > -1 and int(room_data[i][Weekday]) >= 5:
            continue
        if holiday > -1 and holiday != int(room_data[i][Holiday]):
            continue
        if int(room_data[i][Hour]) < start_hour or int(room_data[i][Hour]) >= end_hour:
            continue
        emi_hist.append(room_data[i][EMI])

    #print(room_data
    #room_1_emi_hist_density, room_1_emi_hist_edges = np.histogram(a=room_1_co2, bins='auto', density=True)
    #print(room_1_emi_hist_density)
    if Plot_Gamma == True:
        fit_alpha, fit_loc, fit_beta = stats.gamma.fit(emi_hist)
    if Plot_GMM == True:
        GMM = mix.GaussianMixture(n_components=GMM_components)
        x = np.asarray(emi_hist).reshape(-1,1)
        GMM.fit(x)

        GMM_pdf = []

        for i in range(len(emi_hist)):
            sum = 0.0
            for j in range(GMM.weights_.shape[0]):
                sum+=GMM.weights_[j]*stats.norm.pdf(emi_hist[i], loc=GMM.means_[j][0], scale=GMM.covariances_[j][0][0])
            GMM_pdf.append(sum)

    #print(GMM_pdf)
    plt.hist(x=emi_hist, bins='auto', histtype='step', density=True, label='Histogram')
    if Plot_GMM == True:
        plt.plot(emi_hist,GMM_pdf,'g.', label=('GMM-'+str(GMM.weights_.shape[0])))
    if Plot_Gamma == True:
        plt.plot(emi_hist, stats.gamma.pdf(x=emi_hist, a=fit_alpha, loc=fit_loc, scale=fit_beta), 'r.', label='Gamma')

    plt.title('Probability Density of '+Name+' for Some 14 Consecutive Days')
    yLabel = 'P('+Name+'|'
    if occ > -1:
        yLabel+=('occ='+str(occ))
    if weekday > -1:
        yLabel+=(',weekday='+str(weekday))
    if holiday > -1:
        yLabel+=(',holiday='+str(holiday))
    yLabel+=')'
    yLabel=yLabel.replace('|)',')')
    yLabel=yLabel.replace('|,','|')

    plt.xlabel(Name+' values for room_'+str(room)+' (From hour: '+str(start_hour)+' to '+str(end_hour)+')')
    plt.ylabel(yLabel)

    plt.legend()
    plt.show()
