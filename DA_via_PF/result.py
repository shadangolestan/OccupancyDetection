import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import sklearn.mixture as mix



a = [100.0,
100.0,
100.0,
100.0,
100.0,
100.0,
100.0,
100.0,
100.0,
100.0,
100.0,
100.0,
95.641025641,
88.8095238095,
82.8888888889,
77.7083333333,
79.0196078431,
80.1851851852,
81.2280701754,
82.1666666667]

time = []
for i in range(1, 21):
    time.append(i)

plt.title('Accuracy for DA using 24 hrs data')
plt.xlabel('Prediction Time (x 30 mins)')
plt.ylabel('Accuracy(%)')

plt.plot(time, a)
plt.show()
