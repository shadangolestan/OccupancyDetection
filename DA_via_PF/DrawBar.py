import numpy as np
import matplotlib.pyplot as plt

plt.bar(x=[0.01,  0.11,  0.21], height=[0.85,  0.84,  0.83], width=0.01, color=['g',  'g',  'g'], tick_label=['25',  '50',  '75'], label='Supervised')
plt.xlabel('% of Labeled Data in Target Domain')
plt.ylabel('RMSE')
plt.ylim((.8, .85))
plt.legend()
plt.show()

plt.bar(x=[0.01,  0.11,  0.21], height=[2.49,  2.15,  2.04], width=0.01, color=['r',  'r',  'r'], tick_label=['25',  '50',  '75'], label='Semi-Supervised')
plt.xlabel('% of Labeled Data in Target Domain')
plt.ylabel('RMSE')
plt.ylim((2.0, 2.5))
plt.legend()
plt.show()

"""
plt.bar(x=[0.01, 0.02,  0.11,0.12,  0.21,0.22], height=[2.49,0.85,  2.04,0.84,  2.15,0.83], width=0.01, color=['g','r',  'g','r',  'g','r'], tick_label=['25','',  '50','',  '75',''])
plt.xlabel('% of Labeled Data in Target Domain')
plt.ylabel('RMSE')
#plt.legend()
plt.show()
"""
