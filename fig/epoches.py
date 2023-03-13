import numpy as np
import datetime
from plotly.graph_objs import *
import plotly.offline as py_offline
import random
import matplotlib.pyplot as plt


global font1

font1 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 23,
}

MSE_x = []

for i in np.arange(4,33,4):
    MSE_x.append(i)
    
MSE_5 = [9.27,9.46,9.78,9.99,10.34,10.62,11.01,11.26]
MSE_10 = [8.77,8.97,9.24,9.4,9.58,9.86,10.08,10.31]
MSE_15 = [8.88,8.98,9.18,9.47,9.78,10.14,10.5,10.91]
MSE_20 = [8.9,9.11,9.32,9.5,9.76,10.06,10.33,10.6]
MSE_25 = [8.97,9.09,9.3,9.65,10.07,10.56,11.07,11.59]
MSE_30 = [9.08,9.36,9.61,10.1,10.43,10.78,11.15,11.47]
MSE_35 = [9.27,9.58,9.88,10.21,10.59,11.02,11.44,11.88]


def showLine(x,y,label,marker):
    plt.plot(x, y,label=label,marker=marker)

import matplotlib.ticker as ticker
plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
# plt.gca().xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
 
  
font2 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size' : 16,
}
plt.xlabel('Prediction range(mins)',font2)
plt.ylabel('MSE',font2)

plt.yticks(fontproperties = 'Times New Roman', size = 16)
plt.xticks(fontproperties = 'Times New Roman', size = 16)

# 标题
# plt.title("MSE", font1)
#plt.xlim(0,30)
# plt.ylim(12.5,50)

# o s ^ v . < d * X p h | _
showLine(MSE_x,MSE_5,'5 epochs','v')
showLine(MSE_x,MSE_10,'10 epochs','v')
showLine(MSE_x,MSE_15,'15 epochs','s')  
showLine(MSE_x,MSE_20,'20 epochs','^')
showLine(MSE_x,MSE_25,'25 epochs','|')  
showLine(MSE_x,MSE_30,'30 epochs','o')
showLine(MSE_x,MSE_35,'35 epochs','*')  

# showLine(df_part1,df_part4)
plt.legend(prop={'family' : 'Times New Roman', 'size' : 12})

_x_ticks = range(4,33,4)

plt.xticks(_x_ticks, _x_ticks,rotation=0)
# plt.yticks(fontproperties = 'Times New Roman', size = 12)
# plt.xticks(fontproperties = 'Times New Roman', size = 12)
plt.legend(prop={'family' : 'Times New Roman', 'size' : 12},loc=2)

plt.show()



