import numpy as np
import datetime
from plotly.graph_objs import *
import plotly.offline as py_offline
import random
import matplotlib.pyplot as plt



MSE_x_2m = []
for i in np.arange(2,33,2):
    MSE_x_2m.append(i)
MSE_x_3m = []
for i in np.arange(3,33,3):
    MSE_x_3m.append(i)
MSE_x_4m = []
for i in np.arange(4,33,4):
    MSE_x_4m.append(i)
MSE_x_5m = []
for i in np.arange(5,33,5):
    MSE_x_5m.append(i)
MSE_x_6m = []
for i in np.arange(6,33,6):
    MSE_x_6m.append(i)
MSE_x_7m = []
for i in np.arange(7,33,7):
    MSE_x_7m.append(i)
MSE_x_8m = []
for i in np.arange(8,33,8):
    MSE_x_8m.append(i)
# MSE_x_9m = []
# for i in np.arange(9,61,9):
#     MSE_x_9m.append(i)

####  60 mins
# MSE_y_2m = [8.48,8.99,9.29,9.51,9.78,10.15,10.49,10.81,11.14,11.5,11.87,12.23,12.6,13,13.4,13.81,14.23,14.66,15.12,15.58,16.06,16.55,17.06,17.59,18.13,18.69,19.26,19.85,20.45,21.06]
# MSE_y_3m = [8.6,8.84,9.08,9.23,9.43,9.69,9.9,10.1,10.33,10.54,10.76,10.98,11.2,11.42,11.65,11.88,12.12,12.36,12.88,13.15]
# MSE_y_4m = [8.77,8.97,9.24,9.4,9.58,9.86,10.08,10.31,10.55,10.8,11.05,11.34,11.6,11.91,12.2]
# MSE_y_5m = [9.08,9.36,9.59,9.78,10.1,10.43,10.73,10.99,11.27,11.54,11.82,12.08]
# MSE_y_6m = [9.36,9.7,10.24,10.66,11.2,11.68,12.19,12.67,13.16,13.63]
# MSE_y_7m = [9.36,10.2,10.76,11.75,12.85,13.89,14.9,15.89]
# MSE_y_8m = [9.79,11.12,12.62,14.11,15.33,16.45,17.44]
# # MSE_y_9m = [9.48,10.4,11.69,13.18,14.81,16.66]


MSE_y_2m = [8.48,8.99,9.29,9.51,9.78,10.15,10.49,10.81,11.14,11.5,11.87,12.23,12.6,13,13.4,13.81]
MSE_y_3m = [8.6,8.84,9.08,9.23,9.43,9.69,9.9,10.1,10.33,10.54]
MSE_y_4m = [8.77,8.97,9.24,9.4,9.58,9.86,10.08,10.31]
MSE_y_5m = [9.08,9.36,9.59,9.78,10.1,10.43]
MSE_y_6m = [9.36,9.7,10.24,10.66,11.2]
MSE_y_7m = [9.36,10.2,10.76,11.75]
MSE_y_8m = [9.79,11.12,12.62,14.11]

def showLine(x,y,label,marker='o'):
    plt.plot(x, y,label=label,marker=marker)
        
# 标题
# plt.title("MSE")
# #plt.xlim(0,30)
plt.ylim(8,15)
import matplotlib.ticker as ticker
plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
# plt.gca().xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
 
  
font2 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size' : 16,
}
plt.xlabel('Prediction range(mins)',font2)
plt.ylabel('MSE',font2)

global font1

font1 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 23,
}

plt.xlabel('Prediction range(mins)',font2)
plt.ylabel('MSE',font2)

# plt.yticks(fontproperties = 'Times New Roman', size = 16)
# plt.xticks(fontproperties = 'Times New Roman', size = 16)

import matplotlib.ticker as ticker
plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))

# o s ^ v . < d * X p h | _
showLine(MSE_x_2m,MSE_y_2m,'2-min interval','v')
showLine(MSE_x_3m,MSE_y_3m,'3-min interval','|')  
showLine(MSE_x_4m,MSE_y_4m,'4-min interval','o')
showLine(MSE_x_5m,MSE_y_5m,'5-min interval','s')  
showLine(MSE_x_6m,MSE_y_6m,'6-min interval','^')
showLine(MSE_x_7m,MSE_y_7m,'7-min interval','*')  
showLine(MSE_x_8m,MSE_y_8m,'8-min interval','d')   
# showLine(MSE_x_9m,MSE_y_9m,'9-min interval','_')   

_x_ = range(1,33,1)
_x_ticks = range(4,33,4)

plt.xticks(_x_ticks, _x_ticks,rotation=0)
plt.yticks(fontproperties = 'Times New Roman', size = 16)
plt.xticks(fontproperties = 'Times New Roman', size = 16)

# showLine(df_part1,df_part4)
plt.legend(prop={'family' : 'Times New Roman', 'size' : 13})
plt.show()



