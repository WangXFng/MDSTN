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
    
MSE_HA = [22.12,22.12,22.12,22.12,22.12,22.12,22.12,22.12]
MSE_LSTM = [24.91,25.79,26.75,28.06,29.57,30.01,30.94,31.67]
MSE_ConvLSTM = [13.04,14.41,16.16,17.97,19.81,21.69,23.7,25.88]
MSE_ResNet = [9.48,10.15,10.78,11.43,12.08,12.71,13.38,14.07]
MSE_ST_3DNets = [9.27,9.69,10.28,10.82,11.29,11.78,12.35,12.93]  # 1
MSE_AT_Conv = [9.55,10.05,10.78,11.52,12.29,13.38,14.35,16.03]
MSE_SANN = [9.23,9.59,10.08,10.61,10.96,11.42,12.02,12.53]  # 2
MSE_MDSTN = [8.77,8.97,9.24,9.4,9.58,9.86,10.08,10.31]
MSE_MDSTN_without_time = [9.06,9.17,9.33,9.55,9.77,10.05,10.36,10.7]
MSE_MDSTN_without_poi = [9.11,9.26,9.4,9.75,10.15,10.64,11.19,11.79]


def showLine(x,y,label,marker,color):
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
# plt.ylim(8,25)

# o s ^ v . < d * X p h | _
# showLine(MSE_x,MSE_HA,'HA','X','orange')
# showLine(MSE_x,MSE_LSTM,'LSTM','v','brown')
# showLine(MSE_x,MSE_ConvLSTM,'ConvLSTM','s','steelblue')  
showLine(MSE_x,MSE_ResNet,'ST-ResNet','^','mediumpurple')
showLine(MSE_x,MSE_ST_3DNets,'ST-3DNets','s','darkblue')
showLine(MSE_x,MSE_AT_Conv,'AT-Conv','p','olive')
showLine(MSE_x,MSE_SANN,'SANN','*','lightsteelblue')
showLine(MSE_x,MSE_MDSTN,'MDSTN(ours)','o','red')
showLine(MSE_x,MSE_MDSTN_without_time,'MDSTN without time signal','h','olive')
showLine(MSE_x,MSE_MDSTN_without_poi,'MDSTN without POI signal','d','olive')

# showLine(df_part1,df_part4)
plt.legend(prop={'family' : 'Times New Roman', 'size' : 10},loc=2)


_x_ticks = range(4,33,4)

plt.xticks(_x_ticks, _x_ticks,rotation=0)
# plt.yticks(fontproperties = 'Times New Roman', size = 12)
# plt.xticks(fontproperties = 'Times New Roman', size = 12)
plt.legend(prop={'family' : 'Times New Roman', 'size' : 12},loc=2)
    
plt.show()



