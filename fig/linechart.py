import numpy as np
import datetime
from plotly.graph_objs import *
import plotly.offline as py_offline
import random
import matplotlib.pyplot as plt



# MSE_x_2m = []
# for i in np.arange(2,33,2):
#     MSE_x_2m.append(i)
# MSE_x_3m = []
# for i in np.arange(3,31,3):
#     MSE_x_3m.append(i)
# MSE_x_4m = []
# for i in np.arange(4,33,4):
#     MSE_x_4m.append(i)
# MSE_x_5m = []
# for i in np.arange(5,31,5):
#     MSE_x_5m.append(i)
# MSE_x_6m = []
# for i in np.arange(6,31,6):
#     MSE_x_6m.append(i)
# MSE_x_7m = []
# for i in np.arange(7,29,7):
#     MSE_x_7m.append(i)
# MSE_x_8m = []
# for i in np.arange(8,33,8):
#     MSE_x_8m.append(i)
    
# MSE_y_2m = [12.96,13.72,14.57,15.44,16.38,17.36,18.4,19.55,20.86,22.45,24.39,26.77,29.75,33.57,38.59,45.21]
# MSE_y_3m = [13.46,14.11,14.94,15.8,16.67,17.6,18.69,20.1,21.93,24.36]
# MSE_y_4m = [13.92,14.83,16.22,18.11,20.67,24.05,28.34,33.69]
# MSE_y_5m = [13.84,14.48,15.42,16.54,17.85,19.52]
# MSE_y_6m = [13.46,13.98,14.88,15.99,17.45]
# MSE_y_7m = [13.88,15.07,16.52,18.06]
# MSE_y_8m = [15.51,17.17,21.62,27.91]



# def showLine(x,y,label,marker='o'):
#     plt.plot(x, y,label=label,marker=marker)
        
# # 标题
# plt.title("MSE")
# #plt.xlim(0,30)
# plt.ylim(12.5,35)

# # o s ^ v . < d * X p h | _
# showLine(MSE_x_2m,MSE_y_2m,'2-min interval','v')
# showLine(MSE_x_3m,MSE_y_3m,'3-min interval','s')  
# showLine(MSE_x_4m,MSE_y_4m,'4-min interval','^')
# showLine(MSE_x_5m,MSE_y_5m,'5-min interval','|')  
# showLine(MSE_x_6m,MSE_y_6m,'6-min interval(the best)','o')
# showLine(MSE_x_7m,MSE_y_7m,'7-min interval','*')  
# showLine(MSE_x_8m,MSE_y_8m,'8-min interval','d')   

# # showLine(df_part1,df_part4)
# plt.legend()
# plt.show()
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
MSE_ResNet = [9.44,10.05,10.63,11.2,11.76,12.28,12.87,13.48]
MSE_ST_3DNets = [9.27,9.69,10.28,10.82,11.29,11.78,12.35,12.93]
MSE_PSTRN = [8.77,8.97,9.24,9.4,9.58,9.86,10.08,10.31]
MSE_PSTRN_without_time = [9.06,9.17,9.33,9.55,9.77,10.05,10.36,10.7]
MSE_PSTRN_without_poi = [9.78,9.79,9.89,10.59,10.69,12.98,15.33,19.34]


def showLine(x,y,label,marker='o'):
    plt.plot(x, y,label=label,marker=marker)

import matplotlib.ticker as ticker
plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
# plt.gca().xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
 
  
font2 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size' : 12,
}
plt.xlabel('Prediction range(mins)',font2)
plt.ylabel('MSE',font2)

plt.yticks(fontproperties = 'Times New Roman', size = 12)
plt.xticks(fontproperties = 'Times New Roman', size = 12)

# 标题
# plt.title("MSE", font1)
#plt.xlim(0,30)
# plt.ylim(12.5,50)

# o s ^ v . < d * X p h | _
showLine(MSE_x,MSE_HA,'HA','v')
showLine(MSE_x,MSE_LSTM,'LSTM','v')
showLine(MSE_x,MSE_ConvLSTM,'ConvLSTM','s')  
showLine(MSE_x,MSE_ResNet,'ST-ResNet','^')
showLine(MSE_x,MSE_ST_3DNets,'ST-3DNets','|')  
showLine(MSE_x,MSE_PSTRN,'PSTRN(ours)','o')
showLine(MSE_x,MSE_PSTRN_without_time,'PSTRN without time','*')  
showLine(MSE_x,MSE_PSTRN_without_poi,'PSTRN without poi','d')   

# showLine(df_part1,df_part4)
plt.legend(prop={'family' : 'Times New Roman', 'size' : 10})
plt.show()



