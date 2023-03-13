# from sklearn.metrics import mean_squared_error # 均方误差
# from sklearn.metrics import mean_absolute_error # 平方绝对误差
# import numpy as np
# import matplotlib.pyplot as plt
# import time
# import seaborn as sns
import numpy as np
# from tensorflow.keras.models import load_model
# from pyheatmap.heatmap import HeatMap
# from PIL import Image
#
# import cv2

# 开始绘制热度图
global image
global dataX

# image =cv2.imread('background.png',0)

dataX = np.load("./data/4dataX.npy")
dataX = dataX.reshape((-1,5,16,16))

def getIndex(index,dataX):
    return dataX[index:index+1]


# ============= >>>>>>>>>>>>>> periodicity
import numpy as np
import datetime
from plotly.graph_objs import *
import plotly.offline as py_offline
import random
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator

    

x_ = []
u = [] 
l = []
d = []
r = []
s = []
a = []
# df_part2 = [0.18,0.2,0.21,0.22,0.24,0.23,0.26,0.27,0.3]
# df_part3 = [0.17,0.18,0.2,0.2,0.21,0.21,0.21,0.22,0.23]
# df_part4 = [0.29,0.38,0.52,0.72,0.82,0.94,1.1,1.16,1.2]


# for i in range(0,1440*7,30): #
#     u_ = dataX[i][0][2][8]+dataX[i+1][0][2][8]+dataX[i+2][0][2][8]+dataX[i+3][0][2][8]+dataX[i+4][0][2][8]
#     l_ = dataX[i][1][2][8]+dataX[i+1][1][2][8]+dataX[i+2][1][2][8]+dataX[i+3][1][2][8]+dataX[i+4][1][2][8]
#     d_ = dataX[i][2][2][8]+dataX[i+1][2][2][8]+dataX[i+2][2][2][8]+dataX[i+3][2][2][8]+dataX[i+4][2][2][8]
#     r_ = dataX[i][3][2][8]+dataX[i+1][3][2][8]+dataX[i+2][3][2][8]+dataX[i+3][3][2][8]+dataX[i+4][3][2][8]
#     s_ = dataX[i][4][2][8]+dataX[i+1][4][2][8]+dataX[i+2][4][2][8]+dataX[i+3][4][2][8]+dataX[i+4][4][2][8]
#     a_ = u_+l_+d_+r_+s_
#     u.append(u_/5)
#     l.append(l_/5)
#     d.append(d_/5)
#     r.append(r_/5)
#     s.append(s_/5)
#     a.append(a_/5)
    
line = np.zeros((7*1440))
for i in range(0,1440*7,30): #
    u.append(dataX[i][0][2][8])
    l.append(dataX[i][1][2][8])
    d.append(dataX[i][2][2][8])
    r.append(dataX[i][3][2][8])
    s.append(dataX[i][4][2][8])
    a.append(dataX[i][0][2][8]+dataX[i][1][2][8]+dataX[i][2][2][8]+dataX[i][3][2][8]+dataX[i][4][2][8])

for i in np.arange(48*7): # 7天每天48个时间段，每个时间段半小时
    x_.append(i)

def showLine(x,y,label,marker='o'):
    plt.plot(x, y,label=label) #,label=label ,marker=marker
    
    #设置横坐标需要修改的形式，并注意与数据的横坐标一一对应
    _x_ticks = ["","",""]
    for i in x:
        if(i%6 == 0): #三小时留下一个横坐标
            _x_ticks.append("{}:00".format(int(i%48/2)))
        # else:
        #     _x_ticks.append("")

    plt.xticks(x[:], _x_ticks[:],rotation=60)
    
    x_major_locator=MultipleLocator(6)
    #把x轴的刻度间隔设置为1，并存在变量里
    ax=plt.gca()
    #ax为两条坐标轴的实例
    ax.xaxis.set_major_locator(x_major_locator)
    # plt.ylabel('number of electrical bicycle')
    
    plt.xlabel('Time range (2019.9.11 - 2019.9.18)',font2)
    plt.ylabel('Number of electrical bicycle',font2)

    plt.yticks(fontproperties = 'Times New Roman', size = 12)
    plt.xticks(fontproperties = 'Times New Roman', size = 12)
    
import matplotlib.ticker as ticker
plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
# plt.gca().xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
 
  
font2 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size' : 12,
}

plt.rc('legend', fontsize=10) 
plt.subplot(2,1,1)
# plt.title('',fontsize=16,fontdict=font2)
showLine(x_,u,'Heading upward','v')
showLine(x_,l,'Heading leftward','^')   
showLine(x_,d,'Heading downward','^')   
showLine(x_,r,'Heading rightward','^')   
showLine(x_,a,'Total states','^')   
plt.legend(prop={'family' : 'Times New Roman', 'size' : 10})


plt.show()

