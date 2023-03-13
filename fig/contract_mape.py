import numpy as np
import datetime
from plotly.graph_objs import *
import plotly.offline as py_offline
import random
import matplotlib.pyplot as plt
import matplotlib

from matplotlib.pyplot import *

# matplotlib.rc('font',family='Times New Roman')
plt.rc('font', family='Times New Roman')
# plt.rcParams["font.family"] = "Times New Roman" # change default font
# plt.rcParams['font.family'] = "sans-serif"
# plt.rcParams['font.sans-serif'] = "Helvetica"
# plt.rcParams['text.usetex'] = True
# plt.rcParams['font.family'] ="Times New Roman"
# plt.rcParams['font.sans-serif'] ="Helvetica"
# plt.rcParams["font.family"] = "Times New Roman" # change default font
# plt.rcParams['text.usetex'] = True
# plt.rcParams['mathtext.it'] = 'Times New Roman:italic'
# mathtext.rm  : serif
# mathtext.it  : serif:italic
# mathtext.bf  : serif:bold
# mathtext.fontset: custom
# plt.rc('font', family='times new roman', size=16)
# matplotlib.use("pgf")
# pgf_config = {
#     "font.family":'serif',
#     "font.size": 20,
#     "pgf.rcfonts": False,
#     "text.usetex": True,
#     "pgf.preamble": [
#         r"\usepackage{unicode-math}",
#         #r"\setmathfont{XITS Math}",
#         r"\setmainfont{Times New Roman}",
#         r"\usepackage{xeCJK}",
#         r"\xeCJKsetup{CJKmath=true}",
#         r"\setCJKmainfont{SimSun}",
#     ],
# }

# rcParams.update(pgf_config)

global font1

font1 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 23,
}

x = []

for i in np.arange(4,33,4):
    x.append(i)

HA = [64.05,64.05,64.05,64.05,64.05,64.05,64.05,64.05]
LSTM = [32.33,33.31,33.02,32.44,32.02,32.32,32.47,31.93]
ConvLSTM = [30.58,30.88,31.54,31.9,32.06,32.17,32.35,32.57]
ResNet = [27.93,27.89,28.5,29.19,29.93,30.71,31.5,32.31]
ST_3DNets = [27.27,27.22,27.83,28.43,29,29.63,30.29,30.97]

AT_CONV = [28.79,29.01,29.56,30.37,31.14,32.64,34.32,36.81]
SANN = [26.75,26.76,27.07,27.87,28.4,28.64,29.32,29.91]

MDSTN = [26.13,26.46,26.77,27.17,27.4,27.64,28.02,28.81]
MDSTN_without_time = [27.08,27.37,27.79,28.24,28.91,29.62,30.32,31.04]
MDSTN_without_poi = [28.03,28.13,29.02,29.74,30.91,31.91,33.32,34.62]


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
plt.ylabel('MAPE',font2)

_x_ticks = range(4,33,4)

plt.xticks(_x_ticks, _x_ticks,rotation=0)
plt.yticks(fontproperties = 'Times New Roman', size = 16)
plt.xticks(fontproperties = 'Times New Roman', size = 16)

# plt.title("MSE", font1)
#plt.xlim(0,30)
# plt.ylim(12.5,50)

# o s ^ v . < d * X p h | _
# showLine(x,HA,'HA','X','orange')
# showLine(x,LSTM,'LSTM','v','brown')
# showLine(x,ConvLSTM,'ConvLSTM','s','steelblue')
showLine(x,ResNet,'ST-ResNet','^','mediumpurple')
showLine(x,ST_3DNets,'ST-3DNets','s','darkblue')
showLine(x,AT_CONV,'AT-Conv','p','olive')
showLine(x,SANN,'SANN','*','lightsteelblue')
showLine(x,MDSTN,'MDSTN(ours)','o','red')
showLine(x,MDSTN_without_time,'MDSTN without time signal','h','olive')
showLine(x,MDSTN_without_poi,'MDSTN without POI signal','d','olive')

# showLine(x,ResNet,'ST-ResNet','^','mediumpurple')
# showLine(x,ST_3DNets,'ST-3DNets','|','darkblue')
# showLine(x,AT_CONV,'AT-Conv','o','red')
# showLine(x,SANN,'SANN','o','red')
# showLine(x,MDSTN,'MDSTN(ours)','o','red')
# showLine(x,MDSTN_without_time,'MDSTN without time signal','*','lightsteelblue')
# showLine(x,MDSTN_without_poi, 'MDSTN without POI signal' ,'d','olive')

# showLine(df_part1,df_part4)
plt.legend(prop={'family' : 'Times New Roman', 'size' : 12},loc=2)
plt.show()



