import numpy as np
from tensorflow.keras.models import load_model

from sklearn.metrics import mean_squared_error # 均方误差
from sklearn.metrics import mean_absolute_error # 平方绝对误差


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json

a1 = np.array([[0,0,0,0,0],
               [0,0,40,0,0],
               [0,30,50,40,0],
               [0,0,20,0,0],
               [0,0,0,0,0]])

a2 = np.array([[0,0,0,0,0],
               [0,0,10,0,0],
               [0,5,10,20,0],
               [0,0,5,0,0],
               [0,0,0,0,0]])

b1 = np.array([[0,0,0,0,0],
               [0,0,10,0,0],
               [0,5,15,10,0],
               [0,0,5,0,0],
               [0,0,0,0,0]])

b2 = np.array([[0,0,0,0,0],
               [0,0,10,0,0],
               [0,10,15,10,0],
               [0,0,5,0,0],
               [0,0,0,0,0]])

c1 = np.array([[0,0,0,0,0],
               [0,0,5,0,0],
               [0,15,5,5,0],
               [0,0,0,0,0],
               [0,0,0,0,0]])

c2 = np.array([[0,0,0,0,0],
               [0,0,5,0,0],
               [0,5,5,5,0],
               [0,0,0,0,0],
               [0,0,0,0,0]])

d1 = np.array([[0,0,0,0,0],
               [0,0,0,0,0],
               [0,0,15,0,0],
               [0,0,0,0,0],
               [0,0,0,0,0]])

d2 = np.array([[0,0,0,0,0],
               [0,0,0,0,0],
               [0,0,0,15,0],
               [0,0,0,0,0],
               [0,0,0,0,0]])

def show(c_):
    hmax = sns.heatmap(c_,vmax=30,square=True,robust=False, cmap='rainbow',alpha = 0.8,zorder = 2,annot = True, center=0) # whole heatmap is translucent annot = True,
    # hmax.imshow(image,
    #       aspect = hmax.get_aspect(),
    #       extent = hmax.get_xlim() + hmax.get_ylim(),
    #       zorder = 1) # put the map under the heatmap
    
    
plt.subplot(4,2,1)
show(a1)
plt.subplot(4,2,2)
show(a2)


plt.subplot(4,2,3)
show(b1)
plt.subplot(4,2,4)
show(b2)


plt.subplot(4,2,5)
show(c1)
plt.subplot(4,2,6)
show(c2)

plt.subplot(4,2,7)
show(d1)

plt.subplot(4,2,8)
show(d2)

plt.show()