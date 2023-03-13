import numpy as np
# from tensorflow.keras.models import load_model

from sklearn.metrics import mean_squared_error # 均方误差
from sklearn.metrics import mean_absolute_error # 平方绝对误差


import matplotlib.pyplot as plt
# import seaborn as sns
import numpy as np
import json

### step.1

arr_file_name = ['down','up','left','right']

def resave(label, file_name):
    with open( label +'/json/'+ file_name+'.json','r') as load_f:
        data = json.load(load_f)
        arr = np.array(data)
        print(arr.shape)
        np.save(label+"_"+file_name+".npy", arr)
        print(file_name+".npy save .npy done")
        # print(load_dict)

for ii in range(4):
    for jj in range(4):
        label = str(ii)+"_"+str(jj)
        for file_name in arr_file_name:
            resave(label, file_name)


        up_ = np.load(label + "_up.npy")
        left_ = np.load(label + "_left.npy")
        down_ = np.load(label + "_down.npy")
        right_ = np.load(label + "_right.npy")

        up_ = up_.reshape((-1,1,16,16,1))
        left_ = left_.reshape((-1,1,16,16,1))
        down_ = down_.reshape((-1,1,16,16,1))
        right_ = right_.reshape((-1,1,16,16,1))

        data = np.concatenate([up_, left_, down_, right_], axis=1)

        np.save(label+'.npy', np.minimum(data,30))


        # ### step.2 
        data = data.reshape((-1,4,16,16))

        p_time = 4
        step = 5

        len_ = len(data)-(p_time+1)*step
        print(len_)
            
        dataX = np.zeros((len_,step,data.shape[1],data.shape[2],data.shape[3]))
        dataY = np.zeros((len_,data.shape[1],data.shape[2],data.shape[3]))
        for i in range(len_):
            for j in range(step):
                dataX[i][j] = data[i+j*p_time]
            #print(data.shape,i+(step+1)*p_time)
            for k in range(data.shape[1]):
                dataY[i] = data[i+step*p_time]

        print(dataX.shape)
        print(dataY.shape)


        dataZ = np.zeros((len_,2))
        for i in range(len_):
            time = i%(60*24)  # based on day
            hour = time/60
            minute = time%60
            dataZ[i][0] = hour
            dataZ[i][1] = minute
            
        dataPOI_ = np.load('dataPOI.npy')
        dataPOI = np.zeros((len_,20,16,16))
        for i in range(len_):
            dataPOI[i] = dataPOI_

        mid_ = int(0.8*len_)

        dataX1 = dataX[0:mid_]  
        dataY1 = dataY[0:mid_]
        dataZ1 = dataZ[0:mid_]
        dataPOI1 = dataPOI[0:mid_]

        np.save(label+'DataX.npy', dataX1)
        np.save(label+'DataY.npy', dataY1)
        np.save(label+'DataZ.npy', dataZ1)
        np.save(label+'DataPOI.npy', dataPOI1)

        print(dataX1.shape)
        print(dataY1.shape)
        print(dataZ1.shape)
        print(dataPOI1.shape)

        dataTestX = dataX[mid_+1:len_]
        dataTestY = dataY[mid_+1:len_]
        dataTestZ = dataZ[mid_+1:len_]
        dataTestPOI = dataPOI[mid_+1:len_]

        print(dataTestX.shape)
        print(dataTestY.shape)
        print(dataTestZ.shape)
        print(dataTestPOI.shape)

        np.save(label+'DataTestX.npy', dataTestX)
        np.save(label+'DataTestY.npy', dataTestY)
        np.save(label+'DataTestZ.npy', dataTestZ)
        np.save(label+'DataTestPOI.npy', dataTestPOI)

# step.3

# a1 = np.array([[0,0,0,0,0],
#                [0,0,0,0,0],
#                [0,0,50,0,0],
#                [0,0,0,0,0],
#                [0,0,0,0,0]])

# a2 = np.array([[0,0,0,0,0],
#                [0,0,0,0,0],
#                [0,0,0,20,0],
#                [0,0,30,0,0],
#                [0,0,0,0,0]])

# b1 = np.array([[0,0,0,0,0],
#                [0,0,0,0,0],
#                [0,0,15,0,0],
#                [0,0,0,0,0],
#                [0,0,0,0,0]])

# b2 = np.array([[0,0,0,0,0],
#                [0,0,0,0,0],
#                [0,0,0,15,0],
#                [0,0,0,0,0],
#                [0,0,0,0,0]])

# c1 = np.array([[0,0,0,0,0],
#                [0,0,0,0,0],
#                [0,0,50,0,0],
#                [0,0,0,0,0],
#                [0,0,0,0,0]])

# c2 = np.array([[0,0,0,0,0],
#                [0,0,0,0,0],
#                [0,50,0,0,0],
#                [0,0,0,0,0],
#                [0,0,0,0,0]])

# d1 = np.array([[0,0,0,0,0],
#                [0,0,0,0,0],
#                [0,0,15,0,0],
#                [0,0,0,0,0],
#                [0,0,0,0,0]])

# d2 = np.array([[0,0,0,0,0],
#                [0,0,0,0,0],
#                [0,0,0,15,0],
#                [0,0,0,0,0],
#                [0,0,0,0,0]])

# def show(c_):
#     hmax = sns.heatmap(c_,vmax=50,square=True,robust=True, cmap='Blues',alpha = 0.8,zorder = 2,annot = True) # whole heatmap is translucent annot = True,
#     # hmax.imshow(image,
#     #       aspect = hmax.get_aspect(),
#     #       extent = hmax.get_xlim() + hmax.get_ylim(),
#     #       zorder = 1) # put the map under the heatmap
    
    
# plt.subplot(4,2,1)
# show(a1)
# plt.subplot(4,2,2)
# show(a2)


# plt.subplot(4,2,3)
# show(b1)
# plt.subplot(4,2,4)
# show(b2)


# plt.subplot(4,2,5)
# show(c1)
# plt.subplot(4,2,6)
# show(c2)

# plt.subplot(4,2,7)
# show(d1)

# plt.subplot(4,2,8)
# show(d2)

# plt.show()