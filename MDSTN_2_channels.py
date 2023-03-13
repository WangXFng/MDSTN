import os

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# import tensorflow as tf
import keras.backend.tensorflow_backend as KTF

# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ['KERAS_BACKEND'] = 'tensorflow'
import tensorflow as tf

config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
config.gpu_options.per_process_gpu_memory_fraction = 0.5
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv2D, Conv3D, Dropout, Flatten, BatchNormalization, concatenate, \
    Add, Reshape, LSTM, Lambda, AveragePooling3D
import numpy as np
from tensorflow.keras import Sequential

# import matplotlib.pyplot as plt
import math


def FormatTime(x):
    x = Reshape([2])(x)
    x = Dense(16 * 16)(x)
    return Reshape([1, 16, 16, 1])(x)


def fusion(x, z, poi):  # (-1, 5, 16, 16) (-1, 1, 16, 16,1) (-1, 1, 16, 16,1)

    x_list = tf.unstack(x, axis=1, name="stack")  # (-1, 5, 16, 16) -> 5 * (-1, 16, 16)
    list_ = []
    for i in range(len(x_list)):
        x_n = Reshape([1, 16, 16, 1])(x_list[i])  # (-1, 16, 16) -> (-1, 1, 16, 16, 1)

        x_n = concatenate([x_n, z, poi], axis=1)  # (-1, 1, 16, 16, 1) -> (-1, 3, 16, 16, 1)

        # for j in range(8):
        #     x_n_ = Conv3D(1, (3, 3, 3), activation='relu', padding='same')(x_n)
        #     x_n = Add()([x_n,x_n_])

        x_n = Conv3D(1, (3, 1, 1), activation='relu', padding='Valid')(x_n)  # (-1, 3, 16, 16, 1) -> (-1, 1, 16, 16, 1)

        # x_n = Add()([x_list[i], z, poi]) # ([Lambda(lambda x:x*0.6)(x_list[i]),Lambda(lambda x:x*0.2)(z),Lambda(lambda x:x*0.2)(poi)]) # ([x_list[i], z, poi])
        list_.append(x_n)  # 5 * (-1, 16, 16, 1)

    x = concatenate(list_, axis=1)  # (-1, 5, 16, 16, 1)

    # x = Reshape([5,16,16])(x)

    return x


def Res3DNet(x, z, poi):
    # x = Reshape([5*16*16])(x)
    x = fusion(x, z*0.1, poi*0.1)  # (-1, 5, 16, 16) (-1, 1, 16, 16,1) (-1, 1, 16, 16,1)

    # x = Reshape([5, 16, 16, 1])(x)

    for i in range(8):
        x_ = Conv3D(16, (3, 3, 3), activation='relu', padding='same')(x)

        x_ = BatchNormalization(epsilon=1e-6, weights=None)(x_)
        # x_ = Dropout(0.5)(x_)

        x = Add()([x, x_])

    x = Conv3D(1, (5, 1, 1), activation='relu', padding='VALID')(x)

    x = Reshape([1, 16, 16])(x)
    return x


def self_attention(x):
    v = AveragePooling3D(pool_size=(1, 16, 16), strides=None, padding='valid')(x)
    v = Dense(20)(v)
    return Lambda(lambda x: x[0] * x[1])([x, v])


# x_ = np.load('/data/wxf/code/N2D/data2c/4dataX.npy')
# y_ = np.load('/data/wxf/code/N2D/data2c/4dataY.npy')
# x_t = np.load('/data/wxf/code/N2D/data2c/4dataTestX.npy')
# y_t = np.load('/data/wxf/code/N2D/data2c/4dataTestY.npy')

# (-1, 5, 16, 16, 1)
dataX = np.load("/data/wxf/code/N2D/data2c/dataX.npy")
# (-1, 16, 16, 1)
dataY = np.load("/data/wxf/code/N2D/data2c/dataY.npy")

# (2)
dataZ = np.load("/data/wxf/code/N2D/data2c/dataZ.npy")
# (20, 16, 16)
dataPOI = np.load("/data/wxf/code/N2D/data2c/newDataPOI.npy")

# # print(dataPOI.shape)
# newDataPOI = np.zeros((len(dataZ), 13, 16, 16))
# for i in range(len(dataZ)):
#     newDataPOI[i] = dataPOI
# print(newDataPOI.shape)
# np.save("/data/wxf/code/N2D/data2c/newDataPOI.npy", newDataPOI)

# dataPOI = np.reshape(dataPOI, (-1, 13, 16, 16))

shapeX = (5, 2, 16, 16)
inputX = Input(shape=shapeX, name="inputX")
inputList = tf.unstack(inputX, axis=2, name="stack")  # (-1, 5, 4, 16, 16, 1) -> 4 * (-1, 5, 16, 16, 1)

shapeZ = (2)
inputZ = Input(shape=shapeZ, name='inputZ')  # (2)
z = FormatTime(inputZ)  # （2）-> (16,16)

shapePOI = (13, 16, 16)
inputPOI = Input(shape=shapePOI, name='inputPOI')
poi = Reshape([13, 16, 16, 1])(inputPOI)

poi = self_attention(poi)
poi = Conv3D(1, (13, 1, 1), activation='relu', padding='VALID')(poi)  # -> (-1, 1, 16, 16, 1)

cnnU = Res3DNet(inputList[0], z, poi)  # (-1, 5, 16, 16) -> (-1, 1, 16, 16)
cnnL = Res3DNet(inputList[1], z, poi)
# cnnD = Res3DNet(inputList[2], z, poi)
# cnnR = Res3DNet(inputList[3], z, poi)

cnnout = concatenate([cnnU, cnnL], axis=1)  # (-1, 1, 16, 16) -> (-1, 4, 16, 16)

model = Model(inputs=
{
    'inputX': inputX,
    'inputZ': inputZ,
    'inputPOI': inputPOI
}, outputs=cnnout)

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
model.summary()

# print(dataX.shape, dataY.shape)

history = model.fit(
    {
        'inputX': dataX,
        'inputZ': dataZ,
        'inputPOI': dataPOI
    },
    dataY, batch_size=50, epochs=10, validation_split=0, verbose=1)


#
# model.save_weights('model_weights.h5')
# print('save weight successfully')
# model.save('PSTRN.h5')
# print('save model successfully')

# RMSE和MAPE需要自己实现
# def RMSE(origin,predict):
#     origin = origin[:,1:15,1:15]
#     predict = predict[:,1:15,1:15]
#     origin = np.reshape(origin,(-1))
#     predict = np.reshape(predict,(-1))
#     n = len(origin)
#     sum = 0
#     nn =0
#     for i in range(n):
#         if origin[i] >0.1:
#             loss = abs(origin[i]-predict[i])
#             sum += loss**2
#             # sum += ((loss-2.5) if loss>2.5 else 0)**2
#             nn += 1
#     return math.sqrt(sum/nn)

def MSE(origin, predict):
    # origin = origin[:,1:15,1:15]
    # predict = predict[:,1:15,1:15]
    origin = np.reshape(origin, (-1))
    predict = np.reshape(predict, (-1))
    n = len(origin)
    sum = 0
    nn = 0
    for i in range(n):
        # if origin[i] > 0.1:
        loss = abs(origin[i] - predict[i])
        sum += loss ** 2
        # sum += ((loss-2.5) if loss>2.5 else 0)**2
        nn += 1
    return sum / nn


def MAPE(origin, predict):
    # origin = origin[:,1:15,1:15]
    # predict = predict[:,1:15,1:15]
    origin = np.reshape(origin, (-1))
    predict = np.reshape(predict, (-1))
    n = len(origin)
    sum = 0
    nn = 0
    for i in range(n):
        if origin[i] != 0:
            loss = abs(origin[i] - predict[i])
            sum += loss / origin[i]
            # sum += ((loss-2.5) if loss>2.5 else 0)/origin[i]
        nn += 1
    return sum / nn


def predictIteration(b, z, poi):
    predictions = model.predict(
        {
            'inputX': b,
            'inputZ': z,
            'inputPOI': poi
        })

    return predictions


dataTestX = np.load("/data/wxf/code/N2D/data2c/dataTestX.npy")
dataTestY = np.load("/data/wxf/code/N2D/data2c/dataTestY.npy")
dataTestZ = np.load("/data/wxf/code/N2D/data2c/dataTestZ.npy")
dataTestPOI = np.load("/data/wxf/code/N2D/data2c/newDataTestPOI.npy")
# # print(dataPOI.shape)
# newDataTestPOI = np.zeros((len(dataTestZ), 13, 16, 16))
# for i in range(len(dataTestZ)):
#     newDataTestPOI[i] = dataTestPOI
# print(newDataTestPOI.shape)
# np.save("/data/wxf/code/N2D/data2c/newDataTestPOI.npy", newDataTestPOI)

b = dataTestX
z = dataTestZ
poi = dataTestPOI
step = 1
for i in range(15):
    prediction = predictIteration(b, z, poi)
    # print(dataTestY.shape)
    # print(prediction.shape)

    np.save("/data/wxf/code/N2D/data2c/predictions/o_" + str(i) + ".npy", prediction)  # (-1, 4, 16, 16)
    print(
        "      total      |      in      |      out      | MSE MAPE  " + str(
            i + 1))
    print(str(round(MSE(dataTestY, prediction), 2)) + " " + str(
        round(100 * MAPE(dataTestY, prediction), 2)) + "          " +
          str(round(MSE(dataTestY[:, 0:1, :, :], prediction[:, 0:1, :, :]), 2)) + " " + str(
        round(100 * MAPE(dataTestY[:, 0:1, :, :], prediction[:, 0:1, :, :]), 2)) + "      " +
          str(round(MSE(dataTestY[:, 1:2, :, :], prediction[:, 1:2, :, :]), 2)) + " " + str(
        round(100 * MAPE(dataTestY[:, 1:2, :, :], prediction[:, 1:2, :, :]), 2)) + "      "
          )

    b[0:len(prediction) - step, 0:4, :, :, :] = b[0:len(prediction) - step, 1:5, :, :, :]
    b[0:len(prediction) - step, 4, :, :, :] = prediction[0:len(prediction) - step, :, :, :]
    b = b[0:len(prediction) - step]
    dataTestY = dataTestY[step:len(prediction)]
    z = z[step:len(prediction)]
    poi = poi[step:len(prediction)]
    # print(b.shape, z.shape, poi.shape)

