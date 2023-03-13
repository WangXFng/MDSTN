import numpy as np

#

#

start_map = np.load('start_map_nyc.npy')
end_map = np.load('end_map_nyc.npy')

dataX = np.zeros((len(start_map)-5, 5, 2, 16, 16))
dataY = np.zeros((len(start_map)-5, 2, 16, 16))
dataZ = np.zeros((len(start_map)-5, 2))
dataPOI = np.zeros((13, 16, 16))


for i in range(len(start_map)-5):
    hour, mins = int(i/2), 30*(int(i%2))
    dataZ[i][0], dataZ[i][1] = hour, mins
    dataX[i, :, 0, :, :] = start_map[i:i+5, :, :]
    dataX[i, :, 1, :, :] = end_map[i:i+5, :, :]
    dataY[i, 0, :, :] = start_map[i+5, :, :]
    dataY[i, 1, :, :] = end_map[i+5, :, :]


len_ = int((len(start_map)-5)*0.8)

dataTestX = dataX[len_:].copy()
dataTestZ = dataZ[len_:].copy()
dataTestY = dataY[len_:].copy()
dataTestPOI = dataPOI[len_:].copy()


dataX = dataX[0:len_].copy()
dataZ = dataZ[0:len_].copy()
dataY = dataY[0:len_].copy()
dataPOI = dataPOI[0:len_].copy()

print('dataX', dataX.shape)
print('dataY', dataY.shape)
print('dataZ', dataZ.shape)
print('dataPOI', dataPOI.shape)

print('dataTestX', dataTestX.shape)
print('dataTestY', dataTestY.shape)
print('dataTestZ', dataTestZ.shape)
print('dataTestPOI', dataTestPOI.shape)

np.save('../data/dataX.npy', dataX)
np.save('../data/dataY.npy', dataY)
np.save('../data/dataZ.npy', dataZ)
np.save('../data/dataPOI.npy', dataPOI)

np.save('../data/dataTestX.npy', dataTestX)
np.save('../data/dataTestY.npy', dataTestY)
np.save('../data/dataTestZ.npy', dataTestZ)
np.save('../data/dataTestPOI.npy', dataTestPOI)

