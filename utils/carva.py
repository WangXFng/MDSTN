from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
figure = plt.figure()
ax = Axes3D(figure)
X = np.arange(1,5,1)
Y = np.arange(1,5,1)
X,Y = np.meshgrid(X,Y)
 
# R = np.sqrt(X**2 + Y**2)
# Z = np.cos(R)

Z = np.array([[1,2,3,1],
     [1,2,3,2],
     [1,4,3,2],
     [1,2,3,2]])

ax.plot_surface(X,Y,Z,rstride=1,cstride=1,cmap='rainbow')
# ax.contour3D(X,Y,Z,50,cmap='jet')
plt.show()