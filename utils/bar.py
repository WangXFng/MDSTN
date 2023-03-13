import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
from matplotlib.ticker import MultipleLocator,FuncFormatter
 
# 绘图设置
fig = plt.figure()
ax = fig.gca(projection='3d')  # 三维坐标轴
# X和Y的个数要相同
X = [3,4,5,6] # 64
Y = [3,4,5,6] # 64
Z = np.random.randint(0, 1000, 16) # 生成16个随机整数
# meshgrid把X和Y变成平方长度，比如原来都是4，经过meshgrid和ravel之后，长度都变成了16，因为网格点是16个
xx, yy = np.meshgrid(X, Y)  # 网格化坐标
X, Y = xx.ravel(), yy.ravel()  # 矩阵扁平化
# 设置柱子属性
height = np.zeros_like(Z) # 新建全0数组，shape和Z相同，据说是图中底部的位置
width = depth = 0.3 # 柱子的长和宽
# 颜色数组，长度和Z一致


c = [
     'lightsteelblue','lightsteelblue','lightsteelblue','lightsteelblue',
     'lightsteelblue','lightsteelblue','lightsteelblue','lightsteelblue',
     'lightsteelblue','red','lightsteelblue','lightsteelblue',
     'lightsteelblue','lightsteelblue','lightsteelblue','lightsteelblue'
     ]
# c = ['mediumpurple']*len(Z)

# c = 'mediumpurple'

# 开始画图，注意本来的顺序是X, Y, Z, width, depth, height，但是那样会导致不能形成柱子，只有柱子顶端薄片，所以Z和height要互换
ax.bar3d(X, Y, height, width, depth, Z,  color=c, shade=True)  # width, depth, height

def format_(temp, position):
    return '%d'%(math.pow( 2, temp ))


# x = [1,2,3,4]
# _x_ticks = [8,16,32,74]

ax.xaxis.set_major_formatter(FuncFormatter(format_))#将X,Y,Z轴的坐标轴放大100倍
ax.yaxis.set_major_formatter(FuncFormatter(format_))
# ax.zaxis.set_major_formatter(FuncFormatter(format_))

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()