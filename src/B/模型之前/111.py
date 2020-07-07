import  tensorflow as  tf
import numpy as np
import matplotlib.pyplot as plt


# x= tf.constant([-4.,0.])
#
# def hime(x):
#     return (x[0]**2+x[1]-11)**2+(x[0]+x[1]**2-7)**2
#
# x= np.arange(-6.,6.,0.1)
# y= np.arange(-6.,6.,0.1)
#
# X,Y =tf.meshgrid(x,y)
# Z=hime([X,Y])
#
#
# fig = plt.figure(figsize=(12, 8))
# # ax = plt.gca(projection='3d')
# ax = plt.gca(projection='3d')
# # plt.gca()
# ax.plot_surface(X,Y,Z)
# ax.view_init(60,-30)
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# plt.show()

#
# # for step in range(20):
# #     with tf.GradientTape() as tape:
# #         tape.watch([x])
#

import numpy as np
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


fig = plt.figure(figsize=(8,6))
ax = plt.axes(projection='3d')

xx = np.arange(-50,50,0.5) #x定义域，离散
yy = np.arange(-50,50,0.5) #y定义域，离散
X, Y = np.meshgrid(xx, yy)

Z=X**2+Y**2#需要换图形就改这里
plt.title('Z=X**2+Y**2')#添加标题

ax.plot_surface(X,Y,Z,rstride = 1, cstride = 1,cmap='rainbow')
plt.show()


