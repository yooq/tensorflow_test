import tensorflow as  tf
import  numpy as  np
import  matplotlib.pyplot as  plt
from  sklearn.model_selection import learning_curve

print(tf.__version__)

'''1.x'''
# plotdata = {'batchsize':[],'loss':[]}
# def moving_average(a,w=10):
#     if len(a)<w:
#         return a[:]
#     return [val if idx < w else sum(a[(idx - w):idx]) / w for idx, val in enumerate(a)]
#
#模拟数据与关系
train_X=np.linspace(-1,1,1000)
train_Y=2*train_X+np.random.randn(*train_X.shape)  # *train_X.shape 表示与trian_X具有相同shape
plt.scatter(train_X,train_Y)
plt.legend()
plt.show()

# '''创建模型'''
# # 占位符
# X=tf.placeholder('float')
# Y=tf.placeholder('float')
# # 模型参数
# W=tf.Variable(tf.random_normal([1]),name='weight')
# b=tf.Variable(tf.zeros([1]),name='bias')
# # 前向传播
# z=tf.multiply(X,W)+b
# # 反向传播
# cost=tf.reduce_mean(tf.square(Y-z))
# learning_rate=0.01
# optimizer=tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
#
# #初始化
# init=tf.global_variables_initializer()
# training_epochs=20
# display_step=2
#
# # 启动sess
# with tf.Session() as sess:
#     sess.run(init)
#
#     for epoch in range(training_epochs):
#         for (x,y) in zip(train_X,train_Y):
#             sess.run(optimizer,feed_dict={X:x,Y:y})
#
#         if epoch % display_step == 0:
#             loss=sess.run(cost,feed_dict={X:x,Y:y})
#             print('Epoch:',epoch+1,'    cost:',cost,'   W:',sess.run(W),'   b:',sess.run(b),'')
#             if not (loss=='NA'):
#                 plotdata['batchsize'].append(epoch)
#                 plotdata['loss'].append(loss)
#     print('finish')
#     print("cost=", sess.run(cost, feed_dict={X: train_X, Y: train_Y}), "W=", sess.run(W), "b=", sess.run(b))
#     # 图形显示
#     plt.plot(train_X, train_Y, 'ro', label='Original data')
#     plt.plot(train_X, sess.run(W) * train_X + sess.run(b), label='Fitted line')
#     plt.legend()
#     plt.show()
#
#     plotdata["avgloss"] = moving_average(plotdata["loss"])
#     plt.figure(1)
#     plt.subplot(211)
#     plt.plot(plotdata["batchsize"], plotdata["avgloss"], 'b--')
#     plt.xlabel('Minibatch number')
#     plt.ylabel('Loss')
#     plt.title('Minibatch run vs. Training loss')
#
#     plt.show()
#
#     print("x=0.2，z=", sess.run(z, feed_dict={X: 0.2}))
#



'''
2.0
'''

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(1,input_shape=(1,)))  #Dense 层自动初始化参数
model.compile(optimizer=tf.keras.optimizers.Adam(0.001)
              ,loss=tf.keras.losses.mse
              # ,metrics=['acc']   #这是回归准确率就算了
              )

history = model.fit(train_X
                    ,train_Y
                    ,epochs=5
                    ,batch_size=100
                    ,verbose=2
                    )

loss = history.history['loss']  #[]  每个epoch 的损失函数
# acc = history.history['acc']
print(loss)
pre = model.predict([20])

print(pre)
