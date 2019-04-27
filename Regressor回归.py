# DANGER! DANGER!
#保证程序运行
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

#导入模块并创建数据 
import numpy as np
np.random.seed(1337)  # for reproducibility
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt # 可视化模块

# create some data
X = np.linspace(-1, 1, 200)
np.random.shuffle(X)    # randomize the data
Y = 0.5 * X + 2 + np.random.normal(0, 0.05, (200, ))
# plot data
plt.scatter(X, Y)
plt.show()

X_train, Y_train = X[:160], Y[:160]     # train 前 160 data points
X_test, Y_test = X[160:], Y[160:]       # test 后 40 data points

#建立模型
'''
然后用 Sequential 建立 model， 再用 model.add 添加神经层，添加的是 Dense 全连接神经层。
参数有两个，一个是输入数据和输出数据的维度，本代码的例子中 x 和 y 是一维的。
'''
model = Sequential()
model.add(Dense(output_dim=1, input_dim=1))

#激活模型
#参数中，误差函数用的是 mse 均方误差；优化器用的是 sgd 随机梯度下降法。
# choose loss function and optimizing method
model.compile(loss='mse', optimizer='sgd')

#训练模型 
# training
print('Training -----------')
for step in range(301):
    cost = model.train_on_batch(X_train, Y_train)
    if step % 100 == 0:
        print('train cost: ', cost)

#检验模型
# test
print('\nTesting ------------')
cost = model.evaluate(X_test, Y_test, batch_size=40)
print('test cost:', cost)
W, b = model.layers[0].get_weights()
print('Weights=', W, '\nbiases=', b)

#可视化结果
# plotting the prediction
Y_pred = model.predict(X_test)
plt.scatter(X_test, Y_test)
plt.plot(X_test, Y_pred)
plt.show()
