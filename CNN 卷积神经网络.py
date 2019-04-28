from keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Flatten

import numpy as np
np.random.seed(1337) # for reproducibility
from keras.utils import np_utils
from keras.models import Sequential
from keras.datasets import mnist

# download the mnist to the path '~/.keras/datasets/' if it is the first time to be called
# X shape (60,000 28x28), y shape (10,000, )
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# data pre-processing
X_train = X_train.reshape(-1,1,28,28)
X_test = X_test.reshape(-1,1,28,28)   
y_train = np_utils.to_categorical(y_train, num_classes=10)
y_test = np_utils.to_categorical(y_test, num_classes=10)


print('X_train shape: ',X_train.shape)
print('X_test shape: ',np.shape(X_test ))

model = Sequential()


#添加第一个卷积层，滤波器数量为32，大小是5*5，Padding方法是same即不改变数据的长度和宽带。 
#因为是第一层所以需要说明输入数据的 shape ，激励选择 relu 函数。

model.add(Convolution2D(
    batch_input_shape=(32, 1, 28, 28),
    filters=32,
    kernel_size=5,
    strides=1,
    padding='same',      # Padding method
    data_format='channels_last',
))
model.add(Activation('relu'))

#第一层 pooling(池化，下采样)，分辨率长宽各降低一半，输出数据shape为（32，14，14）
model.add(MaxPooling2D(
    pool_size=2,
    strides=2,
    padding='same',    # Padding method
    data_format='channels_last',
))
#再添加第二卷积层和池化层
model.add(Convolution2D(64, 5, strides=1, padding='same', data_format='channels_last'))
model.add(Activation('relu'))
model.add(MaxPooling2D(2, 2, 'same', data_format='channels_last'))
#经过以上处理之后数据shape为（64，7，7），需要将数据抹平成一维，再添加全连接层1
model.add(Flatten())
model.add(Dense(256))
model.add(Activation('relu'))
#添加全连接层2（即输出层）
model.add(Dense(10))
model.add(Activation('softmax'))
#设置adam优化方法，loss函数, metrics方法来观察输出结果
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
#开始训练模型
print('\nTraining ------------')
model.fit(X_train, y_train, epochs=2, batch_size=32,)
# Evaluate the model with the metrics we defined earlier

print('\nTesting ------------')
loss, accuracy = model.evaluate(X_test, y_test)

print('test loss: ', loss)
print('test accuracy: ', accuracy)
