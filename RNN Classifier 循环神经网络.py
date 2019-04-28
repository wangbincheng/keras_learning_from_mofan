from keras.layers import SimpleRNN, Activation, Dense
import numpy as np
np.random.seed(1337) # for reproducibility
from keras.utils import np_utils
from keras.models import Sequential
from keras.datasets import mnist

# download the mnist to the path '~/.keras/datasets/' if it is the first time to be called
# X shape (60,000 28x28), y shape (10,000, )
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# data pre-processing
X_train = X_train.reshape(-1, 28, 28) / 255.      # normalize
X_test = X_test.reshape(-1, 28, 28) / 255.        # normalize
y_train = np_utils.to_categorical(y_train, num_classes=10)
y_test = np_utils.to_categorical(y_test, num_classes=10)


print('X_train shape: ',X_train.shape)
print('X_test shape: ',np.shape(X_test ))


TIME_STEPS=28  #same as the height of the image
INPUT_SIZE=28   #same as the height of the image
BATCH_SIZE=50
BATCH_INDEX=0
OUTPUT_SIZE=10
CELL_SIZE=50
LR=0.001


model = Sequential()
#添加RNN层，输入为训练数据，输出数据大小由CELL_SIZE定义。
model.add(SimpleRNN(
    # for batch_input_shape, if using tensorflow as the backend, we have to put None for the batch_size.
    # Otherwise, model.evaluate() will get error.
    batch_input_shape=(None, TIME_STEPS, INPUT_SIZE),       
    output_dim=CELL_SIZE,
    unroll=True,
))

#添加输出层，激励函数选择softmax
model.add(Dense(OUTPUT_SIZE))
model.add(Activation('softmax'))

#设置adam优化方法，loss函数, metrics方法来观察输出结果
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
#开始训练模型
for step in range(4001):
    # data shape = (batch_num, steps, inputs/outputs)
    X_batch = X_train[BATCH_INDEX: BATCH_INDEX+BATCH_SIZE, :, :]
    Y_batch = y_train[BATCH_INDEX: BATCH_INDEX+BATCH_SIZE, :]
    cost = model.train_on_batch(X_batch, Y_batch)
    
    BATCH_INDEX += BATCH_SIZE
    BATCH_INDEX = 0 if BATCH_INDEX >= X_train.shape[0] else BATCH_INDEX
    # Evaluate the model with the metrics we defined earlier
    if step % 500 == 0:
        cost, accuracy = model.evaluate(X_test, y_test, batch_size=y_test.shape[0], verbose=False)
        print('test cost: ', cost, 'test accuracy: ', accuracy)
