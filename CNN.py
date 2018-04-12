import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Dense,Activation,Convolution2D,MaxPooling2D,Flatten

#load and process data
(X_train,Y_train),(X_test,Y_test)=mnist.load_data()

plt.imshow(X_train[0], cmap=plt.get_cmap('gray'))
plt.show()

X_train=X_train.reshape(-1,28,28,1)
X_test=X_test.reshape(-1,28,28,1)
Y_train=np_utils.to_categorical(Y_train,num_classes=10)
Y_test=np_utils.to_categorical(Y_test,num_classes=10)

print(X_train.shape)

#build model
model=Sequential()

#conv layer 1
model.add(Convolution2D(
    input_shape=(28,28,1),
    filters=32,
    kernel_size=5,
    strides=1,
    padding='same',
    data_format='channels_first'
))
model.add(Activation("relu"))

#pooling layer 1
model.add(MaxPooling2D(
    pool_size=2,
    strides=2,
    padding="same",
    data_format='channels_first'
))

#conv layer 2
model.add(Convolution2D(
    filters=64,
    kernel_size=5,
    strides=1,padding="same",
    data_format='channels_first'
))
model.add(Activation("relu"))

#pooling layer 2
model.add(MaxPooling2D(
    2,2,"same",data_format='channels_first'
))

#flatten layer and dense
model.add(Flatten())
model.add(Dense(1024))
model.add(Activation("relu"))

model.add(Dense(10))
model.add(Activation("softmax"))

#optimizer
adam=Adam(lr=1e-04)
model.compile(loss="categorical_crossentropy",optimizer=adam,metrics=["accuracy"])

#train model
print("Train start:")
model.fit(X_train,Y_train,epochs=1,batch_size=32)

#evaluate model
loss,accuracy=model.evaluate(X_test,Y_test)
print("loss: ",loss)
print("accuracy: ",accuracy)