import numpy as np
np.random.seed(1337)
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense,Activation
from keras.optimizers import RMSprop

#load data
(X_train,Y_train),(X_test,Y_test)=mnist.load_data()

print(X_train.shape)

#normalize the data, reshape X_train's shape from (60000 28 28) to (60000 784)
X_train=X_train.reshape(X_train.shape[0],-1)/255
X_test=X_test.reshape(X_test.shape[0],-1)/255

#because need to divide into 10 classes, so need to use "to_categorical" to change y into one-hot style
Y_train=np_utils.to_categorical(Y_train,num_classes=10)
Y_test=np_utils.to_categorical(Y_test,num_classes=10)

print(X_train.shape)

#build model
model=Sequential([
    Dense(32,input_dim=784),
    Activation('relu'),
    Dense(10),
    Activation('softmax')
])

#optimizer,use Root Mean Square method
rmsprop=RMSprop(lr=0.001,rho=0.8,epsilon=1e-08,decay=0)

#compile model,metrics means when optimize the loss, calculate the current accuracy
model.compile(optimizer=rmsprop,loss="categorical_crossentropy",metrics=["accuracy"])

#train model
print("Train start:")
model.fit(X_train,Y_train,epochs=2,batch_size=32)

#test model
loss,accuarcy=model.evaluate(X_test,Y_test)
print("loss: ",loss)
print("accuracy: ",accuarcy)

#
print(Y_test[0])
preds=model.predict(X_test,batch_size=32)>0.9
print(preds[0])





