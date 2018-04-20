import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import SimpleRNN,Activation,Dense

TIME_STEPS=28
INPUT_SIZE=28
BATCH_SIZE=50
BATCH_INDEX=0
OUTPUT_SIZE=10
CELL_SIZE=50
LR=0.01

#load and process data
(X_train,Y_train),(X_test,Y_test)=mnist.load_data()


#normalize
X_train=X_train.reshape(-1,28,28)/255
X_test=X_test.reshape(-1,28,28)/255
Y_train=np_utils.to_categorical(Y_train,num_classes=10)
Y_test=np_utils.to_categorical(Y_test,num_classes=10)

#build model
model=Sequential()
model.add(
    SimpleRNN(
        batch_input_shape=(None,TIME_STEPS,INPUT_SIZE),
        output_dim=CELL_SIZE,
        unroll=True
    )
)

model.add(Dense(OUTPUT_SIZE))
model.add(Activation('softmax'))

#optimizer
adam=Adam(lr=LR)
model.compile(loss="categorical_crossentropy",optimizer=adam,metrics=["accuracy"])

#train model & evaluate
for step in range(4001):
    X_batch=X_train[BATCH_INDEX:BATCH_INDEX+BATCH_SIZE,:,:]
    Y_batch=Y_train[BATCH_INDEX:BATCH_INDEX+BATCH_SIZE,:]
    cost=model.train_on_batch(X_batch,Y_batch)
    BATCH_INDEX+=BATCH_SIZE
    BATCH_INDEX=0 if BATCH_INDEX>=X_train.shape[0]else BATCH_INDEX
    if step%500==0:
        cost,accuracy=model.evaluate(X_test,Y_test,batch_size=Y_test.shape[0],verbose=False)
        print("test cost: ",cost,"test accuracy:",accuracy)