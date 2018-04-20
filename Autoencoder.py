import numpy as np
from keras.datasets import  mnist
from keras.models import Model
from keras.layers import Dense,Input
import matplotlib.pyplot as plt


#process data
(X_train,_),(X_test,Y_test)=mnist.load_data()
X_train=X_train.astype("float32")/255-0.5
X_test=X_test.astype("float32")/255-0.5
X_train=X_train.reshape((X_train.shape[0],-1))
X_test=X_test.reshape((X_test.shape[0],-1))
print(X_train.shape)
print(X_test.shape)

#build model
#the target dimension to compress
encoding_dim=2
#placeholder for input
input_img=Input(shape=(784,))

#encoder layers
encoded=Dense(128,activation="relu")(input_img)
encoded=Dense(64,activation="relu")(encoded)
encoded=Dense(10,activation="relu")(encoded)
encoder_output=Dense(encoding_dim)(encoded)

#decode layers
decoded=Dense(10,activation="relu")(encoder_output)
decoded=Dense(64,activation="relu")(decoded)
decoded=Dense(128,activation="relu")(decoded)
decoded=Dense(784,activation="tanh")(decoded)

#autoencoder model
autoencoder=Model(input=input_img,output=decoded)
#encoder model
encoder=Model(input=input_img,output=encoder_output)

#compile model
autoencoder.compile(optimizer="adam",loss="mse")
autoencoder.fit(X_train,X_train,epochs=50,batch_size=256,shuffle=True)

# plotting
encoded_imgs = encoder.predict(X_test)
plt.scatter(encoded_imgs[:, 0], encoded_imgs[:, 1], c=Y_test)
plt.colorbar()
plt.show()