
# LSTM and CNN for sequence classification in the IMDB dataset
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential  #引入Sequential函式
from tensorflow.keras.layers import Dense,Conv1D,MaxPooling1D,Embedding,Dropout
from tensorflow.keras.layers import BatchNormalization,Flatten,LSTM, Bidirectional,Attention
from tensorflow.keras.preprocessing import sequence

# create the model
model = Sequential()
model.add(Conv1D(filters=2,kernel_size=350,input_shape=(700,1),padding="SAME",activation = 'relu'))
model.add(BatchNormalization())
model.add(MaxPooling1D(pool_size=2,strides=1,padding="SAME"))

model.add(Conv1D(filters=10,kernel_size=175,padding="SAME",activation = 'relu'))
model.add(BatchNormalization())
model.add(MaxPooling1D(pool_size=2,strides=1,padding="SAME"))

model.add(Conv1D(filters=20,kernel_size=25,padding="SAME",activation = 'relu'))
model.add(BatchNormalization())
model.add(MaxPooling1D(pool_size=2,strides=1,padding="SAME"))

model.add(Conv1D(filters=40,kernel_size=10,padding="SAME",activation = 'relu'))
model.add(BatchNormalization())
model.add(MaxPooling1D(pool_size=2,strides=1,padding="SAME"))

model.add(Bidirectional(LSTM(128, dropout=0.2, return_sequences=True)))
model.add(Bidirectional(LSTM(350, dropout=0.2)))

model.add(Flatten())    #平坦化

model.add(Dense(2,activation = 'relu'))   #全連接層
print(model.summary())  #輸出模型

#model.fit(X_train, y_train, epochs=3, batch_size=64)    #model.fit(訓練資料, 目標資料, epochs, batch_size)
# # Final evaluation of the model
# scores = model.evaluate(X_test, y_test, verbose=0)
# print("Accuracy: %.2f%%" % (scores[1]*100))

model.save("test.h5")
