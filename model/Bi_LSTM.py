
# LSTM and CNN for sequence classification in the IMDB dataset
import numpy as np
import pandas as pd
import tensorflow as tf
import os
from tensorflow import keras
from tensorflow.keras.models import Sequential  #引入Sequential函式
from tensorflow.keras.layers import *
from tensorflow.keras.preprocessing import sequence

os.environ["CUDA_VISIBLE_DEVICES"]="0"  #看是否用GPU
#load .csv檔的資料
df = pd.read_csv(r'Path where the CSV file is stored\File name.csv')
print(df)

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
# model.add(Attention())
print(model.summary())  #輸出模型
opt = keras.optimizers.Adam(learning_rate=0.001)
model.compile(loss='mean_squared_error',optimizer=opt,metrics=['accuracy'])
# model.fit(X_train, y_train, epochs=20, batch_size=128)    #model.fit(訓練資料, 目標資料, epochs, batch_size,validation_split=0.2從測試集分80%給訓練集,validation_freq測試的間隔次數)
# # Final evaluation of the model
# scores = model.evaluate(X_test, y_test, verbose=0)
# print("Accuracy: %.2f%%" % (scores[1]*100))

model.save("test.h5")
