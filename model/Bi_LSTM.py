import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np
from numpy import genfromtxt
from keras.models import *
import pandas as pd 
from keras.layers.core import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.models import * 
from tensorflow.keras.layers import *
from keras.utils import np_utils
from sklearn.utils import shuffle
import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# LOAD TRAINING DATA
path = 'C:/Users/USER/Desktop/MOST/model/save_light_front_17996/'
data_y = pd.read_csv('C:/Users/USER/Desktop/MOST/model/test.csv',sep=",",encoding='utf-8')

dirs = os.listdir(path)
a=[]
out = []
X_train=[]
SBP=[]
DBP=[]
Y_train=[]
for name in dirs:
    if os.path.splitext(name)[1] == ".csv":
        csv_name = name
        who = csv_name.split(sep='_')[1] #分解csv名稱用
        # print(who)
        data = pd.read_csv(path+csv_name,sep=",",encoding='utf-8')
        data = data.dropna()
        # print(data)
        for row in data_y:
            if row == who:
                bp = data_y[row]
                SBP = np.array(bp[0])
                # SBP.append(bp[0])
                DBP = np.array(bp[1])
                # DBP.append(bp[1])
                Y_train.append([bp[0],bp[1]]) 
                continue
        for row in range(250,len(data),10):   #sliding window 25*700*1
            # a.append([0 for i in range(250)])
            a.append(data[row-250:row])

X_train = np.array(a)
Y_train = np.array(Y_train)
print(X_train)
s1 = X_train.shape[0]
s2 = Y_train.shape[0]
r = s1/s2
# for _ in range(int(r)):
#     for val in Y_train:
#         val
# print(X_train.shape[0])
# print(Y_train.shape[0])
Y_train = np.array([val for val in Y_train for _ in range(int(r))])
print(Y_train.shape)
# print(X_train)
# print(Y_train)
# print(Y_train)
gpu_devices = tf.config.experimental.list_physical_devices("GPU")
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)

def Model():
    model = Sequential()
    model.add(Conv1D(filters=20,kernel_size=9,strides=1,input_shape=(250,1),padding="SAME",activation = 'relu'))  # filters空間維度,kernel_size卷積窗口的長度,strides卷積的步長,
    model.add(MaxPooling1D(pool_size=4,padding="SAME"))   #pool_size窗口大小,strides縮小比例的因數,padding: "valid" 或者 "same"

    model.add(Conv1D(filters=20,kernel_size=9,strides=1,padding="SAME",activation = 'relu'))
    model.add(MaxPooling1D(pool_size=4,padding="SAME"))
    model.add(Dropout(rate =0.1))
    
    model.add(LSTM(64, return_sequences=True))
    model.add(Dropout(rate =0.1))
    model.add(LSTM(128))
    model.add(Dropout(rate =0.1))
    model.add(Dense(2, activation='relu'))
    return model
model = Model()
model.summary()

opt = Adam(learning_rate=0.001)
# adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

model.compile(loss='mean_squared_error',
				optimizer=opt,
				metrics=['accuracy'])

# (Do!) 自訂 batch_size, epochs限制= 20
batch_size = 128
epochs = 20

record = model.fit( (X_train), (Y_train),
					batch_size=batch_size,
					epochs=epochs,
					verbose=1,
					shuffle=True,
                    validation_split=0.33,
                    
					)	

loss	= record.history.get('loss')
acc 	= record.history.get('accuracy')

plt.figure(0)
plt.subplot(121)
plt.plot(range(len(loss)), loss,label='loss')
plt.title('Loss')
plt.legend(loc='upper left')
plt.subplot(122)
plt.plot(range(len(acc)), acc,label='accuracy')
plt.title('Accuracy')
plt.savefig('model.png',dpi=300,format='png')
print('Result saved into model.png')
