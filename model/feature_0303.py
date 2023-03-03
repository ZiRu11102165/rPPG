import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np
from keras.models import *
from tensorflow import keras
import pandas as pd 
import tensorflow as tf 
from keras.layers.core import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.models import * 
from tensorflow.keras.layers import *
from tensorflow.keras.losses import *
from sklearn.model_selection import train_test_split
from keras import backend as K
import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt

# LOAD TRAINING DATA
# LOAD TRAINING DATA
path_feature = 'C:/Users/USER/Desktop/MOST/model/datasets_ofus/choose/'
data_y = pd.read_csv('C:/Users/USER/Desktop/MOST/model/test.csv',sep=",",encoding='utf-8')

dirs = os.listdir(path_feature)

slidwin_data=[]
Re_who = []
X_train=[]
SBP=[]
DBP=[]
Y_train=[]
for name in dirs:
    if os.path.splitext(name)[1] == ".csv":
        csv_name = name
        who = csv_name.split(sep='_')[2] #分解csv名稱用
        data_x = pd.read_csv(path_feature+csv_name,sep=",",encoding='utf-8')
        data_x = data_x.dropna()
        
        for row in range(150,len(data_x),1):   #sliding window *250*1
            Re_who.append(who)
            slidwin_data.append(data_x[row-150:row])
        
for i in range(len(Re_who)):
    bp = data_y[Re_who[i]]
    SBP = bp[0]
    DBP = bp[1]
    Y_train.append([SBP,DBP]) 

X_train = np.array(slidwin_data)
Y_train = np.array(Y_train)

print(X_train.shape)
print(Y_train.shape)

gpu_devices = tf.config.experimental.list_physical_devices("GPU")
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)

def Model():
    model = Sequential()
    # 原本是relu Conv的activation改成 elu or gelu
    model.add(Conv1D(filters=20,kernel_size=9,strides=1,input_shape=(150,1),padding="SAME",activation = 'tanh'))  # filters空間維度,kernel_size卷積窗口的長度,strides卷積的步長,
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=4,padding="SAME"))   #pool_size窗口大小,strides縮小比例的因數,padding: "valid" 或者 "same"
    model.add(Dropout(rate =0.0))
    
    model.add(Conv1D(filters=20,kernel_size=9,strides=1,padding="SAME",activation = 'tanh'))  # filters空間維度,kernel_size卷積窗口的長度,strides卷積的步長,
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=4,padding="SAME"))   #pool_size窗口  大小,strides縮小比例的因數,padding: "valid" 或者 "same"
    model.add(Dropout(rate =0.0))
    
    model.add(Bidirectional(GRU(64, return_sequences=True,activation = 'elu')))
    model.add(Dropout(rate=0.0))
    model.add(Bidirectional(GRU(128,activation = 'elu')))
    model.add(Dropout(rate=0.0))
    model.add(Flatten())
    model.add(Dense(2, activation='relu'))
    return model
model = Model()
model.summary()

# adam with weight decay (adamW)
opt = tf.keras.optimizers.AdamW(learning_rate=0.0001) 

# use scheduler
def scheduler(epoch):  #scheduler用
    lr = K.get_value(model.optimizer.lr)
    if epoch<15:
        lr=lr
    elif epoch % 15 ==0:
        lr *= 0.5
    return lr

learning_rate_re = keras.callbacks.LearningRateScheduler(scheduler)  #scheduler用

model.compile(loss='mean_absolute_error',
				optimizer=opt,
				metrics=[tf.keras.metrics.RootMeanSquaredError()])

# (Do!) 自訂 batch_size, epochs限制= 20
batch_size = 128#128
epochs = 100
seed = 7
X_train, x_test, Y_train, y_test = train_test_split(X_train, Y_train, test_size=0.33, random_state=seed)
record = model.fit( X_train, Y_train,
                    validation_data=(x_test,y_test),
					batch_size=batch_size,
					epochs=epochs,
					verbose=1,
					shuffle=True,
                    callbacks = [learning_rate_re,
                                 tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, mode='min', restore_best_weights=True,)],  #scheduler用
					)	

scores = model.evaluate(x=x_test,y=y_test,)
print(scores[0])

model.save('./model_save/keras_model_CNNBiGRU_feature_PTT_CHROM.h5')

plt.plot(record.history['loss'], label='train loss')
plt.plot(record.history['val_loss'], label='val loss')
plt.xlabel("epoch")
plt.ylabel("Loss")
plt.legend()
plt.savefig('model_feature.png',dpi=300,format='png')
print('Result saved into model.png')

y_pred = model.predict(x_test)

pd.DataFrame(y_pred).to_csv('y_pred.csv')
pd.DataFrame(y_test).to_csv('y_test.csv')
