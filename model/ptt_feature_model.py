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
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ReduceLROnPlateau

from sklearn.model_selection import train_test_split
from keras import backend as K
import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt

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
        
        for row in range(1,len(data_x),1):   #sliding window *250*1
            Re_who.append(who)
            slidwin_data.append(data_x[row-1:row])
        
for i in range(len(Re_who)):
    bp = data_y[Re_who[i]]
    SBP = bp[0]
    DBP = bp[1]
    Y_train.append([SBP,DBP]) 

X_train = np.array(slidwin_data)
Y_train = np.array(Y_train)

print(X_train.shape)
print(Y_train.shape)
X_train, x_test, Y_train, y_test = train_test_split(X_train, Y_train, test_size=0.33, random_state=50)

gpu_devices = tf.config.experimental.list_physical_devices("GPU")
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)
# model start

def residual_block(x, filters, stride=1, l2_reg=0.001):
    shortcut = x
    x = Dense(filters, kernel_regularizer=l2(l2_reg))(x)
    x = Activation('elu')(x)
    if stride > 1:
        shortcut = Dense(filters, kernel_regularizer=l2(l2_reg))(shortcut)
        shortcut = Activation('elu')(shortcut)
    x = Activation('elu')(x + shortcut)
    return x
def fully_connected_resnet(input_shape, num_classes, num_blocks, filters, l2_reg=0.001):
    inputs = Input(shape=input_shape)
    x = inputs
    x = Dense(128, activation='elu')(x)
    x = BatchNormalization()(x)
    for i in range(num_blocks):
        x = residual_block(x, filters, stride=1, l2_reg=l2_reg)
    x = Dense(num_classes, activation='relu')(x)
    model = Model(inputs=inputs, outputs=x)
    return model

input_shape = (1,)
num_classes = 2
num_blocks = 5
filters = 128 #64
l2_reg = 0.01

model = fully_connected_resnet(input_shape, num_classes, num_blocks, filters, l2_reg)
model.summary()
opt = tf.keras.optimizers.Adam(learning_rate=0.0001) 

model.compile(optimizer=opt, loss='mean_absolute_error', metrics=[tf.keras.metrics.RootMeanSquaredError()])
# use scheduler
def scheduler(epoch):  #scheduler用
    lr = K.get_value(model.optimizer.lr)
    if epoch<15:
        lr=lr
    elif epoch % 15 ==0:
        lr *= 0.5
        # lr = lr * tf.math.exp(-0.1)
    return lr

learning_rate_re = keras.callbacks.LearningRateScheduler(scheduler)  #scheduler用

record = model.fit(X_train, Y_train, batch_size=128, epochs=100, validation_data=(x_test, y_test),callbacks = [learning_rate_re,
                                                                                                              tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, mode='min', restore_best_weights=True,)],
                   )
scores = model.evaluate(x=x_test,y=y_test,)
print(scores[0])
# model end
# model.save('./model_save/keras_model_ResNet_feature_PTT.h5')

plt.plot(record.history['loss'], label='train loss')
plt.plot(record.history['val_loss'], label='val loss')
plt.xlabel("epoch")
plt.ylabel("Loss")
plt.legend()
plt.savefig('model_feature.png',dpi=300,format='png')
print('Result saved into model.png')
