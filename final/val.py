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
from statistics import mean
import dataset

# LOAD TRAINING DATA
# path = 'C:/Users/USER/Desktop/MOST/model/save_light_front_17996/'   #前處理後的訊號
path_test = 'C:/Users/USER/Desktop/MOST/model/save_light_front_17996/test/' #前處理後的訊號
# data_y = pd.read_csv('C:/Users/USER/Desktop/MOST/model/test.csv',sep=",",encoding='utf-8')

dirs_test = os.listdir(path_test)
slidwin_data=[]
slidwin_test=[]
out = []
X_train=[]
SBP=[]
DBP=[]
Y_train=[]


data_test = pd.read_csv(path_test+'sean2_30fps_right_face.csv',sep=",",encoding='utf-8')

data_test = data_test.dropna()
for row in range(300,len(data_test),10):   #sliding window 
    slidwin_test.append(data_test[row-300:row])
        
x_test  = np.array(slidwin_test)
# y_test  = np.array([129,86])#116/72 113/70
# print(x_test.shape)

gpu_devices = tf.config.experimental.list_physical_devices("GPU")
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)

model = load_model("./model_save/final/keras_model_CNNBiGRU.h5")
# model.summary()
pre_y = model.predict(x_test)
print(pre_y)
mean_values = np.mean(pre_y, axis=0)
print(mean_values)
pd.DataFrame(pre_y).to_csv('y_pred.csv')
# print(np.mean(pre_y[0]))

# print(np.mean(pre_y[1]))

