import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np
from keras.models import *
import pandas as pd 
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
path = 'C:/Users/USER/Desktop/MOST/model/save_light_front_17996/'
data_y = pd.read_csv('C:/Users/USER/Desktop/MOST/model/test.csv',sep=",",encoding='utf-8')

dirs = os.listdir(path)

slidwin_data=[]
out = []
X_train=[]
SBP=[]
DBP=[]
Y_train=[]
for name in dirs:
    if os.path.splitext(name)[1] == ".csv":
        csv_name = name
        who = csv_name.split(sep='_')[1] #分解csv名稱用
        data_x = pd.read_csv(path+csv_name,sep=",",encoding='utf-8')
        data_x = data_x.dropna()
        for row in data_y:
            if row == who:
                bp = data_y[row]
                SBP = np.array(bp[0])
                DBP = np.array(bp[1])
                Y_train.append([bp[0],bp[1]]) 
                continue
        for row in range(250,len(data_x),10):   #sliding window *250*1
            slidwin_data.append(data_x[row-250:row])

X_train = np.array(slidwin_data)
Y_train = np.array(Y_train)

X_train_s = X_train.shape[0]
Y_train_s = Y_train.shape[0]
r = X_train_s/Y_train_s

Y_train = np.array([val for val in Y_train for _ in range(int(r))])
print(Y_train.shape)

gpu_devices = tf.config.experimental.list_physical_devices("GPU")
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)

# class AttentionLayer(Layer):
#     def __init__(self, **kwargs):
#         super(AttentionLayer, self).__init__(**kwargs)

#     def build(self, input_shape):
#         self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], 1), initializer="normal")
#         self.b = self.add_weight(name="att_bias", shape=(input_shape[1], 1), initializer="zeros")
#         super(AttentionLayer, self).build(input_shape)

#     def call(self, x):
#         e = K.tanh(K.dot(x, self.W) + self.b)
#         a = K.softmax(e, axis=1)
#         output = x * a
#         return K.sum(output, axis=1)

#     def compute_output_shape(self, input_shape):
#         return (input_shape[0], input_shape[-1])

def Model():
    model = Sequential()
    model.add(Conv1D(filters=20,kernel_size=9,strides=1,input_shape=(250,1),padding="SAME",activation = 'relu'))  # filters空間維度,kernel_size卷積窗口的長度,strides卷積的步長,
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=4,padding="SAME"))   #pool_size窗口大小,strides縮小比例的因數,padding: "valid" 或者 "same"
    model.add(Dropout(rate =0.1))
    model.add(Conv1D(filters=20,kernel_size=9,strides=1,padding="SAME",activation = 'relu'))  # filters空間維度,kernel_size卷積窗口的長度,strides卷積的步長,
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=4,padding="SAME"))   #pool_size窗口  大小,strides縮小比例的因數,padding: "valid" 或者 "same"
    model.add(Dropout(rate =0.1))
    
    model.add(Bidirectional(GRU(64, return_sequences=True)))
    model.add(Dropout(rate=0.1))
    model.add(Bidirectional(GRU(128)))
    model.add(Dropout(rate=0.1))
    model.add(Flatten())
    model.add(Dense(2, activation='relu'))
    return model
model = Model()
model.summary()

opt = Adam(learning_rate=0.001)
# adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

model.compile(loss='mean_squared_error',
				optimizer=opt,
				metrics=[tf.keras.metrics.RootMeanSquaredError()])

# (Do!) 自訂 batch_size, epochs限制= 20
batch_size = 128 #128
epochs = 50
seed = 7
X_train, x_test, Y_train, y_test = train_test_split(X_train, Y_train, test_size=0.33, random_state=seed)
record = model.fit( X_train, Y_train,
                    validation_data=(x_test,y_test),
					batch_size=batch_size,
					epochs=epochs,
					verbose=1,
					shuffle=True,
					)	

scores = model.evaluate(x=x_test,y=y_test,)
print(scores[0])

plt.plot(record.history['loss'], label='train loss')
plt.plot(record.history['val_loss'], label='val loss')
plt.xlabel("epoch")
plt.ylabel("Loss")
plt.legend()
plt.savefig('model.png',dpi=300,format='png')
print('Result saved into model.png')

y_pred = model.predict(x_test)
# print(x_test)
print(y_pred)
print(y_test)
# print(type(y_test))
# y_test_out = np.array(y_test)
# for i in y_test_out:
#     print(y_test_out[i][0])

pd.DataFrame(y_pred).to_csv('y_pred.csv')
pd.DataFrame(y_test).to_csv('y_test.csv')
