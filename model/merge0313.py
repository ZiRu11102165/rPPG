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
from keras.utils.vis_utils import plot_model
from sklearn.model_selection import train_test_split
from keras import backend as K
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ReduceLROnPlateau
import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
from keras.layers.merging.concatenate import concatenate

def raw_data():
    # LOAD TRAINING DATA
    path = 'C:/Users/USER/Desktop/MOST/model/save_light_front_17996/one_0309/'
    data_y = pd.read_csv('C:/Users/USER/Desktop/MOST/model/test.csv',sep=",",encoding='utf-8')
    dirs = os.listdir(path)
    slidwin_data=[]
    name_bp=[]
    for name in dirs:
        if os.path.splitext(name)[1] == ".csv":
            csv_name = name
            who = csv_name.split(sep='_')[1] #分解csv名稱用
            data_x = pd.read_csv(path+csv_name,sep=",",encoding='utf-8')
            data_x = data_x.iloc[ :12900]     # 每個資料只取210s
            data_x = data_x.dropna()
            for row in range(60,len(data_x),60):   #sliding window *250*1
                # slidwin_data.append(data_x[row-250:row])
                bp = data_y[who]
                slidwin = data_x[row-60:row]
                name_bp.append([bp[0],bp[1]])
                slidwin_data.append(slidwin)

    r_d = np.array(slidwin_data)
    name_bp = np.array(name_bp)
    return r_d,name_bp


def feature_data():
    # LOAD TRAINING DATA
    path_feature = 'C:/Users/USER/Desktop/MOST/model/datasets_ofus/choose/'
    data_y = pd.read_csv('C:/Users/USER/Desktop/MOST/model/test.csv',sep=",",encoding='utf-8')
    dirs = os.listdir(path_feature)
    slidwin_data=[]
    name_bp = []
    for name in dirs:
        if os.path.splitext(name)[1] == ".csv":
            csv_name = name
            who = csv_name.split(sep='_')[2] #分解csv名稱用
            data_x = pd.read_csv(path_feature+csv_name,sep=",",encoding='utf-8')
            data_x = data_x.iloc[ :215]     # 每個資料只取210s
            data_x = data_x.dropna()
            for row in range(1,len(data_x),1):   #sliding window *250*1
                bp = data_y[who]
                slidwin=data_x[row-1:row]
                name_bp.append([bp[0],bp[1]])
                slidwin_data.append(slidwin)
                # slidwin_data.append(data_x[row-80:row])
            
    f_d = np.array(slidwin_data)
    # f_d = slidwin_data
    name_bp = np.array(name_bp)
    return f_d,name_bp
raw_all = raw_data()
feature_all = feature_data()

r_in = raw_all[0]    
f_in = feature_all[0]  

r_y = raw_all[1]    
f_y = feature_all[1]    

# print(r_in.shape)   
# print(f_in.shape)   
# print(r_y.shape)    
# print(f_y.shape)    

# print((r_y==f_y).all())     # 確認BP值是否一致

# 定義第一個模型，處理第一個輸入
def Model_raw():
    model = Sequential()
    # 原本是relu Conv的activation改成 elu or gelu
    model.add(Conv1D(filters=20,kernel_size=9,strides=1,input_shape=(60,1),padding="SAME",activation = 'relu'))  # filters空間維度,kernel_size卷積窗口的長度,strides卷積的步長,
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=4,padding="SAME"))   #pool_size窗口大小,strides縮小比例的因數,padding: "valid" 或者 "same"
    model.add(Dropout(rate =0.0))
    
    model.add(Conv1D(filters=20,kernel_size=9,strides=1,padding="SAME",activation = 'relu'))  # filters空間維度,kernel_size卷積窗口的長度,strides卷積的步長,
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=4,padding="SAME"))   #pool_size窗口  大小,strides縮小比例的因數,padding: "valid" 或者 "same"
    model.add(Dropout(rate =0.0))
    
    model.add(Bidirectional(GRU(64, return_sequences=True)))
    model.add(Dropout(rate=0.0))
    model.add(Bidirectional(GRU(128)))
    model.add(Dropout(rate=0.0))
    model.add(Flatten())
    model.add(Dense(32, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(2, activation='relu'))
    return model
r_model = Model_raw()
r_model.summary()

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

f_model = fully_connected_resnet(input_shape= (1,), num_classes= 2, num_blocks= 5, filters = 128, l2_reg= 0.01 )
f_model.summary()

#融合兩個模型的輸出
# 定義第一個輸入層
input_raw = Input(shape=(60,1))
x1 = r_model(input_raw)

# # 定義第二個輸入層
input_feature = Input(shape=(1,1))
x2 = f_model(input_feature)


# # 將兩個輸入合併
combined = concatenate([x1, x2])

# 創建全連接層
output = Dense(2, activation='relu')(combined)

# 建立模型
# 定義模型
model = Model(inputs=[input_raw, input_feature], outputs=output)
model.summary()
# 編譯模型
r_opt = tf.keras.optimizers.AdamW(learning_rate=0.001) 
f_opt = tf.keras.optimizers.AdamW(learning_rate=0.001) 
opt = tf.keras.optimizers.AdamW(learning_rate=0.001) 

# use scheduler
def scheduler(epoch):  #scheduler用
    lr = K.get_value(model.optimizer.lr)
    if epoch<15:
        lr=lr
    elif epoch % 15 ==0:
        lr *= 0.5
    return lr
learning_rate_re = keras.callbacks.LearningRateScheduler(scheduler)  #scheduler用


r_model.compile(optimizer=r_opt , 
              loss='mean_absolute_error', 
              metrics=[tf.keras.metrics.RootMeanSquaredError()]
            #    metrics=['accuracy']
               )
f_model.compile(optimizer=f_opt , 
              loss='mean_absolute_error', 
              metrics=[tf.keras.metrics.RootMeanSquaredError()]
            #   metrics=['accuracy']
               )
model.compile(optimizer=opt , 
              loss='mean_absolute_error', 
              metrics=[tf.keras.metrics.RootMeanSquaredError()]
            #   metrics=['accuracy']
             )
# r_model.save('./model_save/keras_r_model_GRU.h5')
# f_model.save('./model_save/keras_f_model_GRU.h5')
# model.save('./model_save/keras_model_merge_GRU.h5')
# (Do!) 自訂 batch_size, epochs限制= 20
batch_size = 128#128
epochs = 50
seed = 7

raw_x_train, raw_x_test, raw_y_train, raw_y_test = train_test_split(r_in, r_y, test_size=0.33, random_state=seed) 
fea_x_train, fea_x_test, fea_y_train, fea_y_test = train_test_split(f_in, f_y, test_size=0.33, random_state=seed) 

# 訓練第一個模型
print('---------------------r_model---------------------')
r_history = r_model.fit(raw_x_train, raw_y_train, validation_data=(raw_x_test, raw_y_test),shuffle=True, epochs=epochs, batch_size=batch_size, verbose=1,
            callbacks = [learning_rate_re, 
                        #  tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, mode='min', restore_best_weights=True,)],  #scheduler用
                        ],
            )
# 獲取第一個模型的輸出
r_scores = r_model.evaluate(x=raw_x_test,y=raw_y_test,)
print(r_scores[0])
# r_output_train = r_model.predict(raw_x_train)
# r_output_val = r_model.predict(raw_x_test)
plt.plot(r_history.history['loss'], label='signal_train loss')
plt.plot(r_history.history['val_loss'], label='signal_val loss')
plt.xlabel("epoch")
plt.ylabel("Loss")
plt.legend()
plt.savefig('merge_model.png',dpi=300,format='png')
print('Result saved into r_model.png')

# 訓練第二個模型
print('---------------------f_model---------------------')
f_history = f_model.fit(fea_x_train, fea_y_train, validation_data=(fea_x_test, fea_y_test),shuffle=True, epochs=epochs, batch_size=batch_size, verbose=1,
            callbacks = [learning_rate_re, 
                        #  tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, mode='min', restore_best_weights=True,)],  #scheduler用
                        ],
            )
# 獲取第二個模型的輸出
f_scores = f_model.evaluate(x=fea_x_test,y=fea_y_test,)
print(f_scores[0])
# f_output_train = f_model.predict(fea_x_train)
# f_output_val = f_model.predict(fea_x_test)
plt.plot(f_history.history['loss'], label='feature_train loss')
plt.plot(f_history.history['val_loss'], label='feature_val loss')
plt.xlabel("epoch")
plt.ylabel("Loss")
plt.legend()
plt.savefig('merge_model.png',dpi=300,format='png')
print('Result saved into f_model.png')

# print(raw_x_train.shape, raw_x_test.shape, raw_y_train.shape, raw_y_test.shape)
# print(fea_x_train.shape, fea_x_test.shape, fea_y_train.shape, fea_y_test.shape)
# 訓練模型
print('---------------------model---------------------')

m_history = model.fit([raw_x_train,fea_x_train], raw_y_train, 
                     validation_data = ([raw_x_test, fea_x_test], raw_y_test),
                     epochs=epochs, batch_size=batch_size, verbose=1,shuffle=True,
                     callbacks = [learning_rate_re,
                                #  tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, mode='min', restore_best_weights=True,)],  #scheduler用
                                 ],
                     )
# 獲取合併模型的輸出
m_scores = model.evaluate(x=[raw_x_test,fea_x_test],y=raw_y_test,)
print(m_scores[0])

plt.plot(m_history.history['loss'], label='merge_train loss')
plt.plot(m_history.history['val_loss'], label='merge_val loss')
plt.xlabel("epoch")
plt.ylabel("Loss")
plt.legend()
plt.savefig('merge_model.png',dpi=300,format='png')
print('Result saved into merge_model.png')

model.save('./model_save/keras_model_merge_GRU.h5')

# ---------------------有問題待解決---------------------
# 1. 目前(0307)r_model的loss: 8.0221 - root_mean_squared_error: 12.9347
# 2. 目前(0307)f_model的loss: 11.5751 - root_mean_squared_error: 15.4181
# 3. merge有問題，目前將r_model跟f_model兩模型預測出的值再丟入FC*3，但loss跟RMSE卡住(loss: 9.8925 - root_mean_squared_error: 14.8490)
# 4. 看是要修正，或是繼續之前方法利用Concatenate(但資料問題要繼續解決)、
# ---------------------有問題待解決---------------------
