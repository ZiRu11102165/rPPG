# 第一個(Epoch:10 , loss: 92.1224 - RMSE: 9.5980)
def Model():
    model = Sequential()
    model.add(Conv1D(filters=20,kernel_size=9,strides=1,input_shape=(250,1),padding="SAME",activation = 'relu'))  # filters空間維度,kernel_size卷積窗口的長度,strides卷積的步長,
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=4,padding="SAME"))   #pool_size窗口大小,strides縮小比例的因數,padding: "valid" 或者 "same"
    model.add(Dropout(rate =0.1))
    model.add(Conv1D(filters=20,kernel_size=9,strides=1,padding="SAME",activation = 'relu'))  # filters空間維度,kernel_size卷積窗口的長度,strides卷積的步長,
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=4,padding="SAME"))   #pool_size窗口大小,strides縮小比例的因數,padding: "valid" 或者 "same"
    model.add(Dropout(rate =0.1))
    
    model.add(Bidirectional(GRU(64, return_sequences=True)))
    model.add(Dropout(rate=0.1))
    model.add(Bidirectional(GRU(128)))
    model.add(Dropout(rate=0.1))
    model.add(Flatten())
    model.add(Dense(2, activation='relu'))
    return model
model = Model()
opt = Adam(learning_rate=0.001)
model.compile(loss='mean_squared_error',optimizer=opt,metrics=['mae'])
batch_size = 32
epochs = 50
seed = 7
X_train, x_test, Y_train, y_test = train_test_split(X_train, Y_train, test_size=0.33, random_state=seed)
record = model.fit( X_train, Y_train,validation_data=(x_test,y_test),batch_size=batch_size,epochs=epochs,verbose=1,shuffle=True,)	
# 第一個

# 第二個

# 第二個

# 第三個

# 第三個
