import pandas as pd
from glob import glob
import os
import numpy as np

# # LOAD TRAINING DATA
# path = 'C:/Users/USER/Desktop/MOST/model/save_light_front_17996/'
# data_y = pd.read_csv('C:/Users/USER/Desktop/MOST/model/test.csv',sep=",",encoding='utf-8')

# dirs = os.listdir(path)

# slidwin_data=[]
# out = []
# X_train=[]
# SBP=[]
# DBP=[]
# Y_train=[]
# for name in dirs:
#     if os.path.splitext(name)[1] == ".csv":
#         csv_name = name
#         who = csv_name.split(sep='_')[1] #分解csv名稱用
#         data_x = pd.read_csv(path+csv_name,sep=",",encoding='utf-8')
#         data_x = data_x.dropna()
#         for row in data_y:
#             if row == who:
#                 bp = data_y[row]
#                 SBP = np.array(bp[0])
#                 DBP = np.array(bp[1])
#                 Y_train.append([bp[0],bp[1]]) 
#                 continue
#         for row in range(250,len(data_x),10):   #sliding window *250*1
#             slidwin_data.append(data_x[row-250:row])

# X_train = np.array(slidwin_data)
# Y_train = np.array(Y_train)

# X_train_s = X_train.shape[0]
# Y_train_s = Y_train.shape[0]
# r = X_train_s/Y_train_s

# Y_train = np.array([val for val in Y_train for _ in range(int(r))])
# print(X_train.shape)
# print(Y_train.shape)
        
######  Martin_dataset_BP  ######
# usecols=['ALLIAS', 'SYSTOLIC', 'DIASTOLIC']
# data_m = pd.read_csv('D:/dataset/physio_dataset_verified/data_collection_record.csv',sep=",",encoding='utf-8', usecols=usecols)
# SBP = np.array(data_m['SYSTOLIC'])
# print(SBP)
# DBP = np.array(data_m['DIASTOLIC'])
# print(DBP)
######  Martin_dataset_BP  ######

######  Martin_dataset  ######

'E:/Martin_dataset/physio_dataset_verified/only_front/'
import numpy as np
import cv2
from cv2 import VideoWriter_fourcc,VideoWriter,imread,resize
from natsort import natsorted
path = 'E:/physio_dataset_verified/only_front/'
dirs = os.listdir(path)
p=[]
name_list = []
img_array = []
for name in dirs:
    name_list.append(name)
# print(name_list)
size = (1920,1080)
fourcc = cv2.VideoWriter_fourcc(*'XVID')

for cont in range(69):
    who = cont
    videowrite = cv2.VideoWriter('./martin_avi/'+name_list[who]+'_30fps.avi',fourcc,30,size)
    read_path = path + name_list[who] + '/rgb/'
    dirs_read = os.listdir(read_path)
    dirs_read.sort(key = lambda x:int(x.split('_')[0]))
    i=0
    for i in range (len(dirs_read)):
        sort_path = read_path + dirs_read[i]
        img = cv2.imread(sort_path)
        # cv2.imshow("windows_name",img)
        cv2.waitKey (0)
        videowrite.write(img)
    cont = cont + 1

######  Martin_dataset  ######

######  our_feature_dataset  ######
# # LOAD TRAINING DATA
# path_feature = 'C:/Users/USER/Desktop/MOST/model/Front_feature/'
# data_y = pd.read_csv('C:/Users/USER/Desktop/MOST/model/test.csv',sep=",",encoding='utf-8')

# dirs = os.listdir(path_feature)

# slidwin_data=[]
# Re_who = []
# X_train=[]
# SBP=[]
# DBP=[]
# Y_train=[]
# for name in dirs:
#     if os.path.splitext(name)[1] == ".csv":
#         csv_name = name
#         who = csv_name.split(sep='_')[1] #分解csv名稱用
#         data_x = pd.read_csv(path_feature+csv_name,sep=",",encoding='utf-8')
#         data_x = data_x.dropna()
        
#         for row in range(100,len(data_x),10):   #sliding window *250*1
#             Re_who.append(who)
#             slidwin_data.append(data_x[row-100:row])
        
# for i in range(len(Re_who)):
#     print(Re_who[i])
#     bp = data_y[Re_who[i]]
#     SBP = bp[0]
#     DBP = bp[1]
#     Y_train.append([SBP,DBP]) 

# X_train = np.array(slidwin_data)
# Y_train = np.array(Y_train)

# print(X_train.shape)
# print(Y_train.shape)

######  our_feature_dataset  ######




