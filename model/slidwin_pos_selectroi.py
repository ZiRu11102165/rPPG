# 1.POS
# 2.normalize(sliding window=30)
# 3.filter
# 4.moving average(sliding window=9)
# 5.peak detection

import cv2
import numpy as np
import matplotlib.pyplot as plt
import read_csv
import pandas as pd
import selectroi
import os
import matplotlib
matplotlib.use('TKAgg')


buffer_size = 32
n=0
m=n-31

Bh_list=[]
Bf_list=[]
Gh_list=[]
Gf_list=[]
Rh_list=[]
Rf_list=[]

alphalist=[]
psealphalist=[]

hs1=[]
hs2=[]
fs1=[]
fs2=[]
i=0
s1=[]
s2=[]

hspposhlist=[]
hspposflist=[]

H=0
F=0

psealphalist_nor = []
norlast = []
psealphalist_mean=[]
psealphalist_sum=[]
meanlist=[]
peaks=[]

'''
0104_shints   0105_cin    0105_hsp    0105_jeff   0105_jihong
0105_long     0105_yee    0105_zizu   0106_brian  0106_chi      0106_sunny
'''
#load檔案
who = '0106_sunny'
pose = 'Front' #Front or Side
addr_file = 'D:/dataset/light/'+who+'/'+who+'/'+pose+'/RGB_60FPS_MJPG/'
dirs = os.listdir(addr_file)
for name in dirs:
  if os.path.splitext(name)[1] == ".avi":
      avi_name = name
video_addr = addr_file+avi_name

roi_out = selectroi.Detect_face(video_addr)
# print(roi_out)
cap = cv2.VideoCapture(video_addr)

# plt.ion()
plt.figure(figsize=(20,10))#會產生視窗所以要擺迴圈外
def OutputCSV():   
    Result = ('D:/dataset/datasetcsv/'+who+'_Forehead.csv')

    df_SAMPLE = pd.DataFrame.from_dict( meanlist )
    df_SAMPLE.to_csv( Result  , index=False )

    print( '成功產出'+Result )
    print("ttttttttttttttttttttt") 

 # --------------------------------產出CSV檔-----------------------------------------
def z_score(data):
    data=np.array(data)
    mu = data.mean()
    #標準差
    std = data.std()
    #標準化後之結果
    z_score_normalized = (data - mu) / std
    return list(z_score_normalized)

while 1:
    
    ok,vid=cap.read()
    n=n+1
    
    if(n>84):
        del Bh_list[0] #維持長度//del:刪掉第_個位置+該位置的值
        del Gh_list[0]
        del Rh_list[0]
        
        del Bf_list[0]
        del Gf_list[0]
        del Rf_list[0]
        
        # Bh_list[1:]=Bh_list[:-1]#??????????????????????????
        # Bh_list[0]=Rh.sum()/(np.count_nonzero(Rh)+1)#?????????????????????????????
        
    Bh,Gh,Rh = cv2.split(vid[roi_out[0]:roi_out[1],roi_out[2]:roi_out[3]])#cv2.split:將圖片猜成三個通道  額頭
    # Bf,Gf,Rf = cv2.split(vid[284:324,818:888])
    # Bh,Gh,Rh = cv2.split(vid[400:500,700:850])#cv2.split:將圖片猜成三個通道
    Bf,Gf,Rf = cv2.split(vid[roi_out[4]:roi_out[5],roi_out[6]:roi_out[7]])# 右臉頰
    Bh_list.append(np.mean(Bh))
    Bf_list.append(np.mean(Bf))
    Gh_list.append(np.mean(Gh))
    Gf_list.append(np.mean(Gf))
    Rh_list.append(np.mean(Rh))
    Rf_list.append(np.mean(Rf))
    if(n>84):
        zBh = z_score(Bh_list)
        zBf = z_score(Bf_list)
        zGh = z_score(Gh_list)
        zGf = z_score(Gf_list)
        zRh = z_score(Rh_list)
        zRf = z_score(Rf_list)
            
    #算標準差(濾波)---------------
        hs1=(np.array(zGh) - np.array(zBh))
        hs2=(-2 * np.array(zRh) + np.array(zGh) + np.array(zBh))
        hs1_f=read_csv.butter_bandpass_filter(hs1, 0.5, 3, 60, order=5)
        hs2_f=read_csv.butter_bandpass_filter(hs2, 0.5, 3, 60, order=5)
        
        fs1=(np.array(zGf) - np.array(zBf))
        fs2=(-2 * np.array(zRf) + np.array(zGf) + np.array(zBf))
        fs1_f=read_csv.butter_bandpass_filter(fs1, 0.5, 3, 60, order=5)
        fs2_f=read_csv.butter_bandpass_filter(fs2, 0.5, 3, 60, order=5)
            
        #分母0會報錯，要注意!!!
        #head-----------
        if(np.std(hs2_f)>0):
            alpha = hs1_f + hs2_f*(np.std(hs1_f)/np.std(hs2_f))
            # alphalist = alpha.tolist()
        else:
            alpha = 0
        
        #pseuedo 8th line--------------
        H=H+(alpha-np.mean(alpha))
        psealphalist = H.tolist()
        
        s1.append(H[-1])#把list最後一個值放進去********
        halphalistwf=read_csv.butter_bandpass_filter(s1, 0.5, 3, 60, order=5)
        # halphalist = alpha.tolist()   
        # plt.plot(psealphalist)
        
         #face-------------
        if(np.std(fs2_f)>0):
            alpha = fs1_f + fs2_f*(np.std(fs1_f)/np.std(fs2_f))
            # alphalist = alpha.tolist()
        else:
            alpha = 0
        
        F=F+(alpha-np.mean(alpha))
        psealphalist = F.tolist()
        
        s2.append(F[-1])
        falphalistwf=read_csv.butter_bandpass_filter(s2, 0.5, 3, 60, order=5)
        # falphalist = alpha.tolist()
        
        
    #step2.normalize(sliding window=30)-----------------------------
        if len(s1)>(30*60*5-2):
            for m in range(len(s1)):
                if(m<31):
                    psealphalist_nor = (z_score(s1[0:30]))
                else:
                    norlast = z_score(s1[m-30:m])
                    psealphalist_nor.append(norlast[-1])
    #step3.filter----------------------------------------------------
            psealphalist_fir=read_csv.butter_bandpass_filter(psealphalist_nor, 0.5, 3, 60, order=5)
    
    #step4.moving average(sliding window=9)--------------------------        
            for a in range(len(psealphalist_nor)):
                if(a<10):
                    psealphalist_mean=np.mean(psealphalist_fir[:a])
                else:
                    # psealphalist_sum.append(psealphalist_fir[a-9:a])
                    psealphalist_mean=np.mean(psealphalist_fir[a-9:a])
                meanlist.append(psealphalist_mean)
                
    # step5.peak detection--------------------------------------------
            for i in range(len(meanlist)):
                if i>1:
                    if(meanlist[i-1]>meanlist[i-2] and meanlist[i]<meanlist[i-1] and meanlist[i-1]>0):#順向跑圖 ...i-2,i-1,i (i-1是peak)
                        peaks.append(i-1)
            del peaks[0]
            for i in peaks:
                plt.plot(i,meanlist[i] , "x", markersize=10,color='r')
            
        # --------------------------------產出CSV檔-----------------------------------------
            OutputCSV()
            
            # print (psealphalist_mean)
            plt.plot(meanlist)
            plt.show()
            # print("nnnnnnnnnnnnnnnnnnnnnn")

                 
