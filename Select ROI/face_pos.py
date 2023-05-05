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
from scipy import signal
import matplotlib
matplotlib.use('TKAgg')
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np
from keras.models import *
from tensorflow import keras
import pandas as pd 
import tensorflow as tf 
from keras.layers.core import *
from tensorflow.keras.optimizers import *
from keras import metrics
from tensorflow.keras.models import * 
from tensorflow.keras.layers import *
from tensorflow.keras.losses import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from keras import backend as K
import rPPG
import heartpy as hp

# --------------------------------產出CSV檔-----------------------------------------
def OutputCSV(data,name):
    Result = ('C:/Users/USER/Desktop/MOST/slidwin_pos/roiroi/m_10_data/10csv/'+name+'.csv')
    df_SAMPLE = pd.DataFrame.from_dict( data )
    df_SAMPLE.to_csv( Result  , index=False )
    print( '成功產出'+Result )

def z_score(data):
    data=np.array(data)
    mu = data.mean()
    #標準差
    std = data.std()
    #標準化後之結果
    z_score_normalized = (data - mu) / std
    return list(z_score_normalized)

#******************normalize********************
def normal(data):
    head_alphalist_nor=[]
    for i in range(len(data)):
        if(i < 31):
            head_alphalist_nor = (z_score(data[0:10]))
        else:
            head_norlast = z_score(data[i-10:i])
            head_alphalist_nor.append(head_norlast[-1])
    return head_alphalist_nor

#*******************filter**********************           
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = signal.Filter(b, a, data)
    return y

#**************moving avearage***************
def movingave(data):
    meanlist=[]
    # psealphalist_fir=read_csv.butter_bandpass_filter(data, 0.5, 3, 60, order=5)
    for a in range(len(data)):
        if(a<10):
            psealphalist_mean=np.mean(data[:a])
        else:
            psealphalist_mean=np.mean(data[a-9:a])
        meanlist.append(psealphalist_mean)
    return meanlist

def POS(Bh_list,Gh_list,Rh_list,H):
    zBh = z_score(Bh_list)
    zGh = z_score(Gh_list)
    zRh = z_score(Rh_list)
        
#算標準差(濾波)---------------
    hs1=(np.array(zGh) - np.array(zBh))
    hs2=(-2 * np.array(zRh) + np.array(zGh) + np.array(zBh))
    hs1_f=read_csv.butter_bandpass_filter(hs1, 0.5, 3, 10, order=5)
    hs2_f=read_csv.butter_bandpass_filter(hs2, 0.5, 3, 10, order=5)
    
        
    #分母0會報錯，要注意!!!
    #head-----------
    if(np.std(hs2_f)>0):
        alpha = hs1_f + hs2_f*(np.std(hs1_f)/np.std(hs2_f))
        # alphalist = alpha.tolist()
    else:
        alpha = 0
    
    #pseuedo 8th line--------------
    H=H+(alpha-np.mean(alpha))
    return (H)

def rPPG_Chromance(Bh_list, Gh_list, Rh_list):

    time_int_r = np.array(Rh_list)
    time_int_g = np.array(Gh_list)
    time_int_b = np.array(Bh_list)

    normalized_r = (time_int_r - np.mean(time_int_r)) / np.std(time_int_r)
    normalized_g = (time_int_g - np.mean(time_int_g)) / np.std(time_int_g)
    normalized_b = (time_int_b - np.mean(time_int_b)) / np.std(time_int_b)
    # normalized_r = time_int_r
    # normalized_g = time_int_g
    # normalized_b = time_int_b

    xs = 3 * normalized_r - 2 * normalized_g
    ys = 1.5 * normalized_r + normalized_g - 1.5 * normalized_b
    # sos = signal.butter(5, [0.5, 3], 'bandpass', fs=60, output='sos')
    # xf = signal.sosfilt(sos, xs)
    # yf = signal.sosfilt(sos, ys)
    b, a = signal.butter(5, [0.7, 3], 'bandpass', fs=10, output='ba')
    # xf = np.array(xs)
    # yf = np.array(ys)
    xf = signal.filtfilt(b, a, xs)
    yf = signal.filtfilt(b, a, ys)

    # xf = signal.sosfilt(sos, xs)
    # yf = signal.sosfilt(sos, ys)
    # xf = xs
    # yf = ys
    x_std = np.std(xf)
    y_std = np.std(yf)
    alpha = x_std / y_std
    time_int_r_f = signal.filtfilt(b, a, time_int_r)
    time_int_b_f = signal.filtfilt(b, a, time_int_b)
    time_int_g_f = signal.filtfilt(b, a, time_int_g)

    s = 3 * (1 - alpha / 2) * time_int_r_f - 2 * (1 + alpha / 2)\
        * time_int_g_f + 3 * alpha / 2 * time_int_b_f

    return (s)

def mean_nonz(input):  # mean non zero
    # return input.sum()/(np.count_nonzero(input[:, :])+1)
    return input.sum()/(np.count_nonzero(input)+1)

def mae(actual, predictions):
    actual, predictions = np.array(actual), np.array(predictions)
    return np.mean(np.abs(actual - predictions))

def rmse(actual, predictions): 
    actual, predictions = np.array(actual), np.array(predictions)
    return np.sqrt(np.mean(np.square(np.subtract(actual,predictions))))

def sw(data):
    slidwin_data=[]
    for row in range(600,len(data),100):   #sliding window *250*1
        slidwin = data[row-600:row]
        slidwin_data.append(slidwin)
    return slidwin_data

def face_detection(frame, typ='', inputModel='', roiplace='', mask_3=''):
    # return roiplace and mask_3 in original list
    # ///\\\///\\\///\\\///\\\///\\\///\\\///\\\///\\\///\\\///\\\///\\\///\\\///\\\///\\\///\\\
    if typ == 'Skin':
        B, G, R = cv2.split(frame)
        roiframe_HSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        H, S, V = cv2.split(roiframe_HSV)

        mask = rPPG.OTSU_Process(img_b=B,
                                 img_g=G,
                                 img_r=R,
                                 img_h=H,
                                 img_s=S,
                                 img_v=V)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        mask = cv2.erode(mask, kernel)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 13))
        mask = cv2.dilate(mask, kernel)
        mask_3.append(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR))
        roiframe = cv2.bitwise_and(frame, mask_3[0])
        # B, G, R = cv2.split(output)
        # roiplace.append((0, 0, frame.shape[1], frame.shape[0]))
    # ///\\\///\\\///\\\///\\\///\\\///\\\///\\\///\\\///\\\///\\\///\\\///\\\///\\\///\\\///\\\
    elif typ == 'Lab model':
        roiframe = ''
        min_confidence = 0.5
        (h, w) = frame.shape[:2]
        inHeight = 300
        inWidth = 300
        inScaleFactor = 1
        meanVal = 127.5
        # 建立模型使用的Input資料blob (比例變更為300 x 300)
        blob = cv2.dnn.blobFromImage(
            frame, inScaleFactor, (inWidth, inHeight), (104.0, 177.0, 123.0))

        # 設定Input資料與取得模型預測結果
        inputModel.setInput(blob)
        detectors = inputModel.forward()

        # loop所有預測結果
        for i in range(0, detectors.shape[2]):
            # 取得預測準確度
            confidence = detectors[0, 0, i, 2]

            # 篩選準確度低於argument設定的值
            if confidence < min_confidence:
                continue
            # 計算bounding box(邊界框)與準確率 - 取得(左上X，左上Y，右下X，右下Y)的值 (記得轉換回原始image的大小)
            box = detectors[0, 0, i, 3:7] * np.array([w, h, w, h])
            # 將邊界框轉成正整數，方便畫圖
            (x0, y0, x1, y1) = box.astype("int")
            # print(box)
            # roiplace.append((x0, y0, x1 - x0, y1 - y0))
            roiplace.append((x0, y0, x1, y1))
    # ///\\\///\\\///\\\///\\\///\\\///\\\///\\\///\\\///\\\///\\\///\\\///\\\///\\\///\\\///\\\
    else:
        print('pls select facedetection typ')
    return roiframe


fig = plt.figure()

# cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
def RR_1024(RR_1024_list,RR_MA):
    for i in range(1024-1):
        if i<100:
            RR_1024_list[i]=RR_1024_list[i+1]
        else:
            RR_1024_list[i]=0
    RR_1024_list[100-1]=RR_MA
    return RR_1024_list
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = signal.lfilter(b, a, data)
    return y
def main(i):
    # subject_name = ['ades6', 'agus6', 'aice6', 'alana6', 'alex6', 'ali6', 'alina6', 'anggur6', 'apu6', 'aqua6', 'ara6', 'arnold6', 'bunny6', 'cici6', 'citra6', 'dadu6', 'daz6', 'dede6', 'deka6', 'dony6', 'drillyana6', 'eja6', 'fanny6', 'farguso6', 'faye6', 'fitsan6', 'fote6', 'gab6', 'gea6', 'gilberts6', 'gusto6', 'harvey6', 'hello6', 'helly6', 'hexa6', 'ikan6', 'kausar6', 'lea6', 'lili6', 'losory6', 'lyla6', 'mal6', 'murli6', 'nasama6', 'neymar6', 'nick6', 'nusa6', 'ocyra6', 'oren6', 'patrick6', 'peter6', 'phaechan6', 'pulpen6', 'ran6', 'redoqx6', 'robert6', 'rosali6', 'ryuu6', 'sally6', 'sania6', 'sean6', 'syntax6', 'sysi6', 'toro6', 'tryx6', 'vel6', 'yukibara6', 'yusuf6']
    #load檔案
    who = i
    # print(who)
    addr_file = 'C:/Users/USER/Desktop/MOST/slidwin_pos/roiroi/m_10_data/'
    video_addr = addr_file+who+'.avi'
    cap = cv2.VideoCapture(video_addr)
    success, image = cap.read()
    ######人臉偵測用
    roi_posi=[]
    prototxt_rgb = r"./rgb.prototxt"
    caffemodel_rgb = r"./rgb.caffemodel"
    net_rgb = cv2.dnn.readNetFromCaffe(prototxt=prototxt_rgb, caffeModel=caffemodel_rgb)
    ######人臉偵測用
    n=0
    Bf_list=[]
    Gf_list=[]
    Rf_list=[]

    pos_out=[]
    times = []  #畫頻譜用
    # record=[]
    F=0
    face_detection(image, typ='Lab model', inputModel=net_rgb, roiplace=roi_posi)
    x1 = roi_posi[0][0]
    y1 = roi_posi[0][1]
    x2 = roi_posi[0][2]
    y2 = roi_posi[0][3]
    while cap.isOpened():
        # if (cap.isOpened()== False): 
        #     print("video record done or error")
            
        
        # x=1280
        # y=720
        # success, image = cap.read()
        success, image = cap.read()
        # image = cv2.flip(image, 0) # 上下垂直翻轉,用影片的時候才要
        
        # image = cv2.resize(image,(x, y))   # 改變圖片尺寸,原本(640,  360)
        
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        # Flip the image horizontally for a seFie-view display.
        # cv2.imshow('show', cv2.flip(image[y1:y2,x1:x2], 1))
        Bf,Gf,Rf = cv2.split(cv2.flip(image[y1:y2,x1:x2],1))
        n=n+1
        if(n>42):
            
            del Bf_list[0]
            del Gf_list[0]
            del Rf_list[0]
        
        
            
        Bf_list.append(np.mean(Bf))
        Gf_list.append(np.mean(Gf))
        Rf_list.append(np.mean(Rf))
        
        # #-~~~~~~~~~~~~~~POS~~~~~~~~~~~~~~~~
        # if(n>42):
        #     F=POS(Bf_list,Gf_list,Rf_list,F)

        #     # s1.append(H[-1])#把list最後一個值放進去********    
        #     pos_out.append(F[-1])
        #     # s4.append(N[-1])

        #-~~~~~~~~~~~~~~chrom~~~~~~~~~~~~~~~~
        if(n>42):
            F=rPPG_Chromance(Bf_list,Gf_list,Rf_list)
            
            pos_out.append(F[-1])
        
        if n>(10*60*1-2):
            F_alphalist_fir=read_csv.butter_bandpass_filter(normal(pos_out), 0.5, 3, 10, order=5)
        #step4.moving average(sliding window=9)--------------------------
            # movingave(psealphalist_fir)
            Fmovingave=movingave(F_alphalist_fir) 
            print(Fmovingave)
            # plt.plot(Fmovingave)
            # plt.show()            
            
            OutputCSV(Fmovingave,who)
            # #############頻譜圖#############
            # data_count=0
            # PR_1024_list = [0 for i in range(1024)]
            # PR_raw=[]
            # spectrogram=[]
            # times=[]
            # PR_max=[]
            # PR_f_max_loc_f_now=0
            # PR_f_max_loc_f_pr=0

            # freq=np.linspace(0.0, 10*1.0/(2.0), 512)  # 200>>>sample_rate(影片fps), 5120>>>最大???
            # while data_count<len(Fmovingave)-2:
            #     data_count+=1
            #     PR_raw.append(Fmovingave[data_count])
            #     if len(PR_raw)>100:
            #         del PR_raw[0]
            #     if data_count>2:
                    
            #         # PR_nor=signal_process.normalization_PR(PR_raw)
            #         PR_fir = butter_bandpass_filter(
            #                                 PR_raw, 1, 3.4, 10, 4)
            #         PR_1024_list=RR_1024(PR_1024_list,PR_fir[-1])
            #         PR_fft = np.fft.fft(PR_1024_list)
            #         PR_fft_abs=abs(PR_fft)
            #         PR_f_max_loc_f_now=np.argmax(PR_fft_abs[:256])
            #         # PR_f_max_loc_f_pr=PR_f_max_loc_f_now    
            #         PR_max.append(freq[PR_f_max_loc_f_now]*60)
            #         times.append(data_count/10)
            #         spectrogram.append(PR_fft_abs[:256])
            # plt.pcolormesh(times, freq[:256]*60, np.array(spectrogram).T)  # 256>>>每次取多少去計算
            # for i in range(1,len(PR_max),1):
            #     plt.plot(i/10,PR_max[i],marker='.',color='r')
            # plt.show()
            # #############頻譜圖#############
            break
            
        if cv2.waitKey(5) & 0xFF == 27:
            print('外部中斷')
            break         
                
        
if __name__ == "__main__":
    people=['ades6', 'agus6', 'aice6', 'alana6', 'alex6', 'ali6', 'alina6', 'anggur6', 'apu6', 'aqua6', 'ara6', 'arnold6', 'bunny6', 'cici6', 'citra6', 'dadu6', 'daz6', 'dede6', 'deka6', 'dony6', 'drillyana6', 'eja6', 'fanny6', 'farguso6', 'faye6', 'fitsan6', 'fote6', 'gab6', 'gea6', 'gilberts6', 'gusto6', 'harvey6', 'hello6', 'helly6', 'hexa6', 'ikan6', 'kausar6', 'lea6', 'lili6', 'losory6', 'lyla6', 'mal6', 'murli6', 'nasama6', 'neymar6', 'nick6', 'nusa6', 'ocyra6', 'oren6', 'patrick6', 'peter6', 'phaechan6', 'ran6', 'redoqx6', 'robert6', 'rosali6', 'ryuu6', 'sally6', 'sania6', 'sean6', 'syntax6', 'sysi6', 'toro6', 'tryx6', 'vel6', 'yukibara6', 'yusuf6']
    for i in people:
        print('start:'+i)
        main(i)
        



'''
times.append(data_count/20)  #data_count>>>資料總數?
spectrogram.append(PR_fft_abs[:256])  # 256>>>每次取多少去計算
freq=np.linspace(0.0, sample_rate*1.0/(2.0), 5120)  # 200>>>sample_rate(影片fps), 5120>>>最大???
plt.pcolormesh(times, freq[:256]*60, np.array(spectrogram).T)  # 256>>>每次取多少去計算
'''
