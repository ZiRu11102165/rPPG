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
        
# --------------------------------產出CSV檔-----------------------------------------
def OutputCSV(data,name,date,position):
    Result = ('C:/Users/USER/Desktop/MOST/final/rt_csv/'+date+'_'+name+'_'+position+'.csv')
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
            head_alphalist_nor = (z_score(data[0:30]))
        else:
            head_norlast = z_score(data[i-30:i])
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
    y = signal.lfilter(b, a, data)
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
    hs1_f=read_csv.butter_bandpass_filter(hs1, 0.5, 3, 30, order=5)
    hs2_f=read_csv.butter_bandpass_filter(hs2, 0.5, 3, 30, order=5)
        
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

def mean_nonz(input):  # mean non zero
    # return input.sum()/(np.count_nonzero(input[:, :])+1)
    return input.sum()/(np.count_nonzero(input)+1)


fig = plt.figure()

import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

def main():
    n=0

    Bh_list=[]
    Brf_list=[]
    Gh_list=[]
    Grf_list=[]
    Rh_list=[]
    Rrf_list=[]
    Bn_list=[]
    Blf_list=[]
    Gn_list=[]
    Glf_list=[]
    Rn_list=[]
    Rlf_list=[]

    s1=[]
    s2=[]
    s3=[]
    s4=[]

    H=0
    RF=0
    LF=0
    N=0
    print('請輸入日期：')
    date=input()
    print('請輸入姓名：')
    name=input()
    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as face_mesh:
        cap = cv2.VideoCapture(0)
        while cap.isOpened():
            if (cap.isOpened()== False): 
                print("video record done or error")
            x=1280
            y=720
            success, image = cap.read()
            image = cv2.resize(image,(x, y))   # 改變圖片尺寸,原本(640,  360)
            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                continue

            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image0 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            im = np.zeros(image0.shape[:2],dtype = "uint8")
            results = face_mesh.process(image)

            # Draw the face mesh annotations on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if results.multi_face_landmarks :
                hx1 =int(results.multi_face_landmarks[0].landmark[109].x * x)
                hy1 =int(results.multi_face_landmarks[0].landmark[109].y * y)
                hx2 =int(results.multi_face_landmarks[0].landmark[338].x * x)
                hy2 =int(results.multi_face_landmarks[0].landmark[338].y * y)
                hx3 =int(results.multi_face_landmarks[0].landmark[296].x * x)
                hy3 =int(results.multi_face_landmarks[0].landmark[296].y * y)
                hx4 =int(results.multi_face_landmarks[0].landmark[66].x * x)
                hy4 =int(results.multi_face_landmarks[0].landmark[66].y * y)
                points0 = np.array([[hx1, hy1], [hx2, hy2], [hx3, hy3],[hx4, hy4]], np.int32)#多邊形的點
                rfx1 =int(results.multi_face_landmarks[0].landmark[234].x * x)
                rfy1 =int(results.multi_face_landmarks[0].landmark[234].y * y)
                rfx2 =int(results.multi_face_landmarks[0].landmark[118].x * x)
                rfy2 =int(results.multi_face_landmarks[0].landmark[118].y * y)
                rfx3 =int(results.multi_face_landmarks[0].landmark[36].x * x)
                rfy3 =int(results.multi_face_landmarks[0].landmark[36].y * y)
                rfx4 =int(results.multi_face_landmarks[0].landmark[210].x * x)
                rfy4 =int(results.multi_face_landmarks[0].landmark[210].y * y)
                rfx5 =int(results.multi_face_landmarks[0].landmark[58].x * x)
                rfy5 =int(results.multi_face_landmarks[0].landmark[58].y * y)
                points1 = np.array([[rfx1, rfy1], [rfx2, rfy2], [rfx3, rfy3],[rfx4, rfy4], [rfx5, rfy5]], np.int32)#多邊形的點
                lfx1 =int(results.multi_face_landmarks[0].landmark[347].x * x)
                lfy1 =int(results.multi_face_landmarks[0].landmark[347].y * y)
                lfx2 =int(results.multi_face_landmarks[0].landmark[454].x * x)
                lfy2 =int(results.multi_face_landmarks[0].landmark[454].y * y)
                lfx3 =int(results.multi_face_landmarks[0].landmark[288].x * x)
                lfy3 =int(results.multi_face_landmarks[0].landmark[288].y * y)
                lfx4 =int(results.multi_face_landmarks[0].landmark[430].x * x)
                lfy4 =int(results.multi_face_landmarks[0].landmark[430].y * y)
                lfx5 =int(results.multi_face_landmarks[0].landmark[266].x * x)
                lfy5 =int(results.multi_face_landmarks[0].landmark[266].y * y)
                points2 = np.array([[lfx1, lfy1], [lfx2, lfy2], [lfx3, lfy3],[lfx4, lfy4], [lfx5, lfy5]], np.int32)#多邊形的點
                nx1 =int(results.multi_face_landmarks[0].landmark[114].x * x)
                ny1 =int(results.multi_face_landmarks[0].landmark[114].y * y)
                nx2 =int(results.multi_face_landmarks[0].landmark[343].x * x)
                ny2 =int(results.multi_face_landmarks[0].landmark[343].y * y)
                nx3 =int(results.multi_face_landmarks[0].landmark[279].x * x)
                ny3 =int(results.multi_face_landmarks[0].landmark[279].y * y)
                nx4 =int(results.multi_face_landmarks[0].landmark[5].x * x)
                ny4 =int(results.multi_face_landmarks[0].landmark[5].y * y)
                nx5 =int(results.multi_face_landmarks[0].landmark[49].x * x)
                ny5 =int(results.multi_face_landmarks[0].landmark[49].y * y)
                points3 = np.array([[nx1, ny1], [nx2, ny2],[nx3, ny3], [nx4, ny4], [nx5, ny5]], np.int32)#多邊形的點
            # Flip the image horizontally for a selfie-view display.
            cv2.imshow('MediaPipe Face Mesh', cv2.flip(image, 1))
            
            cv2.fillPoly(im, pts=[points0],color=(255,255,255))
            cv2.fillPoly(im, pts=[points1],color=(255,255,255))
            cv2.fillPoly(im, pts=[points2],color=(255,255,255))
            cv2.fillPoly(im, pts=[points3],color=(255,255,255))
            masked = cv2.bitwise_and(image,image,mask=im)
            # masked_show = cv2.cvtColor(masked, cv2.COLOR_BGR2RGB)
            cv2.imshow('MediaPipe masked_show', cv2.flip(masked, 1))
            # out.write(masked)
            Bn,Gn,Rn = cv2.split(cv2.flip(masked[ny1:ny3,nx1:nx3],1)) 
            Brf,Grf,Rrf = cv2.split(cv2.flip(masked[rfy2:rfy4,rfx1:rfx3],1))   
            Bh,Gh,Rh = cv2.split(cv2.flip(masked[hy1:hy3,hx4:hx3],1))         
            Blf,Glf,Rlf = cv2.split(cv2.flip(masked[lfy1:lfy4,lfx5:lfx2],1))   
            
            if cv2.waitKey(5) & 0xFF == 27:
                print('外部中斷')
                break
            n=n+1
            print(n)
            # if(len(Bh_list)>84):
            if(n>42):
                del Bh_list[0] #維持長度//del:刪掉第_個位置+該位置的值
                del Gh_list[0]
                del Rh_list[0]
                
                del Brf_list[0]
                del Grf_list[0]
                del Rrf_list[0]
                
                del Bn_list[0] #維持長度//del:刪掉第_個位置+該位置的值
                del Gn_list[0]
                del Rn_list[0]
                
                del Blf_list[0]
                del Glf_list[0]
                del Rlf_list[0]
                
            Bh_list.append(np.mean(Bh))
            Brf_list.append(np.mean(Brf))
            Gh_list.append(np.mean(Gh))
            Grf_list.append(np.mean(Grf))
            Rh_list.append(np.mean(Rh))
            Rrf_list.append(np.mean(Rrf))
            
            Bn_list.append(np.mean(Bn))
            Blf_list.append(np.mean(Blf))
            Gn_list.append(np.mean(Gn))
            Glf_list.append(np.mean(Glf))
            Rn_list.append(np.mean(Rn))
            Rlf_list.append(np.mean(Rlf))
            
            # ~~~~~~~~~~POS~~~~~~~~~~
            if(n>42):
                H=POS(Bh_list,Gh_list,Rh_list,H)
                RF=POS(Brf_list,Grf_list,Rrf_list,RF)
                LF=POS(Blf_list,Glf_list,Rlf_list,LF)
                N=POS(Bn_list,Gn_list,Rn_list,N)

                s1.append(H[-1])#把list最後一個值放進去********    
                s2.append(RF[-1])
                s3.append(LF[-1])
                s4.append(N[-1])
            if n>(30*60*1-2): #ori.......
                H_alphalist_fir=read_csv.butter_bandpass_filter(normal(s1), 0.5, 3, 30, order=5)
                RF_alphalist_fir=read_csv.butter_bandpass_filter(normal(s2), 0.5, 3, 30, order=5)
                LF_alphalist_fir=read_csv.butter_bandpass_filter(normal(s3), 0.5, 3, 30, order=5)
                N_alphalist_fir=read_csv.butter_bandpass_filter(normal(s4), 0.5, 3, 30, order=5)
            #step4.moving average(sliding window=9)--------------------------
                # movingave(psealphalist_fir)
                Hmovingave=movingave(H_alphalist_fir)
                RFmovingave=movingave(RF_alphalist_fir)   
                Nmovingave=movingave(N_alphalist_fir)
                LFmovingave=movingave(LF_alphalist_fir) 
                # print(Hmovingave)
                # plt.plot(Hmovingave)
                # plt.show()   
                OutputCSV(Hmovingave,name,date,'Forehead')
                OutputCSV(RFmovingave,name,date,'rightface')
                OutputCSV(LFmovingave,name,date,'leftface')
                OutputCSV(Nmovingave,name,date,'nose')
                
if __name__ == "__main__":
    main()
