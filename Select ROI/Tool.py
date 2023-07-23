import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from scipy import signal
import mediapipe as mp


##########################看檔案用的##########################
def read_csv(path):
    data = pd.read_csv(path,sep=",",encoding='utf-8')
    return data

def draw_two_wave(data1,data2):
    # 創建兩子圖，共享X軸
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    # 繪製PPG
    ax1.plot(data1, 'r')
    ax1.set_ylabel('data1')
    # ax1.set_ylim([0, 100])  # 设置 y1 的显示范围
    # 繪製EKG
    ax2.plot(data2, 'b')
    ax2.set_ylabel('data2')
    # ax2.set_ylim([-1.2, 1.2])  # 设置 y2 的显示范围
    # 標題
    fig.suptitle('Compare Two Data')
    # 兩圖間距
    plt.subplots_adjust(wspace=0.3)
    # 显示图表
    plt.show()
    
def read_avi(path):
    cap = cv2.VideoCapture(path) # 讀取電腦中的影片,開相機太慢(cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    while cap.isOpened():
        success, frame = cap.read()             # 讀取影片的每一幀
        # frame = cv2.flip(frame, 0) # 上下垂直翻轉,用影片的時候才要
        if not success:
            print("Cannot receive frame")   # 如果讀取錯誤，印出訊息
            break
        cv2.imshow('oxxostudio', frame)     # 如果讀取成功，顯示該幀的畫面
        if cv2.waitKey(5) & 0xFF == 27:
            break
    cap.release()                           # 所有作業都完成後，釋放資源
    cv2.destroyAllWindows()                 # 結束所有視窗

##########################看檔案用的##########################

####################################################處理影片用的####################################################
def mask(frame, point):
    mask_image = np.zeros_like(frame)   # 建立一個與 frame 尺寸相同的黑色遮罩影像
    point = np.array(point, dtype=np.int32)
    cv2.fillPoly(mask_image, [point], (255, 255, 255))  # 在遮罩上繪製白色多邊形，多邊形的範圍就是 point 列表中的座標點
    masked_frame = cv2.bitwise_and(frame, mask_image)   # 只保留mask內的區域
    # cv2.imshow('Masked Frame', masked_frame)
    return masked_frame


def calculate_non_zero_means(masked_frame):
    # 計算影像中非零像素的數量
    num_non_zero_pixels = cv2.countNonZero(cv2.cvtColor(masked_frame, cv2.COLOR_BGR2GRAY))
    B, G, R = cv2.split(cv2.flip(masked_frame, 1))  # 分割全部影像的 B、G、R 通道
    # 計算區塊中各個顏色通道的總和
    sum_b = B.sum()
    sum_g = G.sum()
    sum_r = R.sum()
    # 計算區塊中各個顏色通道的平均值（如果非零像素為 0，避免除以 0）
    means_b = sum_b / num_non_zero_pixels if num_non_zero_pixels > 0 else 0
    means_g = sum_g / num_non_zero_pixels if num_non_zero_pixels > 0 else 0
    means_r = sum_r / num_non_zero_pixels if num_non_zero_pixels > 0 else 0

    return means_b, means_g, means_r

def mp_roi_hand(frame, points): # 記得要給point_indexes = [9, 1, 0]
    # 將 BGR 轉換成 RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 偵測手掌
    mp_hands_module = mp.solutions.hands
    hands = mp_hands_module.Hands()
    results = hands.process(frame_rgb)

    # 取得影像的寬度和高度
    height, width, _ = frame.shape

    detected_points = []

    if results.multi_hand_landmarks:
        # 取得第一個偵測到的手掌
        hand_landmarks = results.multi_hand_landmarks[0]

        # 取得想要註記的點的座標
        for point_index in points:
            point_x = int(hand_landmarks.landmark[point_index].x * width)
            point_y = int(hand_landmarks.landmark[point_index].y * height)
            detected_points.append((point_x, point_y))

    return detected_points

def mp_roi_face(frame, points):
    # 將 BGR 轉換成 RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 偵測臉部特徵點
    mp_face_mesh_module = mp.solutions.face_mesh
    face_mesh = mp_face_mesh_module.FaceMesh()
    results = face_mesh.process(frame_rgb)

    # 取得影像的寬度和高度
    height, width, _ = frame.shape

    detected_points = []

    if results.multi_face_landmarks:
        # 取得第一個偵測到的臉部
        face_landmarks = results.multi_face_landmarks[0]

        # 取得想要註記的點的座標
        for point_index in points:
            point_x = int(face_landmarks.landmark[point_index].x * width)
            point_y = int(face_landmarks.landmark[point_index].y * height)
            detected_points.append((point_x, point_y))

    return detected_points

####################################################處理影片用的####################################################

##############################################################################處理訊號用的##############################################################################

# #*******************filter**********************           
def butter_bandpass(lowcut, highcut, fs, order):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return b, a
def butter_bandpass_filter(data, lowcut, highcut, fs, order):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = signal.lfilter(b, a, data)
    return y
#******************normalize********************
def z_score(data):
    data=np.array(data)
    mu = data.mean()
    #標準差
    std = data.std()
    #標準化後之結果
    z_score_normalized = (data - mu) / std
    return list(z_score_normalized)
def normal(data):
    head_alphalist_nor=[]
    for i in range(len(data)):
        if(i < 31):
            head_alphalist_nor = (z_score(data[0:30]))
        else:
            head_norlast = z_score(data[i-30:i])
            head_alphalist_nor.append(head_norlast[-1])
    return head_alphalist_nor
#**************moving avearage***************
def movingave(data):
    fps = 30
    meanlist=[]
    for a in range(len(data)):
        if(a<fps):
            psealphalist_mean=np.mean(data[:a])
        else:
            psealphalist_mean=np.mean(data[a-fps-1:a])
        meanlist.append(psealphalist_mean)
    return meanlist

def rPPG_Chromance(Bh_list, Gh_list, Rh_list):

    time_int_r = np.array(Rh_list)
    time_int_g = np.array(Gh_list)
    time_int_b = np.array(Bh_list)

    normalized_r = (time_int_r - np.mean(time_int_r)) / np.std(time_int_r)
    normalized_g = (time_int_g - np.mean(time_int_g)) / np.std(time_int_g)
    normalized_b = (time_int_b - np.mean(time_int_b)) / np.std(time_int_b)


    xs = 3 * normalized_r - 2 * normalized_g
    ys = 1.5 * normalized_r + normalized_g - 1.5 * normalized_b
    b, a = signal.butter(4, [0.7, 3.4], 'bandpass', fs=30, output='ba')
    # xf = np.array(xs)
    # yf = np.array(ys)
    xf = signal.filtfilt(b, a, xs)
    yf = signal.filtfilt(b, a, ys)

    x_std = np.std(xf)
    y_std = np.std(yf)
    alpha = x_std / y_std
    time_int_r_f = signal.filtfilt(b, a, time_int_r)
    time_int_b_f = signal.filtfilt(b, a, time_int_b)
    time_int_g_f = signal.filtfilt(b, a, time_int_g)

    s = 3 * (1 - alpha / 2) * time_int_r_f - 2 * (1 + alpha / 2)\
        * time_int_g_f + 3 * alpha / 2 * time_int_b_f
    return (s)  

def POS(Bh_list,Gh_list,Rh_list,H):
    fps = 30
    zBh = z_score(Bh_list)
    zGh = z_score(Gh_list)
    zRh = z_score(Rh_list)
    
    lowcut=0.5
    highcut=3
    order=5
    # 算標準差(濾波)---------------
    hs1=(np.array(zGh) - np.array(zBh))
    hs2=(-2 * np.array(zRh) + np.array(zGh) + np.array(zBh))
    hs1_f=butter_bandpass_filter(hs1, lowcut, highcut, fps, order) #read_csv.butter_bandpass_filter(hs1, 0.5, 3, fps, order=5)
    hs2_f=butter_bandpass_filter(hs2, lowcut, highcut, fps, order) #read_csv.butter_bandpass_filter(hs1, 0.5, 3, fps, order=5)
    
    
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

##############################################################################處理訊號用的##############################################################################


# if __name__ == "__main__":    #測試用
#     # 讀取影片
#     video_path = r'C:\Users\USER\Desktop\MOST\IRB\BP_SPO2_dataset\sub_0\Sub_0_face_30fps_2023-07-21 19_48_30.avi'
#     cap = cv2.VideoCapture(video_path)

#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break
#         cv2.imshow('base Frame', frame)
#         # 在這裡設定 point 座標點，這是一個示例，您可以根據需要設定不同的座標點
#         # point = [(397, 189), (471, 66), (417, 24)]
#         point_indexes = [9, 1, 0]
#         detected_points = mp_roi_hand(frame, point_indexes)
#         # 呼叫 mask 函數進行遮罩處理
#         mask(frame, detected_points)

#         if cv2.waitKey(5) & 0xFF == 27:
#             break

#     # 釋放資源
#     cap.release()
#     cv2.destroyAllWindows()
