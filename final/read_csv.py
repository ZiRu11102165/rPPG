import csv
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import signal

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


# 開啟 CSV 檔案
def read_RawData(path):
    try:
        with open(path, newline='') as csvfile:

            # 讀取 CSV 檔案內容
            output=[]
            rows = csv.reader(csvfile)  

            # 以迴圈輸出每一列
            for row in rows:
                results = [ i for i in row ]

                output.append(results)
        print("read rawdata finished")
        return output
    except:
        return None
# def read_set(path):
#     try:
#         with open(path, newline='') as csvfile:

#             # 讀取 CSV 檔案內容
#             output=[]
#             rows = csv.reader(csvfile)

#             # 以迴圈輸出每一列
#             for row in rows:
#                 results = [ i for i in row ]
#                 output.append(results)
#                 print(output)
#         print("read setting finished")
#         return output
#     except:
#         return None
        
def display(data):
    for i in range(len(data)):
        img=np.reshape(data[i],(120,160))
        img_8bit = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        cv2.imshow("PreviewThermal", img_8bit)

            # break when q is pressed
        key = cv2.waitKey(1)
        if key == ord("q"):
            break
if __name__ == "__main__":

    # coll=[9,7,0,9,9,9,9,9,6,9]
    # coll.append(898)
    # print(len(coll))
    # print(coll)



    coll=[]
    colr=[]
    peaks=[]

    a=read_RawData(r'C:\Users\ASUS\OneDrive - 國立臺灣科技大學\桌面\旺\light_0104_shints__Front_h_G.csv')
    # print(len(a[0]))
    for i in range(len(a)):
        print(a[i][0])
        coll.append(float(a[i][0]))
        colr.append(float(a[i][1]))
    for i in range(len(colr)):
        if i>0 and i<len(colr)-1:
            if(colr[i-1]<colr[i] and colr[i+1]<colr[i]):
                peaks.append(i)
                # print(coll[i])

    # while(colr[i-1]<colr[i] and colr[i+1]<colr[i]):
    #     print(colr)
    # while True:
    #     print('3')

    # print(coll)
    # print(colr)     
    print(peaks)
    print("0324")



    
    #fs=訊號取樣頻率(samplerate)
    colr_filter=butter_bandpass_filter(data=colr, lowcut=0.1, highcut=0.5, fs=10, order=4)
    plt.subplot(211)
    plt.plot(coll,colr)
    for i in peaks:
        print(i)
        print("0614")
        plt.plot(coll[i], colr[i], "x", markersize=8,color='r')
    plt.subplot(212)
    plt.plot(coll,colr_filter)
    plt.show()


    
