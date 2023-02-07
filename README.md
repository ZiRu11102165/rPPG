# rPPG
rPPG signal processing

## selectROI
### 20230131 selectROI.py
![](https://github.com/ZiRu11102165/rPPG/blob/main/picture/selectROI_20230131.png)
目前可框 + 有儲存ROI,目標完成所有ROI座標
### 20230201 extskin.py
因selectROI.py要手動框roi很麻煩，改成用dlib偵測人臉68個點，來嘗試
![](https://github.com/ZiRu11102165/rPPG/blob/main/picture/20230201.png)
但上面的dlib不夠我們使用，因此改使用MediaPipe
![](https://raw.githubusercontent.com/google/mediapipe/master/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png)
下面是直接使用官方程式碼做測試
![20230201MediaPipe](https://user-images.githubusercontent.com/124028666/215977076-c555d452-f077-4900-a6f5-45e2a3599d26.png)
### 20230202 
contours.py 利用灰階與二值化來分割，不確定 訊號會不會有問題 及 仍可能會有沒消除乾淨的 
![image](https://user-images.githubusercontent.com/124028666/216247004-8ca3e855-294d-46cc-bef3-0263bffdb39e.png)
![image](https://user-images.githubusercontent.com/124028666/216502841-c2cc3e98-1bca-420d-898e-c51d19ceddb1.png)

## model
### 20230207
model 
Bi_LSTM (+ attention) complete/None

![image](https://user-images.githubusercontent.com/124028666/217289423-39cf2633-7f0b-4264-a8f9-6e9dbbc10c91.png)

Bi_GRU  (+ attention)  None/None
