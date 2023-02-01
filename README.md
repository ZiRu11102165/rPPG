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
