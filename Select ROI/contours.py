import cv2
import numpy as np 
import pandas as pd
import mediapipe as mp
import os
import glob

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh
'''
0104_shints   0105_cin    0105_hsp    0105_jeff   0105_jihong
0105_long     0105_yee    0105_zizu   0106_brian  0106_chi      0106_sunny
'''
#load檔案
who = '0106_brian'
pose = 'Front' #Front or Side
addr_file = 'D:/dataset/light/'+who+'/'+who+'/'+pose+'/RGB_60FPS_MJPG/'
dirs = os.listdir(addr_file)
for name in dirs:
  if os.path.splitext(name)[1] == ".avi":
      avi_name = name
video_addr = addr_file+avi_name
cap = cv2.VideoCapture(video_addr)

def Detect_face(camera_idx):
  with mp_face_mesh.FaceMesh(
      max_num_faces=1,
      refine_landmarks=True,
      min_detection_confidence=0.5,
      min_tracking_confidence=0.5) as face_mesh:
    while cap.isOpened():
      if (cap.isOpened()== False): 
        print("video record done or error")
      
      success, image = cap.read()

      image = cv2.flip(image, 0) # 上下垂直翻轉
      if not success:
        print("Ignoring empty camera frame.")
        # If loading a video, use 'break' instead of 'continue'.
        continue

      # To improve performance, optionally mark the image as not writeable to
      # pass by reference.
      image.flags.writeable = False
      image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
      image0 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
      image_g = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
      im = np.zeros(image0.shape[:2],dtype = "uint8")
      results = face_mesh.process(image)

      # Draw the face mesh annotations on the image.
      image.flags.writeable = True
      image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
      if results.multi_face_landmarks:
          hx1 =int(results.multi_face_landmarks[0].landmark[109].x * 1280)
          hy1 =int(results.multi_face_landmarks[0].landmark[109].y * 720)
          hx2 =int(results.multi_face_landmarks[0].landmark[338].x * 1280)
          hy2 =int(results.multi_face_landmarks[0].landmark[338].y * 720)
          hx3 =int(results.multi_face_landmarks[0].landmark[296].x * 1280)
          hy3 =int(results.multi_face_landmarks[0].landmark[296].y * 720)
          hx4 =int(results.multi_face_landmarks[0].landmark[66].x * 1280)
          hy4 =int(results.multi_face_landmarks[0].landmark[66].y * 720)
          points0 = np.array([[hx1, hy1], [hx2, hy2], [hx3, hy3],[hx4, hy4]], np.int32)#多邊形的點
          rfx1 =int(results.multi_face_landmarks[0].landmark[234].x * 1280)
          rfy1 =int(results.multi_face_landmarks[0].landmark[234].y * 720)
          rfx2 =int(results.multi_face_landmarks[0].landmark[118].x * 1280)
          rfy2 =int(results.multi_face_landmarks[0].landmark[118].y * 720)
          rfx3 =int(results.multi_face_landmarks[0].landmark[36].x * 1280)
          rfy3 =int(results.multi_face_landmarks[0].landmark[36].y * 720)
          rfx4 =int(results.multi_face_landmarks[0].landmark[210].x * 1280)
          rfy4 =int(results.multi_face_landmarks[0].landmark[210].y * 720)
          rfx5 =int(results.multi_face_landmarks[0].landmark[58].x * 1280)
          rfy5 =int(results.multi_face_landmarks[0].landmark[58].y * 720)
          points1 = np.array([[rfx1, rfy1], [rfx2, rfy2], [rfx3, rfy3],[rfx4, rfy4], [rfx5, rfy5]], np.int32)#多邊形的點
          lfx1 =int(results.multi_face_landmarks[0].landmark[347].x * 1280)
          lfy1 =int(results.multi_face_landmarks[0].landmark[347].y * 720)
          lfx2 =int(results.multi_face_landmarks[0].landmark[454].x * 1280)
          lfy2 =int(results.multi_face_landmarks[0].landmark[454].y * 720)
          lfx3 =int(results.multi_face_landmarks[0].landmark[288].x * 1280)
          lfy3 =int(results.multi_face_landmarks[0].landmark[288].y * 720)
          lfx4 =int(results.multi_face_landmarks[0].landmark[430].x * 1280)
          lfy4 =int(results.multi_face_landmarks[0].landmark[430].y * 720)
          lfx5 =int(results.multi_face_landmarks[0].landmark[266].x * 1280)
          lfy5 =int(results.multi_face_landmarks[0].landmark[266].y * 720)
          points2 = np.array([[lfx1, lfy1], [lfx2, lfy2], [lfx3, lfy3],[lfx4, lfy4], [lfx5, lfy5]], np.int32)#多邊形的點
          nx1 =int(results.multi_face_landmarks[0].landmark[114].x * 1280)
          ny1 =int(results.multi_face_landmarks[0].landmark[114].y * 720)
          nx2 =int(results.multi_face_landmarks[0].landmark[343].x * 1280)
          ny2 =int(results.multi_face_landmarks[0].landmark[343].y * 720)
          nx3 =int(results.multi_face_landmarks[0].landmark[279].x * 1280)
          ny3 =int(results.multi_face_landmarks[0].landmark[279].y * 720)
          nx4 =int(results.multi_face_landmarks[0].landmark[5].x * 1280)
          ny4 =int(results.multi_face_landmarks[0].landmark[5].y * 720)
          nx5 =int(results.multi_face_landmarks[0].landmark[49].x * 1280)
          ny5 =int(results.multi_face_landmarks[0].landmark[49].y * 720)
          points3 = np.array([[nx1, ny1], [nx2, ny2],[nx3, ny3], [nx4, ny4], [nx5, ny5]], np.int32)#多邊形的點
      

      # Flip the image horizontally for a selfie-view display.
      cv2.imshow('MediaPipe Face Mesh', cv2.flip(image, 1))
      cv2.polylines(image0, pts=[points0], isClosed=True, color=(0,255,0),thickness=3) #畫多邊形
      cv2.polylines(image0, pts=[points1], isClosed=True, color=(0,255,0),thickness=3) #畫多邊形
      cv2.polylines(image0, pts=[points2], isClosed=True, color=(0,255,0),thickness=3) #畫多邊形
      cv2.polylines(image0, pts=[points3], isClosed=True, color=(0,255,0),thickness=3) #畫多邊形
      mask = im
      cv2.fillPoly(im, pts=[points0],color=(255,255,255))
      cv2.fillPoly(im, pts=[points1],color=(255,255,255))
      cv2.fillPoly(im, pts=[points2],color=(255,255,255))
      cv2.fillPoly(im, pts=[points3],color=(255,255,255))
      masked = cv2.bitwise_and(image,image,mask=im)
      cv2.imshow("mask",cv2.flip(masked, 1))

            
    #   cv2.imshow('Forehead',cv2.flip(mask[hy1:hy2,hx1:hx2],1))
      cv2.imshow('Forehead',cv2.flip(masked[ny1:ny3,nx1:nx3],1))       # 114x,114y,279x,279y
      cv2.imshow('right face',cv2.flip(masked[rfy2:rfy4,rfx1:rfx3],1)) # rfx1,rfy2,rfx3,rfy4
      cv2.imshow('left face',cv2.flip(masked[lfy1:lfy4,lfx5:lfx2],1))  # lfx5,lfy1,lfx2,lfy4
      cv2.imshow('nose',cv2.flip(masked[hy1:hy3,hx4:hx3],1))           # 66x,109y,296x,296y

    #   rgb_frame_h_g = cv2.split(cv2.flip(mask[hy1:hy2,hx1:hx2],1))[1]
      #rgb_frame_rf_g = cv2.split(cv2.flip(mask[rfy1:rfy2,rfx1:rfx2],1))[1]

      if cv2.waitKey(5) & 0xFF == 27:
        break 

        

if __name__=='__main__':
  Detect_face(video_addr)
  
