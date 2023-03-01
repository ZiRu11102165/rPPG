import cv2
import numpy as np 
import pandas as pd
import mediapipe as mp
import os


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

# #load檔案 Martin的
# # C:/Users/USER/Desktop/MOST/model/martin_avi/
# addr_file = 'C:/Users/USER/Desktop/MOST/model/martin_avi/'
# dirs = os.listdir(addr_file)
# avi_name = []
# who=[]
# for name in dirs:
#   if os.path.splitext(name)[1] == ".avi":
#       who.append(os.path.splitext(name)[0]) 
#       avi_name.append(name)
# # print(avi_name)
# # print(who)
# no = 49 #第幾個影片
# who_name = who[no]
# video_addr = addr_file+avi_name[no] #共69個影片
# # print(who_name)
# # print(video_addr)

# #load檔案 Martin的

# cap = cv2.VideoCapture(video_addr)

def Detect_face(in_addr):
  with mp_face_mesh.FaceMesh(
      max_num_faces=1,
      refine_landmarks=True,
      min_detection_confidence=0.5,
      min_tracking_confidence=0.5) as face_mesh:
    cap = cv2.VideoCapture(in_addr)
    while cap.isOpened():
      if (cap.isOpened()== False): 
        print("video record done or error")
      
      success, image = cap.read()

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

          hx1 =int(results.multi_face_landmarks[0].landmark[109].x * 1920)
          hy1 =int(results.multi_face_landmarks[0].landmark[109].y * 1080)
          hx2 =int(results.multi_face_landmarks[0].landmark[338].x * 1920)
          hy2 =int(results.multi_face_landmarks[0].landmark[338].y * 1080)
          hx3 =int(results.multi_face_landmarks[0].landmark[296].x * 1920)
          hy3 =int(results.multi_face_landmarks[0].landmark[296].y * 1080)
          hx4 =int(results.multi_face_landmarks[0].landmark[66].x * 1920)
          hy4 =int(results.multi_face_landmarks[0].landmark[66].y * 1080)
          points0 = np.array([[hx1, hy1], [hx2, hy2], [hx3, hy3],[hx4, hy4]], np.int32)#多邊形的點
          
          rfx1 =int(results.multi_face_landmarks[0].landmark[234].x * 1920)
          rfy1 =int(results.multi_face_landmarks[0].landmark[234].y * 1080)
          rfx2 =int(results.multi_face_landmarks[0].landmark[118].x * 1920)
          rfy2 =int(results.multi_face_landmarks[0].landmark[118].y * 1080)
          rfx3 =int(results.multi_face_landmarks[0].landmark[36].x * 1920)
          rfy3 =int(results.multi_face_landmarks[0].landmark[36].y * 1080)
          rfx4 =int(results.multi_face_landmarks[0].landmark[210].x * 1920)
          rfy4 =int(results.multi_face_landmarks[0].landmark[210].y * 1080)
          rfx5 =int(results.multi_face_landmarks[0].landmark[58].x * 1920)
          rfy5 =int(results.multi_face_landmarks[0].landmark[58].y * 1080)
          points1 = np.array([[rfx1, rfy1], [rfx2, rfy2], [rfx3, rfy3],[rfx4, rfy4], [rfx5, rfy5]], np.int32)#多邊形的點
          lfx1 =int(results.multi_face_landmarks[0].landmark[347].x * 1920)
          lfy1 =int(results.multi_face_landmarks[0].landmark[347].y * 1080)
          lfx2 =int(results.multi_face_landmarks[0].landmark[454].x * 1920)
          lfy2 =int(results.multi_face_landmarks[0].landmark[454].y * 1080)
          lfx3 =int(results.multi_face_landmarks[0].landmark[288].x * 1920)
          lfy3 =int(results.multi_face_landmarks[0].landmark[288].y * 1080)
          lfx4 =int(results.multi_face_landmarks[0].landmark[430].x * 1920)
          lfy4 =int(results.multi_face_landmarks[0].landmark[430].y * 1080)
          lfx5 =int(results.multi_face_landmarks[0].landmark[266].x * 1920)
          lfy5 =int(results.multi_face_landmarks[0].landmark[266].y * 1080)
          points2 = np.array([[lfx1, lfy1], [lfx2, lfy2], [lfx3, lfy3],[lfx4, lfy4], [lfx5, lfy5]], np.int32)#多邊形的點
          nx1 =int(results.multi_face_landmarks[0].landmark[114].x * 1920)
          ny1 =int(results.multi_face_landmarks[0].landmark[114].y * 1080)
          nx2 =int(results.multi_face_landmarks[0].landmark[343].x * 1920)
          ny2 =int(results.multi_face_landmarks[0].landmark[343].y * 1080)
          nx3 =int(results.multi_face_landmarks[0].landmark[279].x * 1920)
          ny3 =int(results.multi_face_landmarks[0].landmark[279].y * 1080)
          nx4 =int(results.multi_face_landmarks[0].landmark[5].x * 1920)
          ny4 =int(results.multi_face_landmarks[0].landmark[5].y * 1080)
          nx5 =int(results.multi_face_landmarks[0].landmark[49].x * 1920)
          ny5 =int(results.multi_face_landmarks[0].landmark[49].y * 1080)
          points3 = np.array([[nx1, ny1], [nx2, ny2],[nx3, ny3], [nx4, ny4], [nx5, ny5]], np.int32)#多邊形的點
          out=[hy1,hy3,hx4,hx3,rfy2,rfy4,rfx1,rfx3,lfy1,lfy4,lfx5,lfx2,ny1,ny3,nx1,nx3]
          
      #out=[hy1,hy3,hx4,hx3,rfy2,rfy4,rfx1,rfx3,lfy1,lfy4,lfx5,lfx2,ny1,ny3,nx1,nx3]
      

      # Flip the image horizontally for a selfie-view display.
      # cv2.imshow('MediaPipe Face Mesh', cv2.flip(image, 1))
      
      cv2.fillPoly(im, pts=[points0],color=(255,255,255))
      cv2.fillPoly(im, pts=[points1],color=(255,255,255))
      cv2.fillPoly(im, pts=[points2],color=(255,255,255))
      cv2.fillPoly(im, pts=[points3],color=(255,255,255))
      # masked = cv2.bitwise_and(image,image,mask=im)
      # cv2.imshow("mask",cv2.flip(masked, 1))

      # cv2.imshow('h',cv2.flip(masked[hy1:hy3,hx4:hx3],1))
      # cv2.imshow('right face',cv2.flip(masked[rfy2:rfy4,rfx1:rfx3],1)) # rfx1,rfy2,rfx3,rfy4
      # cv2.imshow('left face',cv2.flip(masked[lfy1:lfy4,lfx5:lfx2],1))  # lfx5,lfy1,lfx2,lfy4
      # cv2.imshow('nose',cv2.flip(masked[ny1:ny3,nx1:nx3],1))           # 66x,109y,296x,296y


      if cv2.waitKey(5) & 0xFF == 27:
        break 
      print(out)
      return out
      
# if __name__=='__main__':
#   Detect_face(video_addr)
  
