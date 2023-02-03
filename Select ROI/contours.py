import cv2
import numpy as np 
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
who = '0104_shints'
pose = 'Front' #Front or Side
addr_file = 'D:/dataset/light/'+who+'/'+who+'/'+pose+'/RGB_60FPS_MJPG/'
dirs = os.listdir(addr_file)
for name in dirs:
  if os.path.splitext(name)[1] == ".avi":
      avi_name = name
video_addr = addr_file+avi_name
cap = cv2.VideoCapture(video_addr)

def Detect_face(camera_idx):
  drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
  with mp_face_mesh.FaceMesh(
      max_num_faces=1,
      refine_landmarks=True,
      min_detection_confidence=0.5,
      min_tracking_confidence=0.5) as face_mesh:
    while cap.isOpened():
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
      results = face_mesh.process(image)

      # Draw the face mesh annotations on the image.
      image.flags.writeable = True
      image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
      if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
        #   mp_drawing.draw_landmarks(
        #       image=image,
        #       landmark_list=face_landmarks,
        #       connections=mp_face_mesh.FACEMESH_TESSELATION,
        #       landmark_drawing_spec=None,
        #       connection_drawing_spec=mp_drawing_styles
        #       .get_default_face_mesh_tesselation_style())
          hx0  =int(results.multi_face_landmarks[0].landmark[66].x * 1280)
          hy0  =int(results.multi_face_landmarks[0].landmark[66].y * 720*0.95)
          hx1  =int(results.multi_face_landmarks[0].landmark[109].x * 1280)
          hy1  =int(results.multi_face_landmarks[0].landmark[109].y * 720)
          hx2  =int(results.multi_face_landmarks[0].landmark[296].x * 1280) #66x,109y:296x,66y
          hy2  =int(results.multi_face_landmarks[0].landmark[296].y * 720)
          
          rfx1 =int(results.multi_face_landmarks[0].landmark[116].x * 1280)
          rfy1 =int(results.multi_face_landmarks[0].landmark[116].y * 720)
          rfx2 =int(results.multi_face_landmarks[0].landmark[36].x * 1280)
          rfy2 =int(results.multi_face_landmarks[0].landmark[36].y * 720)

          lfx1 =int(results.multi_face_landmarks[0].landmark[345].x * 1280)
          lfy1 =int(results.multi_face_landmarks[0].landmark[345].y * 720)
          lfx2 =int(results.multi_face_landmarks[0].landmark[266].x * 1280)
          lfy2 =int(results.multi_face_landmarks[0].landmark[266].y * 720)

          nx1 =int(results.multi_face_landmarks[0].landmark[47].x * 1280)
          ny1 =int(results.multi_face_landmarks[0].landmark[47].y * 720)
          nx2 =int(results.multi_face_landmarks[0].landmark[327].x * 1280)
          ny2 =int(results.multi_face_landmarks[0].landmark[327].y * 720)
          
          #y1,y2,x1,x2座標
          h_out  = 'H:y1,y2,x1,x2 =', hy1 ,  hy0,  hx0,  hx2
          rf_out = 'RF:y1,y2,x1,x2=', rfy1, rfy2, rfx1, rfx2
          lf_out = 'LF:y1,y2,x1,x2=', lfy1, lfy2, lfx1, lfx2
          n_out  = 'N:y1,y2,x1,x2 =', ny1 ,  ny2,  nx1,  nx2
          print(h_out)
          print(rf_out)
          print(lf_out)
          print(n_out)
          #用來確定框的範圍
          cv2.rectangle(image0, [hx0,hy1], [hx2,hy0], (255, 0, 0), 2)
          cv2.rectangle(image0, [rfx1,rfy1], [rfx2,rfy2], (255, 0, 0), 2)
          cv2.rectangle(image0, [lfx1,lfy2], [lfx2,lfy1], (255, 0, 0), 2)
          cv2.rectangle(image0, [nx1,ny1], [nx2,ny2], (255, 0, 0), 2)        

      # Flip the image horizontally for a selfie-view display.
      cv2.imshow('MediaPipe Face Mesh', cv2.flip(image, 1))
      #cv2.imshow('0',image_g)
      ret,thresh = cv2.threshold(image_g,127,255,cv2.THRESH_BINARY)
      #cv2.imshow('1',cv2.flip(thresh, 1))
      masked = cv2.bitwise_and(image,image,mask=thresh)
      cv2.imshow('2',cv2.flip(masked, 1))
      
      cv2.imshow('Forehead',cv2.flip(masked[hy1:hy0,hx0:hx2],1))
      cv2.imshow('right face',cv2.flip(masked[rfy1:rfy2,rfx1:rfx2],1))
      cv2.imshow('left face',cv2.flip(masked[lfy1:lfy2,lfx2:lfx1],1))
      cv2.imshow('nose',cv2.flip(masked[ny1:ny2,nx1:nx2],1))

      cv2.imshow('1',cv2.flip(image0, 1))

      if cv2.waitKey(5) & 0xFF == 27:
        break

if __name__=='__main__':
  Detect_face(video_addr)
