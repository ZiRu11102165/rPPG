import cv2
import numpy as np 
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

#load檔案
who = '0105_jeff'
addr_file = 'D:/dataset/light/'+who+'/'+who+'/Front/RGB_60FPS_MJPG/'
video_name = 'output_0105_051537_PM_60FPS.avi'
video_addr = addr_file + video_name
cap = cv2.VideoCapture(video_addr)

# # 紀錄點
# h_list  = [66,   65, 222, 107,  55, 221,   9,   8, 193, 168, 336, 285, 417, 296, 295, 442, 441                                             ]
# rf_list = [234,  93, 132, 227, 137, 177, 116, 123, 147, 117,  50, 187, 118, 205, 119, 101, 100, 36                                         ]
# lf_list = [454, 323, 361, 447, 366, 401, 345, 352, 376, 346, 441, 280, 347, 425, 330, 266, 348, 329                                        ]
# n_list  = [217, 198, 131, 174, 236, 134, 220, 196,   2,  51,  45, 197, 195,   5,   4, 419, 248, 281, 275, 399, 456, 363, 440, 437, 420, 360]

def Detect_face(camera_idx):
  drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
  with mp_face_mesh.FaceMesh(
      max_num_faces=1,
      refine_landmarks=True,
      min_detection_confidence=0.5,
      min_tracking_confidence=0.5) as face_mesh:
    while cap.isOpened():
      success, image = cap.read()
      h, w, d = image.shape
      image = cv2.flip(image, 0) # 上下垂直翻轉
      if not success:
        print("Ignoring empty camera frame.")
        # If loading a video, use 'break' instead of 'continue'.
        continue

      # To improve performance, optionally mark the image as not writeable to
      # pass by reference.
      image.flags.writeable = False
      image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
      results = face_mesh.process(image)

      # Draw the face mesh annotations on the image.
      image.flags.writeable = True
      image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
      if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
          # mp_drawing.draw_landmarks(
          #     image=image,
          #     landmark_list=face_landmarks,
          #     connections=mp_face_mesh.FACEMESH_TESSELATION,
          #     landmark_drawing_spec=None,
          #     connection_drawing_spec=mp_drawing_styles
          #     .get_default_face_mesh_tesselation_style())
          hx1 =int(results.multi_face_landmarks[0].landmark[109].x * 1280)
          hy1 =int(results.multi_face_landmarks[0].landmark[109].y * 720)
          hx2=int(results.multi_face_landmarks[0].landmark[296].x * 1280)
          hy2=int(results.multi_face_landmarks[0].landmark[296].y * 720)
          
          rfx1 =int(results.multi_face_landmarks[0].landmark[227].x * 1280)
          rfy1 =int(results.multi_face_landmarks[0].landmark[227].y * 720)
          rfx2=int(results.multi_face_landmarks[0].landmark[142].x * 1280)
          rfy2=int(results.multi_face_landmarks[0].landmark[142].y * 720)
          print(hx1,hy1)
          cv2.imshow('Forehead',image[hy1:hy2,hx1:hx2])
          cv2.imshow('right face',image[rfy1:rfy2,rfx1:rfx2])
          

      # Flip the image horizontally for a selfie-view display.
      cv2.imshow('MediaPipe Face Mesh', cv2.flip(image, 1))
      if cv2.waitKey(5) & 0xFF == 27:
        break

if __name__=='__main__':
  Detect_face(video_addr)
