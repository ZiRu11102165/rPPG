import cv2
import numpy as np 
import dlib

hog_face_detector = dlib.get_frontal_face_detector()
dlib_facelandmark = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

#load檔案
who = '0104_shints'
addr_file = 'D:/dataset/light/'+who+'/'+who+'/Front/RGB_60FPS_MJPG/'
video_name = 'output_0104_073707_PM_60FPS.avi'
video_addr = addr_file + video_name
cap = cv2.VideoCapture(video_addr)

def Detect_face(camera_idx):
  while True:
      ret, frame = cap.read()
      frame = cv2.flip(frame, 0) # 上下垂直翻轉
      gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
      faces = hog_face_detector(gray,0)
      for i in range(len(faces)):
                landmarks = np.matrix([[p.x, p.y] for p in dlib_facelandmark(frame, faces[i]).parts()])
                for idx, point in enumerate(landmarks):
                    pos = (point[0, 0], point[0, 1])
                    # print(idx, pos)
                    cv2.circle(frame, pos, 1, color=(0, 255, 0))
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(frame, str(idx + 1), pos, font, 0.4, (0, 255, 255), 1, cv2.LINE_AA)


      cv2.imshow("Face Landmarks", frame)

      key = cv2.waitKey(1)
      if key == 27:
          break
  cap.release()
  cv2.destroyAllWindows()

if __name__=='__main__':
  Detect_face(video_addr)
