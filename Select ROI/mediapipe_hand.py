import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils          # mediapipe 繪圖方法
mp_drawing_styles = mp.solutions.drawing_styles  # mediapipe 繪圖樣式
mp_hands = mp.solutions.hands                    # mediapipe 偵測手掌方法

rppg_check_path = "E:\BP_IRB_Lab_dataset\Sub_99\Sub_99_hand_30fps_2023-07-19 17_28_19.avi"
cap = cv2.VideoCapture(rppg_check_path)

if not cap.isOpened():
    print("無法讀取影片")
    exit()

# 讀取第一幀影片
ret, frame = cap.read()

# 檢查是否成功讀取到第一幀
if ret:
    x = 640
    y = 480
    frame2 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)   # 將 BGR 轉換成 RGB
    results = mp_hands.Hands().process(frame2)        # 偵測手掌
    if results.multi_hand_landmarks:
        point9_x = int(results.multi_hand_landmarks[0].landmark[9].x * x)
        point9_y = int(results.multi_hand_landmarks[0].landmark[9].y * y)
        point1_x = int(results.multi_hand_landmarks[0].landmark[1].x * x)
        point1_y = int(results.multi_hand_landmarks[0].landmark[1].y * y)
        point0_x = int(results.multi_hand_landmarks[0].landmark[0].x * x)
        point0_y = int(results.multi_hand_landmarks[0].landmark[0].y * y)
        center_point = ((point9_x + point0_x) // 2, (point9_y + point0_y) // 2)
        # print(center_point)
        
        x1 = center_point[0] - 50 // 2
        y1 = center_point[1] - 50 // 2
        pt1 = (x1, y1)
        pt2 = (x1 + 50, y1 + 50)
        x2 = point1_x - 30 // 2
        y2 = point1_y - 30 // 2
        pt3 = (x2, y2)
        pt4 = (x2 + 30, y2 + 30)

while cap.isOpened():
    ret, frame = cap.read()  # 讀取下一幀影片

    if not ret:
        break

    # cv2.rectangle(frame, pt1, pt2, (255, 0, 0), 2)  # 繪製在影片上
    # cv2.rectangle(frame, pt3, pt4, (0, 255, 0), 2)  # 繪製在影片上
    # cv2.imshow('rectangle', frame)                  # 繪製在影片上
    cv2.imshow('roi1', frame[y1:y1+50,x1:x1+50])
    cv2.imshow('roi2', frame[y2:y2+30,x2:x2+30])
    B_roi1,G_roi1,R_roi1 = cv2.split(cv2.flip(frame[y1:y1+50,x1:x1+50],1)) 
    B_roi2,G_roi2,R_roi2 = cv2.split(cv2.flip(frame[y2:y2+30,x2:x2+30],1))
    
    # 按下 'q' 鍵結束迴圈
    if cv2.waitKey(5) & 0xFF == 27:
        break

# 釋放資源並關閉視窗
cap.release()
cv2.destroyAllWindows()
