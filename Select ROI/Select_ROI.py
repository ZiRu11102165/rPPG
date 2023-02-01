import cv2, sys
import numpy as np

#load檔案
who = '0104_shints'
where = 'right_malar' # 眉間+下額頭 glabella_and_lwer_medal_forehead,右頰 right_malar,左頰 left_malar,鼻子 nose

addr_file = 'D:/dataset/light/'+who+'/'+who+'/Front/RGB_60FPS_MJPG/'
video_name = 'output_0104_073707_PM_60FPS.avi'
video_addr = addr_file + video_name
cap = cv2.VideoCapture(video_addr)

cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)
record_name = ('light_'+ who + '_Front_'+ where) #記錄ROI用

# #Our ROI, defined by two points(第一次框選)
# p1, p2 = None, None
# p3, p4 = None, None
# state = 0
# record = 0

# 直接輸入上次數值
p1, p2 = [818, 284], [888, 324] #上次ROI框選範圍數值
p3, p4 = [888, 284], [818, 324] #前大後大
state = -1
record = 1

# Called every time a mouse event happen
def on_mouse(event, x, y, flags, userdata):
    global state, p1, p2,p3,p4
    # Left click
    if event == cv2.EVENT_LBUTTONUP:
        # Select first point
        if state == 0:
            p1 = [x   ,y   ]
            p2 = [x+70,y+40]    #固定方框大小 眉間+下額頭:x+60,y+80,右頰x+70,y+40 ,左頰x+70,y+40 ,鼻子 
            p3 = [x+70,y   ]    #固定方框大小 眉間+下額頭:x+60,y   ,右頰x+70,y    ,左頰x+70,y    ,鼻子
            p4 = [x   ,y+40]    #固定方框大小 眉間+下額頭:x   ,y+80,右頰x   ,y+40 ,左頰x   ,y+40 ,鼻子
            state += 1

    # Right click (刪除 ROI)
    if event == cv2.EVENT_RBUTTONUP:
        p1, p2 = None,None 
        state = 0

# Register the mouse callback
cv2.setMouseCallback('Frame', on_mouse)

if record == 1:
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # out_roi = cv2.VideoWriter(record_name+'.avi', fourcc, 60.0, (1280,720))
    out_roi = cv2.VideoWriter('TEST.avi', fourcc, 60.0, (1280,720))

if (cap.isOpened()== False): 
    print("video record done or error")

while cap.isOpened():
    val, frame = cap.read()
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  #灰階
    #一開始就有數值，就直接畫出
    if state == -1:
        cv2.rectangle(frame, p1, p2, (255, 0, 0), 2)
                             
    # If a ROI is selected, draw it
    if state > 0:
        cv2.rectangle(frame, p1, p2, (255, 0, 0), 2)
        
    if record == 1:
        # Show roi_image
        site = np.array([[p1,p3,p2,p4]],dtype=np.int32) 
        mask = np.zeros(frame.shape[:2],dtype="uint8") #生成全黑畫面
        cv2.polylines(mask,site,1,255)
        cv2.fillPoly(mask,site,255)
        #cv2.imshow("mask",mask)
        masked = cv2.bitwise_and(frame,frame,mask=mask)
        cv2.namedWindow('masked', cv2.WINDOW_NORMAL)
        cv2.imshow("masked",masked)

    elif record == 0:
        out = record_name , p1 , p2
        print(out)    

    # Show image
    cv2.imshow('Frame', frame)
    #save roi_image
    out_roi.write(masked)  

    # B,G,R = cv2.split(cut_frame) #取得影像RGB參數
    # print(B,G,R)

    # Let OpenCV manage window events
    key = cv2.waitKey(50)
    # If ESCAPE key pressed, stop
    if key == 27:
        cap.release()

