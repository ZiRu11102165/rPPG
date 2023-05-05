import os
import cv2
from PIL import Image

def load(name):
    #load檔案
    who = name
    pose = 'Front' #Front or Side
    addr_file = 'D:/dataset/light/'+who+'/'+who+'/'+pose+'/RGB_60FPS_MJPG/'
    dirs = os.listdir(addr_file)
    for name in dirs:
        if os.path.splitext(name)[1] == ".avi":
            avi_name = name
    video_addr = addr_file+avi_name
    return video_addr

def save_img(name):
    video_path = load(name)
    vc = cv2.VideoCapture(video_path) 
    c=0
    rval=vc.isOpened()

    while rval:  
        if c<3601:
            c = c + 1
            print(c)
            rval, frame = vc.read()
            pic_path = './our60to10/'+name+'/'
            if rval:
                cv2.imwrite(pic_path + str(c) + '.jpg', frame)
                cv2.waitKey(1)
            else:
                break
        else:
            break
    vc.release()
    print('save_success')


'''
0104_shints   0105_cin    0105_hsp    0105_jeff   0105_jihong
0105_long     0105_yee    0105_zizu   0106_brian  0106_chi      0106_sunny
'''
if __name__ == '__main__':
    name = ['0105_cin','0105_hsp','0105_jeff','0105_jihong','0105_long', '0105_yee', '0105_zizu', '0106_brian', '0106_chi', '0106_sunny']
    for i in name:
        print(i)
        save_img(i)

