'''
视频推流
'''
import platform
import os

# 推流地址
PUSH_FLOW_URL = 'rtmp://127.0.0.1:1935/myapp/20200429102839988?secretKey=20200429102839988-123456'
# 拉流地址
PULL_FLOW_URL = 'rtmp://127.0.0.1:1935/myapp/20200429102839988?secretKey=20200429102839988-456123'


def winVideoPushFlow():
    # cmd = 'ffmpeg -f dshow -rtbufsize 702000k  -i video="HD WebCam" -s 640x360 -g 5 -vcodec libx264 -r 10 -b:v 1000k   -ab 128k -f flv  '
    cmd = 'ffmpeg -re -i D:\output\webcamera.flv -vcodec libx264 -acodec aac -strict -2 -f flv '
    cmd = cmd + PUSH_FLOW_URL
    print('推流命令：', cmd)
    os.system(cmd)


def linuxVideoPushFlow():
    cmd = 'ffmpeg -i /dev/video0 -s 640x360 -vcodec libx264 -g 5  -max_delay 100 -r 5 -b 700000 -b:v 1000k   -ab 128k -f flv  '
    cmd = cmd + PUSH_FLOW_URL
    print('推流命令：', cmd)
    os.system(cmd)


if __name__ == '__main__':
    sys = platform.system()
    if sys == "Windows":
        winVideoPushFlow()
    elif sys == "Linux":
        linuxVideoPushFlow()
