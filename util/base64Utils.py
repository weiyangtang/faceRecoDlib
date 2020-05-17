# coding: utf-8
# python3
import cv2
import base64
from PIL import Image
from io import BytesIO
import os


def frame2base64(frame):
    img = Image.fromarray(frame)  # 将每一帧转为Image
    output_buffer = BytesIO()  # 创建一个BytesIO
    img.save(output_buffer, format='JPEG')  # 写入output_buffer
    byte_data = output_buffer.getvalue()  # 在内存中读取
    base64_data = base64.b64encode(byte_data)  # 转为BASE64
    return base64_data  # 转码成功 返回base64编码


def generate():
    camera = cv2.VideoCapture(0)
    try:
        # while True:
        ret, frame = camera.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow("camera", frame)
        base64_data = frame2base64(frame)
        print(base64_data)
        # if cv2.waitKey(1) & 0xFF == ord("q"):
        #     # break
    except Exception as e:
        print(e)
    finally:
        # 释放资源
        camera.release()
        cv2.destroyAllWindows()


def image2base64(image_path):
    if os.path.exists(image_path):
        with open(image_path, 'rb') as file:
            base64_data = base64.b64encode(file.read())
            s = base64_data.decode()
            print('data:image/jpeg;base64,%s' % s)
        return 'data:image/jpeg;base64,%s' % s
    return ''


if __name__ == '__main__':
    generate()
    image2base64('data/face/liuni.jpg')
