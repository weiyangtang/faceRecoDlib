'''
opencv 人脸检测，dlib 提取人脸特征点，优化了官网的demo
'''
import time

import cv2
import dlib
import numpy
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from util import opencvChineseUtils
from util.mysqlUtil import mysqlUtil

predictor_path = 'data/data_dlib/shape_predictor_68_face_landmarks.dat'
face_rec_model_path = 'data/data_dlib/dlib_face_recognition_resnet_model_v1.dat'
path_features_known_csv = 'data/features_all.csv'

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)
facerec = dlib.face_recognition_model_v1(face_rec_model_path)

# opencv 的人脸识别分类器
classifier = cv2.CascadeClassifier('data/data_dlib/haarcascade_frontalface_default.xml')


class face_reco(object):
    def __init__(self):
        self.known_name, self.face_features, self.face_base64 = self.load_faces_data()

    # 载入已录入的人脸数据
    def load_faces_data(self):
        mysqlConfig = {'host': '47.100.78.117', 'passwd': 'TANG123456', 'user': 'root', 'database': 'videosurveillance',
                       'port': 3306}
        mysqlUtils = mysqlUtil(mysqlConfig)
        sql = 'select * from face;'
        record = mysqlUtils.executeSql(sql)
        known_name = []
        face_features = []
        face_base64 = []
        for i in range(len(record)):
            known_name.append(record[i][1])
            # face_features.append(record[i][2])
            face_feature_str = record[i][2].split(',')
            face_feature = []
            for j in range(len(face_feature_str)):
                face_feature.append(float(face_feature_str[j]))
            face_features.append(face_feature)
            face_base64.append(record[i][3])
        print(known_name)
        print(face_features)
        print(len(face_features))
        print(face_base64)
        return known_name, face_features, face_base64

    def cv2ImgAddText(img, text, left, top, textColor=(0, 255, 0), textSize=20):
        if (isinstance(img, numpy.ndarray)):  # 判断是否OpenCV图片类型
            img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img)
        fontText = ImageFont.truetype(
            "font/simsun.ttc", textSize, encoding="utf-8")
        draw.text((left, top), text, textColor, font=fontText)
        return cv2.cvtColor(numpy.asarray(img), cv2.COLOR_RGB2BGR)

    # 计算两个128D向量间的欧式距离
    # Compute the e-distance between two 128D features
    def return_euclidean_distance(self, feature_1, feature_2):
        print(type(feature_1), type(feature_2))
        feature_1 = np.array(feature_1)
        feature_2 = np.array(feature_2)
        dist = np.sqrt(np.sum(np.square(feature_1 - feature_2)))

        dot = np.sum(np.multiply(feature_1, feature_1))
        norm = np.linalg.norm(feature_1) * np.linalg.norm(feature_1)
        res = dot / norm
        print('余弦相似度为%f' % res)
        print('欧式距离为%f' % dist)

        return dist

    def frame_face_reco(self, frame):
        image_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # opencv 人脸检测
        faces = classifier.detectMultiScale(image_gray, scaleFactor=1.2, minNeighbors=3, minSize=(64, 64))
        if len(faces) > 0:
            for face in faces:
                x, y, w, h = face
                face_rectangle = dlib.dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
                shape = predictor(frame, face_rectangle)
                # 人脸特征点
                face_descriptor = facerec.compute_face_descriptor(frame, shape)
                print(type(face_descriptor))
                e_distance_list = []
                for i in range(len(self.known_name)):
                    e_distance_tmp = self.return_euclidean_distance(face_descriptor, self.face_features[i])
                    e_distance_list.append(e_distance_tmp)
                similarity_person_index = e_distance_list.index(min(e_distance_list))
                print('max similarity person is', self.known_name[similarity_person_index], ' min distance is ',
                      e_distance_list[similarity_person_index])
                cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)),
                              (0, 255, 255), 3)
                frame = opencvChineseUtils.cv2ImgAddText(frame, self.known_name[similarity_person_index], int(x),
                                                         int(y))
        return frame


if __name__ == '__main__':
    faceReco = face_reco()
    # cap = cv2.VideoCapture(0)
    # frameIndex = 0
    # while cap.isOpened():
    #     success, frame = cap.read()
    #     frameIndex = (frameIndex + 1) % 3
    #     if frameIndex != 0:
    #         continue
    #     frame = faceReco.frame_face_reco(frame)
    #     cv2.waitKey(1)
    #     cv2.imshow("camera", frame)
    # img = io.imread('../data/face/tangweiyang.jpg')
    image_path = r'D:\CodeData\graduate\video_surveillance_dlib\static\uploads\1589368880723.jpg'
    img = cv2.imread(image_path)
    begin_time = time.time()
    frame = faceReco.frame_face_reco(img)
    print(time.time() - begin_time)
    cv2.imshow('face_reco_optimize', frame)
    cv2.imwrite('demo.jpg',frame)
    cv2.waitKey(0)
