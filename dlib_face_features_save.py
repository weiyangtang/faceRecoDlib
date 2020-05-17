'''
遍历人脸文件夹内照片，获取人脸128D特征点保存到csv文件
'''

import dlib
import cv2
import os
import csv
import numpy as np
from skimage import io
from util import base64Utils
from util.mysqlUtil import mysqlUtil

predictor_path = 'data/data_dlib/shape_predictor_68_face_landmarks.dat'
face_rec_model_path = 'data/data_dlib/dlib_face_recognition_resnet_model_v1.dat'
path_features_known_csv = 'data/features_all.csv'

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)
facerec = dlib.face_recognition_model_v1(face_rec_model_path)

images_dir_path = 'data/face/'


def face_features_128D(image_path):
    image = io.imread(image_path)
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    faces = detector(image_gray, 1)

    if len(faces) == 1:
        for face in faces:
            shape = predictor(image, face)
            face_descriptor = facerec.compute_face_descriptor(image_gray, shape)
    else:
        face_descriptor = 0

    return face_descriptor


def all_face_features(images_dir_path):
    faces_features = []
    people_name = []
    faces_base64 = []
    if os.path.exists(images_dir_path) and os.path.isdir(images_dir_path):
        for file in os.listdir(images_dir_path):
            face_base64 = base64Utils.image2base64(images_dir_path + file)
            faces_base64.append(face_base64)
            features_128d = face_features_128D(images_dir_path + file)
            if features_128d != 0:
                faces_features.append(features_128d)
                people_name.append(file.split('.')[0].split('/')[-1])

    return people_name, faces_features, faces_base64


def faces_write_mysql():
    mysqlConfig = {'host': '47.100.78.117', 'passwd': 'TANG123456', 'user': 'root', 'database': 'videosurveillance',
                   'port': 3306}

    mysqlUtils = mysqlUtil(mysqlConfig)
    people_name, faces_features, faces_base64 = all_face_features(images_dir_path)
    for i in range(len(people_name)):
        feature_str = ''
        for face_feature in faces_features[i]:
            feature_str += str(face_feature) + ','
        print(people_name[i], feature_str[-1], faces_base64[i])
        sql = "insert into face(user_name,face_features,face_image) values('%s','%s','%s');" % (
            people_name[i], feature_str[:-1], faces_base64[i])
        print(sql)
        mysqlUtils.executeSql(sql)


def faces_feature_write_csv(csv_path):
    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        people_name, faces_features, faces_base64 = all_face_features(images_dir_path)
        for i in range(len(people_name)):
            list = []
            list.append(people_name[i])
            feature_str = ''
            for face_feature in faces_features[i]:
                list.append(face_feature)
                feature_str += str(face_feature)
            print(feature_str)
            print('\n', len(feature_str))
            print(list)
            writer.writerow(list)


def single_face_feature_write_mysql(image_path, person_name):
    if os.path.exists(image_path):
        face_base64 = base64Utils.image2base64(image_path)
        features_128d = face_features_128D(image_path)
        feature_str = ''
        for face_feature in features_128d:
            feature_str += str(face_feature) + ','
        sql = "insert into face(user_name,face_features,face_image) values('%s','%s','%s');" % (
            person_name, feature_str[:-1], face_base64)
        print(sql)
        mysqlConfig = {'host': '47.100.78.117', 'passwd': 'TANG123456', 'user': 'root', 'database': 'videosurveillance',
                       'port': 3306}
        mysqlUtils = mysqlUtil(mysqlConfig)
        mysqlUtils.executeSql(sql)


if __name__ == '__main__':
    # csv_path = 'data/face_feature_data.csv'
    # faces_feature_write_csv(csv_path)
    faces_write_mysql()
    # user_name = 'tang'
    # face_features = 'dfjeojfowei'
    # face_image = 'imagessss'
    # sql = "insert into face(user_name,face_features,face_image) values('%s','%s','%s');" % (
    #     user_name, face_features, face_image)
    # print(sql)
