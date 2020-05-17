import dlib
import cv2

predictor_path = '../data/data_dlib/shape_predictor_68_face_landmarks.dat'
face_rec_model_path = '../data/data_dlib/dlib_face_recognition_resnet_model_v1.dat'
path_features_known_csv = '../data/features_all.csv'

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)
facerec = dlib.face_recognition_model_v1(face_rec_model_path)

# opencv 的人脸识别分类器
classifier = cv2.CascadeClassifier('../data/data_dlib/haarcascade_frontalface_default.xml')

if __name__ == '__main__':
    img = cv2.imread('../static/uploads/tangweiyang.jpg')

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    faces_position = classifier.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=3, minSize=(64, 64))
    print(faces_position)

    image_gray = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    faces = detector(image_gray, 1)

    # print(type(faces))
    # print(faces)
    # print(type(faces[0]))
    # print(faces[0].left(), faces[0].right(), faces[0].top(), faces[0].bottom())
    # shape = predictor(img, faces[0])
    # face_descriptor = facerec.compute_face_descriptor(image_gray, shape)
    # # face_features_cap.append(face_descriptor)
    # print(type(face_descriptor))
    # print(face_descriptor)
    # print('============================')

    rectangle = dlib.dlib.rectangle(132, 63, 213, 143)
    rectangles = dlib.dlib.rectangles()
    rectangles.append(rectangle)
    print(rectangles)

    shape = predictor(img, rectangle)
    face_descriptor = facerec.compute_face_descriptor(image_gray, shape)
    # face_features_cap.append(face_descriptor)
    print(face_descriptor)
