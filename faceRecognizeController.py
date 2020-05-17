from flask import Flask, render_template, Response, request, jsonify, send_file
import cv2
import os
from pypinyin import pinyin, lazy_pinyin
from werkzeug.utils import secure_filename
from PIL import Image
import dlib_face_features_save
import face_reco_optimize1
import io
from util import base64Utils

# 推流地址
PUSH_FLOW_URL = 'rtmp://127.0.0.1:1935/myapp/20200429102839988?secretKey=20200429102839988-123456'
# 拉流地址
PULL_FLOW_URL = 'rtmp://127.0.0.1:1935/myapp/20200429102839988?secretKey=20200429102839988-456123'


class VideoCamera(object):
    def __init__(self):
        # 通过opencv获取实时视频流
        self.video = cv2.VideoCapture(PULL_FLOW_URL)
        self.faceReco = face_reco_optimize1.face_reco()

    def __del__(self):
        self.video.release()

    def get_frame(self):
        success, image = self.video.read()
        image = self.faceReco.frame_face_reco(image)
        ret, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes()

    def direct_get_frame(self):
        success, image = self.video.read()
        ret, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes()


app = Flask(__name__)


@app.after_request
def cors(environ):
    environ.headers['Access-Control-Allow-Origin'] = '*'
    environ.headers['Access-Control-Allow-Method'] = '*'
    environ.headers['Access-Control-Allow-Headers'] = 'x-requested-with,content-type'
    return environ


@app.errorhandler(500)
def internal_error(error):
    return jsonify({'code': 500, 'msg': '服务器发生错误'})


def gen(camera):
    frameIndex = 0
    while True:
        frameIndex = (frameIndex + 1) % 2
        if frameIndex == 0:
            frame = camera.get_frame()
        else:
            frame = camera.direct_get_frame()
        # 使用generator函数输出视频流， 每次请求输出的content类型是image/jpeg
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


@app.route('/')  # 主页
def index():
    # jinja2模板，具体格式保存在index.html文件中
    return render_template('index.html')


@app.route('/video_feed')  # 这个地址返回视频流响应
def video_feed():
    return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/upload', methods=['POST'])
def upload():
    f = request.files['file']
    name = request.form.get('userName')
    print(name)
    basepath = os.path.dirname(__file__)  # 当前文件所在路径
    print(f.filename)
    suffix = f.filename[:-1].split('.')[1]
    print(lazy_pinyin(f.filename))
    filename = "".join(lazy_pinyin(f.filename))
    print(filename)
    upload_path = os.path.join(basepath, r'static\uploads', secure_filename(filename))  # 注意：没有的文件夹一定要先创建，不然会提示没有该路径
    f.save(upload_path)
    image = Image.open(upload_path)
    # print(image.mode)
    img = image.convert('RGB')
    img.save(upload_path)
    dlib_face_features_save.single_face_feature_write_mysql(upload_path, name)
    return jsonify({'code': 200, 'msg': f.filename + '上传成功', 'userName': name})


@app.route('/face_reco', methods=['POST'])
def face_reco():
    f = request.files['file']
    basepath = os.path.dirname(__file__)  # 当前文件所在路径
    print(f.filename)
    suffix = f.filename[:-1].split('.')[1]
    print(lazy_pinyin(f.filename))
    filename = "".join(lazy_pinyin(f.filename))
    print(filename)
    upload_path = os.path.join(basepath, 'static/uploads', secure_filename(filename))  # 注意：没有的文件夹一定要先创建，不然会提示没有该路径
    # upload_path = 'D:/CodeData/graduate/video_surveillance_dlib/static/uploads/' + secure_filename(filename)
    f.save(upload_path)
    print('保存路径', upload_path)
    # image = Image.open(upload_path)
    image = Image.open(upload_path)
    img = image.convert('RGB')
    img.save(upload_path)
    img = cv2.imread(upload_path)

    faceReco = face_reco_optimize1.face_reco()
    frame = faceReco.frame_face_reco(img)

    image_path = 'static/uploads/demo.jpg'
    cv2.imwrite(image_path, frame)
    return base64Utils.image2base64(image_path)
    # with open("static/uploads/demo.jpg", 'rb') as bites:
    #     return send_file(
    #         io.BytesIO(bites.read()),
    #         attachment_filename='demo.jpg',
    #         mimetype='image/jpg'
    #     )
    #
    # return Response(generateFrame(frame), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/testdownload')
def test_face_reco():
    with open("static/uploads/demo.jpg", 'rb') as bites:
        return send_file(
            io.BytesIO(bites.read()),
            attachment_filename='demo.jpg',
            mimetype='image/jpg'
        )

    return Response(generateFrame(frame), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=5000)
