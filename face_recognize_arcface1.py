# encoding: utf-8
import cv2
import os
import numpy as np
# from net.mtcnn import mtcnn
import utils.utils as utils
# from net.inception import InceptionResNetV1
import insightface
import datetime

class face_rec():
    def __init__(self):
        # 创建mtcnn对象
        # 检测图片中的人脸
        self.retinaface_model = insightface.model_zoo.get_model('retinaface_mnet025_v1')
        self.retinaface_model.prepare(ctx_id = -1, nms=0.4)
        # 门限函数
        # self.threshold = [0.5,0.8,0.9]

        # 载入arcface
        # 将检测到的人脸转化为512维的向量
        self.arcface_model = insightface.model_zoo.get_model('arcface_r100_v1')
        self.arcface_model.prepare(ctx_id = -1)
        
        #-----------------------------------------------#
        #   对数据库中的人脸进行编码
        #   known_face_encodings中存储的是编码后的人脸
        #   known_face_names为人脸的名字
        #-----------------------------------------------#
        face_list = os.listdir("face_dataset")
        # print(face_list)

        self.known_face_encodings=[]

        self.known_face_names=[]

        timea = datetime.datetime.now()
        for face in face_list:
            name = face.split(".")[0]
            
            img = cv2.imread("./face_dataset/"+face)
            # print(type(img))
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

            # 检测人脸
            # rectangles = self.mtcnn_model.detectFace(img, self.threshold)
            rectangle, landmark = self.retinaface_model.detect(img, threshold=0.5, scale=1.0)

            # 转化成正方形
            rectangle = utils.rect2square(np.array(rectangle))
            # arcface要传入一个112x112的图片
            # rectangle = rectangles[0]
            # 记下他们的landmark
            # landmark = (np.reshape(rectangle[5:15],(5,2)) - np.array([int(rectangle[0]),int(rectangle[1])]))/(rectangle[3]-rectangle[1])*112

            # print(rectangle.shape)
            # print(landmark.shape)
            crop_img = img[int(rectangle[0, 1]):int(rectangle[0, 3]), int(rectangle[0, 0]):int(rectangle[0, 2])]
            crop_img = cv2.resize(crop_img,(112,112))
            # print(crop_img.shape[0:2])

            new_img,_ = utils.Alignment_1(crop_img,landmark[0])
            # print(new_img.shape[0:2])
            # new_img = np.expand_dims(new_img,0)
            # print(new_img.shape[0:2])
            # 将检测到的人脸传入到arcface的模型中，实现512维特征向量的提取
            # face_encoding = utils.calc_128_vec(self.facenet_model,new_img)
            face_encoding = self.arcface_model.get_embedding(new_img)
            self.known_face_encodings.append(face_encoding)
            self.known_face_names.append(name)
        self.known_face_encodings = np.array(self.known_face_encodings).reshape(6,512)
        timeb = datetime.datetime.now()
        diff = timeb - timea
        print("Building database:", diff.total_seconds(), 'seconds')
        # print(type(self.known_face_encodings))
        # print(self.known_face_names)

    def recognize(self,draw):
        #-----------------------------------------------#
        #   人脸识别
        #   先定位，再进行数据库匹配
        #-----------------------------------------------#
        height,width,_ = np.shape(draw)
        draw_rgb = cv2.cvtColor(draw,cv2.COLOR_BGR2RGB)

        # 检测人脸
        time_0 = datetime.datetime.now()
        rectangles, landmarks = self.retinaface_model.detect(draw_rgb, threshold=0.5, scale=1.0)

        if len(rectangles)==0:
            return

        # 转化成正方形
        rectangles = utils.rect2square(np.array(rectangles,dtype=np.int32))
        # rectangles[:,0] = np.clip(rectangles[:,0],0,width)
        # rectangles[:,1] = np.clip(rectangles[:,1],0,height)
        # rectangles[:,2] = np.clip(rectangles[:,2],0,width)
        # rectangles[:,3] = np.clip(rectangles[:,3],0,height)
        time_now = datetime.datetime.now()
        detect_time = time_now - time_0
        print('Detection Time:', detect_time.total_seconds(), 'seconds')
        #-----------------------------------------------#
        #   对检测到的人脸进行编码
        #-----------------------------------------------#
        time_0 = datetime.datetime.now()
        face_encodings = []
        for rectangle, landmark in zip(rectangles, landmarks):
            # landmark = (np.reshape(rectangle[5:15],(5,2)) - np.array([int(rectangle[0]),int(rectangle[1])]))/(rectangle[3]-rectangle[1])*112

            crop_img = draw_rgb[int(rectangle[1]):int(rectangle[3]), int(rectangle[0]):int(rectangle[2])]
            crop_img = cv2.resize(crop_img,(112,112))

            new_img,_ = utils.Alignment_1(crop_img,landmark)
            # new_img = np.expand_dims(new_img,0)

            # face_encoding = utils.calc_128_vec(self.facenet_model,new_img)
            face_encoding = self.arcface_model.get_embedding(new_img)
            face_encodings.append(face_encoding)
        # print(np.linalg.norm(face_encodings, axis=1))
        

        face_names = []
        for face_encoding in face_encodings:
            # 取出一张脸并与数据库中所有的人脸进行对比，计算得分
            # print(type(face_encoding))
            matches = utils.compare_faces_1(self.known_face_encodings, face_encoding, tolerance = 20)
            # print(matches)
            name = "Unknown"
            # 找出距离最近的人脸
            face_distances = utils.face_distance_1(self.known_face_encodings, face_encoding)
            # print(face_distances)
            # 取出这个最近人脸的评分
            best_match_index = np.argmin(face_distances)
            # print(best_match_index)
            if matches[best_match_index]:
                name = self.known_face_names[best_match_index]
            face_names.append(name)
        time_now = datetime.datetime.now()
        rec_time = time_now - time_0
        print('Recognition Time:', rec_time.total_seconds(), 'seconds')

        rectangles = rectangles[:,0:4]
        #-----------------------------------------------#
        #   画框~!~
        #-----------------------------------------------#
        for (left, top, right, bottom), name in zip(rectangles, face_names):
            cv2.rectangle(draw, (left, top), (right, bottom), (0, 0, 255), 2)
            # print(1)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(draw, name, (left , bottom - 15), font, 0.5, (255, 255, 255), 2) 
        return draw

if __name__ == "__main__":

    dududu = face_rec()
    # video_capture = cv2.VideoCapture(0)

    # while True:
    #     ret, draw = video_capture.read()
    #     dududu.recognize(draw) 
    #     cv2.imshow('Video', draw)
    #     if cv2.waitKey(20) & 0xFF == ord('q'):
    #         break

    # video_capture.release()
    # cv2.destroyAllWindows()

    image_path = 'test_data/2.jpeg'
    draw = utils.read_image_gbk(image_path)
    dududu.recognize(draw)
    cv2.imshow('Video', draw)
    cv2.waitKey(0)
    cv2.destroyAllWindows() 
    # utils.cv_show_image('face recognition', draw)
    # if cv2.waitKey(20) & 0xFF == ord('q'):
    #    cv2.destroyAllWindows() 
