# encoding: utf-8
import cv2
import os
import numpy as np
import utils.utils as utils
import insightface
import datetime
import hnsw

class face_rec():
    def __init__(self):
        self.retinaface_model = insightface.model_zoo.get_model('retinaface_mnet025_v1')
        self.retinaface_model.prepare(ctx_id=0, nms=0.4)
        self.arcface_model = insightface.model_zoo.get_model('arcface_r100_v1')
        self.arcface_model.prepare(ctx_id=0)
        self.p = load_index('IJBC_index.bin', dim=512)

    def recognize(self, draw):
        height, width, _ = np.shape(draw)
        draw_rgb = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)
        time_0 = datetime.datetime.now()
        rectangles, landmarks = self.retinaface_model.detect(draw_rgb, threshold=0.5, scale=1.0)
        if len(rectangles) == 0:
            return
        # 转化成正方形
        rectangles = utils.rect2square(np.array(rectangles, dtype=np.int32))

        time_now = datetime.datetime.now()
        detect_time = time_now - time_0
        print('Detection Time:', detect_time.total_seconds(), 'seconds')
        # -----------------------------------------------#
        #   对检测到的人脸进行编码
        # -----------------------------------------------#
        time_0 = datetime.datetime.now()
        face_encodings = []
        for rectangle, landmark in zip(rectangles, landmarks):
            # landmark = (np.reshape(rectangle[5:15],(5,2)) - np.array([int(rectangle[0]),int(rectangle[1])]))/(rectangle[3]-rectangle[1])*112

            crop_img = draw_rgb[int(rectangle[1]):int(rectangle[3]), int(rectangle[0]):int(rectangle[2])]
            crop_img = cv2.resize(crop_img, (112, 112))

            new_img, _ = utils.Alignment_1(crop_img, landmark)
            # new_img = np.expand_dims(new_img,0)

            face_encoding = self.arcface_model.get_embedding(new_img)
            face_encodings.append(face_encoding)

        face_names = []
        for face_encoding in face_encodings:
            # Query the elements for themselves
            name, distances = self.p.knn_query(face_encoding, k=1) 
            print(distances)
            face_names.append(name)
        time_now = datetime.datetime.now()
        rec_time = time_now - time_0
        print('Recognition Time:', rec_time.total_seconds(), 'seconds')
        
        #-----------------------------------------------#
        #   画框~!~
        #-----------------------------------------------#
        rectangles = rectangles[:, 0:4]
        for (left, top, right, bottom), name in zip(rectangles, face_names):
            cv2.rectangle(draw, (left, top), (right, bottom), (0, 0, 255), 1)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(draw, name, (left, bottom - 15), font, 0.5, (0, 0, 255), 1)
        return draw


if __name__ == "__main__":
    dududu = face_rec()
    image_path = 'test_data/000_2.BMP'
    draw = utils.read_image_gbk(image_path)
    dududu.recognize(draw)
    cv2.imshow('Video', draw)
    cv2.waitKey(0)
    cv2.destroyAllWindows()