# encoding: utf-8
import cv2
import os
import numpy as np
import utils.utils as utils
import retinaface
import arcface
import datetime
import hnsw
import argparse
import mxnet as mx
from mxnet import ndarray as nd

class face_rec():
    def __init__(self, dt_model, rg_model, index, dim, ctx_id):
        #self.retinaface_model = insightface.model_zoo.get_model('retinaface_mnet025_v1')
        #self.retinaface_model.prepare(ctx_id=0, nms=0.4)
        self.retinaface_model = retinaface.FaceDetector(dt_model, rac='net3')
        self.retinaface_model.prepare(ctx_id=ctx_id, nms=0.4)
        self.arcface_model = arcface.FaceRecognition(rg_model)
        self.arcface_model.prepare(ctx_id=ctx_id)
        #self.arcface_model = insightface.model_zoo.get_model('arcface_r100_v1')
        #self.arcface_model.prepare(ctx_id=0)
        self.p = hnsw.load_index(index, dim=dim)
        self.p.set_ef(25)

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
            #print(face_encoding.shape)
            name = 'Unknown'
            # Query the elements for themselves
            names, distances = self.p.knn_query(face_encoding, k=1) # 返回的距离是 1 - cosine
            # print(distances)
            matches = list(1 - distances >= 0.45)
            best_match_index = np.argmax(distances)
            if matches[best_match_index]:
                name = str(names[best_match_index][0]) # names 多了个维度
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
    parser = argparse.ArgumentParser(description='do verification')
    # general
    parser.add_argument('--data-dir', default='', help='')
    parser.add_argument('--model1',
                        default='./model/retinaface_mnet025_v1/mnet10-0000.params',
                        help='path to load model.')
    parser.add_argument('--model2',
                        default='./model/arcface_r100_v1/model-0000.params',
                        help='path to load model.')
    parser.add_argument('--target',
                        default='./test_data/obama.jpg',
                        help='test targets.')
    parser.add_argument('--index', default='./database/IJBC_index_512.bin', help='')
    parser.add_argument('--gpu', default=0, type=int, help='gpu id')
    parser.add_argument('--batch-size', default=32, type=int, help='')
    parser.add_argument('--max', default='', type=str, help='')
    parser.add_argument('--dim', default=512, type=int, help='')
    args = parser.parse_args()
    #prefix1 = args.model1.split(',')[0]
    #prefix2 = args.model2.split(',')[0]

    dududu = face_rec(args.model1, args.model2, args.index, args.dim, args.gpu)
    image_path = args.target
    draw = utils.read_image_gbk(image_path)
    dududu.recognize(draw)
    cv2.imshow('Video', draw)
    cv2.waitKey(0)
    cv2.destroyAllWindows()