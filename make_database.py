import cv2
import os
import numpy as np
import utils.utils as utils
import insightface
import datetime

def getFilePathList(file_dir):
    '''
    获取file_dir目录下，所有文本路径，包括子目录文件
    :param rootDir:
    :return:
    '''
    filePath_list = []
    for walk in os.walk(file_dir):
        part_filePath_list = [os.path.join(walk[0], file) for file in walk[2]]
        filePath_list.extend(part_filePath_list)
    return filePath_list

def get_files_list(file_dir, postfix=None):
    '''
    获得file_dir目录下，后缀名为postfix所有文件列表，包括子目录
    :param file_dir:
    :param postfix: ['*.jpg','*.png'],postfix=None表示全部文件
    :return:
    '''
    file_list = []
    filePath_list = getFilePathList(file_dir)
    if postfix is None:
        file_list = filePath_list
    else:
        postfix = [p.split('.')[-1] for p in postfix]
        for file in filePath_list:
            basename = os.path.basename(file)  # 获得路径下的文件名
            postfix_name = basename.split('.')[-1]
            if postfix_name in postfix:
                file_list.append(file)
    file_list.sort()
    return file_list

def make_database():
    # prepare models
    retinaface_model = insightface.model_zoo.get_model('retinaface_mnet025_v1')
    retinaface_model.prepare(ctx_id=-1, nms=0.4)
    arcface_model = insightface.model_zoo.get_model('arcface_r100_v1')
    arcface_model.prepare(ctx_id=-1)
    
    face_list = getFilePathList('./face_dataset/casia')
    face_list.sort()
    known_face_encodings = []
    known_face_names = []
    timea = datetime.datetime.now()
    for face in face_list:
        name = face.split(os.sep)[-2]
        # print(name)
        # print(type(name))
        image_path = os.path.join('./face_dataset/casia', name, name + '_0.bmp') # 取第一张图片作数据库
        if(face != image_path): 
            continue
        print(image_path)
        img = cv2.imread(image_path)
        # print(type(img)) 
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        rectangle, landmark = retinaface_model.detect(img, threshold=0.5, scale=1.0)
        rectangle = utils.rect2square(np.array(rectangle))
        crop_img = img[int(rectangle[0, 1]):int(rectangle[0, 3]), int(rectangle[0, 0]):int(rectangle[0, 2])]
        crop_img = cv2.resize(crop_img, (112, 112))
        new_img, _ = utils.Alignment_1(crop_img, landmark[0])
        face_encoding = arcface_model.get_embedding(new_img)
        known_face_encodings.append(face_encoding)
        known_face_names.append(name)
    face_num = len(known_face_names)
    known_face_encodings = np.array(known_face_encodings).reshape(face_num, 512)
    np.save('faceEmbedding_casia', known_face_encodings)
    np.save('name_casia', known_face_names)
    timeb = datetime.datetime.now()
    diff = timeb - timea
    print("Building database:", diff.total_seconds(), 'seconds')    
        
if __name__ == "__main__":
    make_database()
    