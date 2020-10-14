# encoding: utf-8
import sys
from operator import itemgetter
import numpy as np
from numpy.linalg import norm
import cv2
import math
import matplotlib.pyplot as plt

#-----------------------------#
#   将长方形调整为正方形
#-----------------------------#
def rect2square(rectangles):
    w = rectangles[:,2] - rectangles[:,0]
    h = rectangles[:,3] - rectangles[:,1]
    l = np.maximum(w,h).T
    rectangles[:,0] = rectangles[:,0] + w*0.5 - l*0.5
    rectangles[:,1] = rectangles[:,1] + h*0.5 - l*0.5 
    rectangles[:,2:4] = rectangles[:,0:2] + np.repeat([l], 2, axis = 0).T 
    return rectangles

#-------------------------------------#
#   人脸对齐
#-------------------------------------#
def Alignment_1(img,landmark):
    # print(landmark.shape)
    if landmark.shape[0]==68:
        x = landmark[36,0] - landmark[45,0]
        y = landmark[36,1] - landmark[45,1]
    elif landmark.shape[0]==5:
        # print(landmark[0,0])
        # print(landmark[0,1])
        # print(landmark[1,0])
        # print(landmark[1,1])
        x = landmark[0,0] - landmark[1,0]
        y = landmark[0,1] - landmark[1,1]

    if x==0:
        angle = 0
    else: 
        angle = math.atan(y/x)*180/math.pi

    center = (img.shape[1]//2, img.shape[0]//2)

    RotationMatrix = cv2.getRotationMatrix2D(center, angle, 1)
    new_img = cv2.warpAffine(img,RotationMatrix,(img.shape[1],img.shape[0])) 

    RotationMatrix = np.array(RotationMatrix)
    new_landmark = []
    for i in range(landmark.shape[0]):
        pts = []    
        pts.append(RotationMatrix[0,0]*landmark[i,0]+RotationMatrix[0,1]*landmark[i,1]+RotationMatrix[0,2])
        pts.append(RotationMatrix[1,0]*landmark[i,0]+RotationMatrix[1,1]*landmark[i,1]+RotationMatrix[1,2])
        new_landmark.append(pts)

    new_landmark = np.array(new_landmark)

    return new_img, new_landmark

def Alignment_2(img,std_landmark,landmark):
    def Transformation(std_landmark,landmark):
        std_landmark = np.matrix(std_landmark).astype(np.float64)
        landmark = np.matrix(landmark).astype(np.float64)

        c1 = np.mean(std_landmark, axis=0)
        c2 = np.mean(landmark, axis=0)
        std_landmark -= c1
        landmark -= c2

        s1 = np.std(std_landmark)
        s2 = np.std(landmark)
        std_landmark /= s1
        landmark /= s2 

        U, S, Vt = np.linalg.svd(std_landmark.T * landmark)
        R = (U * Vt).T

        return np.vstack([np.hstack(((s2 / s1) * R, c2.T - (s2 / s1) * R * c1.T)),np.matrix([0., 0., 1.])])

    Trans_Matrix = Transformation(std_landmark,landmark) # Shape: 3 * 3
    Trans_Matrix = Trans_Matrix[:2]
    Trans_Matrix = cv2.invertAffineTransform(Trans_Matrix)
    new_img = cv2.warpAffine(img,Trans_Matrix,(img.shape[1],img.shape[0]))

    Trans_Matrix = np.array(Trans_Matrix)
    new_landmark = []
    for i in range(landmark.shape[0]):
        pts = []    
        pts.append(Trans_Matrix[0,0]*landmark[i,0]+Trans_Matrix[0,1]*landmark[i,1]+Trans_Matrix[0,2])
        pts.append(Trans_Matrix[1,0]*landmark[i,0]+Trans_Matrix[1,1]*landmark[i,1]+Trans_Matrix[1,2])
        new_landmark.append(pts)

    new_landmark = np.array(new_landmark)

    return new_img, new_landmark

#---------------------------------#
#   l2标准化
#---------------------------------#
def l2_normalize(x, axis=-1, epsilon=1e-10):
    output = x / np.sqrt(np.maximum(np.sum(np.square(x), axis=axis, keepdims=True), epsilon))
    return output
#---------------------------------#
#   计算128特征值
#---------------------------------#
def calc_128_vec(model,img):
    face_img = pre_process(img)
    pre = model.predict(face_img)
    pre = l2_normalize(np.concatenate(pre))
    pre = np.reshape(pre,[128])
    return pre

#---------------------------------#
#   计算人脸距离
#---------------------------------#
def face_distance_1(face_encodings, face_to_compare):
    if len(face_encodings) == 0:
        return np.empty((0))
    return np.linalg.norm(face_encodings - face_to_compare, axis=1)

def compute_sim(face_encodings, face_to_compare):
    if len(face_encodings) == 0:
        return np.empty((0))
    dist_list = []
    for face_encoding in face_encodings:
        dist = np.dot(face_to_compare, face_encoding)/(norm(face_to_compare)*norm(face_encoding))
        dist_list.append(dist)
    return np.asarray(dist_list)

def face_distance(face_encodings, face_to_compare):
    if len(face_encodings) == 0:
        return np.empty((0))
    dist_list = []
    for i in range(len(face_encodings)):
        face_encodings = np.mat(face_encodings)
        dist = np.sqrt(np.square(np.subtract(face_to_compare, face_encodings[i, :])))
        dist_list.append(dist)
    return dist_list

#---------------------------------#
#   比较人脸
#---------------------------------#
def compare_faces(known_face_encodings, face_encoding_to_check, tolerance=0.6):
    dis = face_distance(known_face_encodings, face_encoding_to_check) 
    res = []
    for i in range(len(dis)):
        res.append(dis[i] <= tolerance)
    return res
    # return list(dis <= tolerance)

def compare_faces_1(known_face_encodings, face_encoding_to_check, tolerance=18):
    dis = compute_sim(known_face_encodings, face_encoding_to_check)
    # print(dis)
    return list(dis >= tolerance)

def read_image_gbk(filename, resize_height=None, resize_width=None, normalization=False,colorSpace='RGB'):
    '''
    解决imread不能读取中文路径的问题,读取图片数据,默认返回的是uint8,[0,255]
    :param filename:
    :param resize_height:
    :param resize_width:
    :param normalization:是否归一化到[0.,1.0]
    :param colorSpace 输出格式：RGB or BGR
    :return: 返回的RGB图片数据
    '''
    with open(filename, 'rb') as f:
        data = f.read()
        data = np.asarray(bytearray(data), dtype="uint8")
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
     # 或者：
     # bgr_image=cv2.imdecode(np.fromfile(filename,dtype=np.uint8),cv2.IMREAD_COLOR)
    # if bgr_image is None:
    #     print("Warning:不存在:{}", filename)
    #     return None
    # if len(bgr_image.shape) == 2:  # 若是灰度图则转为三通道
    #     print("Warning:gray image", filename)
    #     bgr_image = cv2.cvtColor(bgr_image, cv2.COLOR_GRAY2BGR)
    # if colorSpace=='RGB':
    #     image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)  # 将BGR转为RGB
    # elif colorSpace=="BGR":
    #     image=bgr_image
    # else:
    #     exit(0)
     # show_image(filename,image)
     # image=Image.open(filename)
    # image = resize_image(image,resize_height,resize_width)
    # image = np.asanyarray(image)
    # if normalization:
    #     image=image_normalization(image)
     # show_image("src resize image",image)
    return image

def resize_image(image,resize_height, resize_width):
    '''
    :param image:
    :param resize_height:
    :param resize_width:
    :return:
    '''
    image_shape=np.shape(image)
    height=image_shape[0]
    width=image_shape[1]
    if (resize_height is None) and (resize_width is None):#错误写法：resize_height and resize_width is None
        return image
    if resize_height is None:
        resize_height=int(height*resize_width/width)
    elif resize_width is None:
        resize_width=int(width*resize_height/height)
    image = cv2.resize(image, dsize=(resize_width, resize_height))
    return image

def cv_show_image(title, image, type='rgb'):
    '''
    调用OpenCV显示RGB图片
    :param title: 图像标题
    :param image: 输入RGB图像
    :param type:'rgb' or 'bgr'
    :return:
    '''
    channels=image.shape[-1]
    if channels==3 and type=='rgb':
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # 将BGR转为RGB
    cv2.imshow(title, image)
    cv2.waitKey(0)

def load_name_list(filename):
    with open(filename, mode="r",encoding='utf-8') as f:
        content_list = f.readlines()
        content_list = [content.rstrip() for content in content_list]
        return content_list