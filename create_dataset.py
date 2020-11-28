import os
import numpy as np
import pickle
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import timeit
import sklearn
import cv2
import sys
import glob
sys.path.append('./recognition')
from embedding import Embedding
from menpo.visualize import print_progress
from menpo.visualize.viewmatplotlib import sample_colours_from_colourmap
from prettytable import PrettyTable
from pathlib import Path
import warnings 
warnings.filterwarnings("ignore")  

def get_image_feature(img_path, img_list_path, model_path, gpu_id):
    img_list = open(img_list_path)
    embedding = Embedding(model_path, 0, gpu_id)
    files = img_list.readlines()
    img_feats = []
    faceness_scores = []
    face_names = []
    for img_index, each_line in enumerate(print_progress(files)):
        name_lmk_score = each_line.strip().split(' ')
        face_name = name_lmk_score[0].split('.')[0]
        img_name = os.path.join(img_path, name_lmk_score[0])
        img = cv2.imread(img_name)
        lmk = np.array([float(x) for x in name_lmk_score[1:-1]], dtype=np.float32)
        lmk = lmk.reshape( (5,2) )
        img_feats.append(embedding.get(img,lmk))
        faceness_scores.append(name_lmk_score[-1])
        face_names.append(face_name)
    img_feats = np.array(img_feats).astype(np.float32)
    faceness_scores = np.array(faceness_scores).astype(np.float32)
    return img_feats, faceness_scores, face_names


start = timeit.default_timer()
#img_feats = read_image_feature('./MS1MV2/IJBB_MS1MV2_r100_arcface.pkl')
img_path = './IJBC/loose_crop'
img_list_path = './IJBC/meta/ijbc_name_5pts_score.txt'
model_path = './pretrained_models/MS1MV2-ResNet100-Arcface/model'
gpu_id = 0
img_feats, faceness_scores, face_names = get_image_feature(img_path, img_list_path, model_path, gpu_id)
np.save('faceEmbedding_IJBC', img_feats)
np.save('faceness_scores', faceness_scores)
np.save('name_IJBC', face_names)
stop = timeit.default_timer()
print('Time: %.2f s. ' % (stop - start))
print('Feature Shape: ({} , {}) .'.format(img_feats.shape[0], img_feats.shape[1]))