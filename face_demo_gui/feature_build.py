import sys
# sys.path.insert(0, '/util/python')
import cPickle
import cv2
import numpy as np
import os
import csv
import matplotlib.pyplot as plt
import time
# from keras_mtcnn import detect as MTCNNDetector
#from keras_mtcnn import Run_model_caffe_weight as MTCNNDetector
from keras_mtcnn import tools_matrix as tools
from keras_mtcnn.MTCNN import create_Kao_Onet, create_Kao_Rnet, create_Kao_Pnet
import threading
from align_faces_new import CropFace
from models import Xception
from CenterLoss import get_center_loss
from keras.models import load_model
from keras.models import Model
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Input
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras import backend as K
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping,Callback
from keras.optimizers import Adam, SGD, RMSprop
from keras.layers.core import Lambda
from keras.layers import merge
from keras.utils import Sequence


threshold = [0.6, 0.6, 0.7]
factor = 0.709
OFFSET_PERCENT = (0, 0)
NUM_CLASSES = 75109


model_name = 'xception'
# model_root_dir = '/mnt/hal/UBFacePipeline/Recognition/vgg_face_keras'
# if not os.path.exists(sys.argv[1]):
#     raise Exception('Model file path - %s does not exist!!' % sys.argv[1])
model_file_path = '/home/dmohan/Workspace/faceDemo/Face/xception_MSCeleb.16.h5'
model_file_name = model_file_path[model_file_path.rfind('/')+1:]
print 'Model file name: ', model_file_name

_, preprocess_fn = Xception(num_classes=NUM_CLASSES, only_preprocess_fn_needed=True)
# model, preprocess_fn = DenseNet(classes=NUM_CLASSES, reduction=0.5, weights_path='./DenseNet/weights/densenet121_weights_tf.h5', only_preprocess_fn_needed=only_preprocess_fn_needed)

center_loss = get_center_loss(0.5, NUM_CLASSES, 2048, beta=0.003)

# Load latest model
print 'Loading model from file %s...' % model_file_path
model = load_model(model_file_path, custom_objects={'_center_loss_func': center_loss})
    
# If model was multi-gpu model, extract original nested model
for l in model.layers:
    if type(l) == Model:
        model = l
        model.name = model_name
        print 'Loaded model weights for model %s...' % model_name

if model.name != model_name:
    raise Exception('ERROR: Provided model name (%s) doesn\'t match loaded model (%s)' % (model_name, model.name))

# # Get model with both label predictions and features
# feat_dim = model.layers[-2].output.shape[-1]
# print 'Feature dimension: ', feat_dim
model = Model(inputs=model.inputs, outputs=[model.layers[-1].output, model.layers[-2].output], name=model.name)
#model = Model(inputs=model.inputs, outputs=model.layers[-5].output, name=model.name)
if os.path.exists('gui_dict.pickle'):
    with open('gui_dict.pickle','rb') as handle:
        dict_personimages = cPickle.load(handle)


def feature_vector(img,rectangles,model,preprocess_fn,person):
    alignedFace = []
    for i in range(len(rectangles)):
        x = int(rectangles[i][0])
        y = int(rectangles[i][1])
        w = int(rectangles[i][2]) - x
        h = int(rectangles[i][3]) - y
        landmarkPoints = []
        for j in range(5, 15, 2):
            landmarkPoints.append(int(rectangles[i][j]))
            landmarkPoints.append(int(rectangles[i][j + 1]))
        
        
        # alignedImg = CropFace(img, OFFSET_PERCENT,
        #                  eye_left=(int(np.round(float(best_det['LEFT_EYE_X']))), int(np.round(float(best_det['LEFT_EYE_Y'])))),
        #                  eye_right=(int(np.round(float(best_det['RIGHT_EYE_X']))), int(np.round(float(best_det['RIGHT_EYE_Y'])))),
        #                  nose_tip =(int(np.round(float(best_det['NOSE_X']))),int(np.round(float(best_det['NOSE_Y'])))),
        #                  mouth_left =(int(np.round(float(best_det['LEFT_MOUTH_CORNER_X']))), int(np.round(float(best_det['LEFT_MOUTH_CORNER_Y'])))),
        #                  mouth_right =(int(np.round(float(best_det['RIGHT_MOUTH_CORNER_X']))),
        #                  int(np.round(float(best_det['RIGHT_MOUTH_CORNER_Y'])))))
        alignedImg = CropFace(img, OFFSET_PERCENT,
                         eye_left=(int(np.round(landmarkPoints[0])), int(np.round(float(landmarkPoints[1])))),
                         eye_right=(int(np.round(float(landmarkPoints[2]))), int(np.round(float(landmarkPoints[3])))),
                         nose_tip =(int(np.round(float(landmarkPoints[4]))),int(np.round(float(landmarkPoints[5])))),
                         mouth_left =(int(np.round(float(landmarkPoints[6]))), int(np.round(float(landmarkPoints[7])))),
                         mouth_right =(int(np.round(float(landmarkPoints[8]))),
                         int(np.round(float(landmarkPoints[9])))))

    # crop = np.array(alignedImg).astype(np.float32) / 255.0

        crop = alignedImg
        crop = cv2.resize(crop,(299, 299)).astype(np.float32)
        crop = crop / 255.0
        print 'crop shape', crop.shape
        cv2.imshow("window",crop)
        print person 
        c = cv2.waitKey(1) & 0xFF
        if c == ord('q'):
            continue

        alignedFace.append(crop)

    alignedFace = np.asarray(alignedFace)
    print 'aligned Face',alignedFace.shape
    
    alignedFace = preprocess_fn(alignedFace)
    print type(alignedFace)

    cos_sim = model.predict(alignedFace)
    return cos_sim[1][0]

def generate_vectors():

    feature_dict = dict()

    for name in dict_personimages:
    #     for fr_pair in range(len(dict_personimages(person))):
        feature_dict[name] = {}
        for fr_pair in range(len(dict_personimages[name])):
            vectorlist = []
            frame = dict_personimages[name][fr_pair][0]
            rectangle = dict_personimages[name][fr_pair][1]
            feature = feature_vector(frame, rectangle, model, preprocess_fn, name)
            vectorlist.append(feature)
        feature_dict[name] = vectorlist

    print feature_dict
    with open('gui_feature.pickle','wb') as handle:
        cPickle.dump(feature_dict, handle)






