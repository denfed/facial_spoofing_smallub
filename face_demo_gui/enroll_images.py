import cv2
import numpy as np
import os
import csv
import matplotlib.pyplot as plt
from keras_mtcnn import Run_model_caffe_weight as MTCNNDetector
from keras_mtcnn import tools_matrix as tools
from keras_mtcnn.MTCNN import create_Kao_Onet, create_Kao_Rnet, create_Kao_Pnet
import cPickle
from os import listdir
from os.path import isfile, join
import glob
import dlib
from imutils import face_utils




def opencvDetectFace(img,detector,predictor):
    rectangles = []
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    rects = detector(gray, 1)
    #rects = detector.detectMultiScale(gray, 1.3, 5)
    for rect in rects:
        landmarks = []
    # determine the facial landmarks for the face region, then
    # convert the facial landmark (x, y)-coordinates to a NumPy
    # array
        #rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        landmarks.append(rect.left())
        landmarks.append(rect.top())
        landmarks.append(rect.right())
        landmarks.append(rect.bottom())
        landmarks.append(1)
        landmarks.append((int(np.round(np.mean(shape[36:41,:],axis=0))[0])))
        landmarks.append((int(np.round(np.mean(shape[36:41,:],axis=0))[1])))
        landmarks.append((int(np.round(np.mean(shape[42:47,:],axis=0))[0])))
        landmarks.append(int(np.round(np.mean(shape[42:47,:],axis=0)[1])))
        landmarks.append(int(np.round(shape[30])[0]))
        landmarks.append(int(np.round(shape[30])[1]))
        landmarks.append(int(np.round(shape[48])[0]))
        landmarks.append(int(np.round(shape[48])[1]))
        landmarks.append(int(np.round(shape[54])[0]))
        landmarks.append(int(np.round(shape[54])[1]))

        rectangles.append(landmarks)
    return rectangles

def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # Compute overlap
    x_overlap = max(0, (xB - xA + 1))
    y_overlap = max(0, (yB - yA + 1))

    # compute the area of intersection rectangle
    interArea = x_overlap * y_overlap

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    if float(boxAArea + boxBArea - interArea) <= 0:
        iou = 0
    else:
        iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou

def images_to_array(directory, singleimg):


    threshold = [0.6, 0.6, 0.7]
    Pnet = create_Kao_Pnet(weight_path='12net.h5')
    Rnet = create_Kao_Rnet(weight_path='24net.h5')
    Onet = create_Kao_Onet(weight_path='48net.h5')

    face_points = 'shape_predictor_68_face_landmarks.dat'
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(face_points)

    # with open(path+'/'+'Name_Map.txt') as f:
    #     names = []
    #     for line in f:
    #         line = line.rstrip()
    #         names.append(line.split(","))
    if singleimg is False:
        images = []

        # CHANGE TO WALKING PICTURE DIRECTORY

        for image in glob.glob(directory + "/*" ):
            # image_name = glob.glob(path+"/"+name[0]+'*')
            frame = cv2.imread(image)
            cv2.imshow('window',frame)
            c = cv2.waitKey(0)
            # if len(name)<4:
            #     name.append("")
            rectangles = MTCNNDetector.detectFace(frame, Pnet, Rnet, Onet, threshold)
            opencv_rectangles = opencvDetectFace(frame,detector,predictor)
            detected_rectangle = []
            for openrec in opencv_rectangles:
                boxA = openrec[0:4]
                flag = 0
                for rec in rectangles:
                    boxB = rec[0:4]
                    if bb_intersection_over_union(boxA,boxB) > 0.5:
                        flag = 1
                        break
                if flag == 0:
                    detected_rectangle.append(openrec)
            for i in detected_rectangle:
                rectangles.append(i)


            draw = frame.copy()
            if len(rectangles)>0:
                frame_and_rectangle = []
                frame_and_rectangle.append(frame)
                frame_and_rectangle.append(rectangles)
            images.append(frame_and_rectangle)

        return images
    else:
        # image_name = glob.glob(path+"/"+name[0]+'*')
        frame = cv2.imread(directory)
        cv2.imshow('window', frame)
        c = cv2.waitKey(0)
        # if len(name)<4:
        #     name.append("")
        rectangles = MTCNNDetector.detectFace(frame, Pnet, Rnet, Onet, threshold)
        opencv_rectangles = opencvDetectFace(frame, detector, predictor)
        detected_rectangle = []
        for openrec in opencv_rectangles:
            boxA = openrec[0:4]
            flag = 0
            for rec in rectangles:
                boxB = rec[0:4]
                if bb_intersection_over_union(boxA, boxB) > 0.5:
                    flag = 1
                    break
            if flag == 0:
                detected_rectangle.append(openrec)
        for i in detected_rectangle:
            rectangles.append(i)

        draw = frame.copy()
        frame_and_rectangle = []
        if len(rectangles) > 0:
            frame_and_rectangle = []
            frame_and_rectangle.append(frame)
            frame_and_rectangle.append(rectangles)

        return frame_and_rectangle



#         # for rectangle in rectangles:
#         #     if rectangle is not None:
#         #         W = -int(rectangle[0]) + int(rectangle[2])
#         #         H = -int(rectangle[1]) + int(rectangle[3])
#         #         paddingH = 0.01 * W
#         #         paddingW = 0.02 * H
#         #         crop_img = frame[int(rectangle[1]+paddingH):int(rectangle[3]-paddingH), int(rectangle[0]-paddingW):int(rectangle[2]+paddingW)]
#         #         crop_img = cv2.cvtColor(crop_img, cv2.COLOR_RGB2GRAY)
#         #         if crop_img is None:
#         #             continue
#         #         if crop_img.shape[0] < 0 or crop_img.shape[1] < 0:
#         #             continue
#         #         cv2.rectangle(draw, (int(rectangle[0]), int(rectangle[1])), (int(rectangle[2]), int(rectangle[3])), (255, 0, 0), 1)

#         #         for i in range(5, 15, 2):
#         #             cv2.circle(draw, (int(rectangle[i + 0]), int(rectangle[i + 1])), 2, (0, 255, 0))
#         #         cv2.imshow('window',draw)
#         #         c = cv2.waitKey(0)
#         #         if c & 0xFF == ord('q'):    
#         #             break


#     #cv2.imread(path+img_path)
# #cap = cv2.VideoCapture(0)




















# name = raw_input("Please enter the name of the enrollee : ")
# if name in dict_personimages:
#     print 'Username already enrolled!!! do you want overwrite'
#     ans = raw_input()
#     if ans == 'no':
#         name = None
# dict_personimages[name] = {}
# onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]

# for img_path in onlyfiles:
#     print img_path
#     frame = cv2.imread(path+'/'+img_path)
#     rectangles = MTCNNDetector.detectFace(frame, Pnet, Rnet, Onet, threshold)
#     draw = frame.copy()
#     if len(rectangles)>0:
#         for rectangle in rectangles:
#             if rectangle is not None:
#                 W = -int(rectangle[0]) + int(rectangle[2])
#                 H = -int(rectangle[1]) + int(rectangle[3])
#                 paddingH = 0.01 * W
#                 paddingW = 0.02 * H
#                 crop_img = frame[int(rectangle[1]+paddingH):int(rectangle[3]-paddingH), int(rectangle[0]-paddingW):int(rectangle[2]+paddingW)]
#                 crop_img = cv2.cvtColor(crop_img, cv2.COLOR_RGB2GRAY)
#                 if crop_img is None:
#                     continue
#                 if crop_img.shape[0] < 0 or crop_img.shape[1] < 0:
#                     continue
#                 cv2.rectangle(draw, (int(rectangle[0]), int(rectangle[1])), (int(rectangle[2]), int(rectangle[3])), (255, 0, 0), 1)

#                 for i in range(5, 15, 2):
#                     cv2.circle(draw, (int(rectangle[i + 0]), int(rectangle[i + 1])), 2, (0, 255, 0))
#     else:
#         print 'photo is shit'
#         continue
#     cv2.imshow('window',draw)
#     c = cv2.waitKey(0)
#     if c & 0xFF == ord('q'):
#         break
#     elif c & 0xFF == ord('s'):
#         print 'frontal or profile'
#         ans = raw_input()
#         if ans == 'frontal':
#             dict_personimages[name]['frontal_frame'] = frame
#             dict_personimages[name]['frontal_rectangles'] = rectangles
#         elif ans == 'profile':
#             dict_personimages[name]['profile_frame'] = frame
#             dict_personimages[name]['profile_rectangles'] = rectangles    
#     cv2.destroyWindow("window")
# print "Enrollment complete for :",name   
# print dict_personimages
# with open('enroll_face_withprof.pickle','wb') as handle:
#     cPickle.dump(dict_personimages, handle)




#     