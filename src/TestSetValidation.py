# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 14:59:55 2017

@author: wilkenshuang
"""

import csv
import sys
import os
import cv2
from keras.models import load_model
import numpy as np
from skimage.io import *
from utils.datasets import get_labels
from utils.inference import detect_faces
from utils.inference import draw_text
from utils.inference import draw_bounding_box
from utils.inference import apply_offsets
from utils.inference import load_detection_model
from utils.inference import load_image
from utils.preprocessor import preprocess_input
import gc
import skimage.transform as transform
from keras.utils import np_utils
import matplotlib.pyplot as plt

# parameters for loading data and images
img_fold_path='D:/Kaggle/Facial_Expression_Recognition/test'
detection_model_path = '../trained_models/detection_models/haarcascade_frontalface_default.xml'
emotion_model_path = '../trained_models/emotion_models/fer2013_mini_XCEPTION.102-0.66.hdf5'
emotion_labels = get_labels('fer2013')
data_dir='D:/Kaggle/Facial_Expression_Recognition/fer2013.csv'
train_dir='D:/Kaggle/Facial_Expression_Recognition/train'
test_dir='D:/Kaggle/Facial_Expression_Recognition/test'
data=open(data_dir,'r')

# hyper-parameters for bounding boxes shape
emotion_offsets = (0, 0)

# loading models
face_detection = load_detection_model(detection_model_path)
emotion_classifier = load_model(emotion_model_path, compile=False)

# getting input model shapes for inference
emotion_target_size = emotion_classifier.input_shape[1:3]

# loading labels
labels_Pritest=[]
for row in csv.DictReader(data):
    value=str(row['Usage'])
    if value=='PrivateTest':
        emotion = int(row['emotion'])
        labels_Pritest.append(emotion)
   
# garbage collection        
gc.collect()

# loading images
test_img=np.zeros((len(labels_Pritest),96,96))
j=count=0
show=[]
for i in os.listdir(img_fold_path):
    test=np.squeeze(load_image(os.path.join(img_fold_path,i),grayscale=True)).astype('uint8')
    faces = detect_faces(face_detection, test)
    for face_coordinates in faces:
        x1, x2, y1, y2 = apply_offsets(face_coordinates, emotion_offsets)
        test2 = test[x1:x2, y1:y2]
        
    try:
        test2 = transform.resize(test2,emotion_target_size)
    except:
        continue
    test2 = np.expand_dims(test2, 0)
    test2 = np.expand_dims(test2, -1)
    emotion_label_arg = np.argmax(emotion_classifier.predict(test2))
    if labels_Pritest[j]==emotion_label_arg:
        show.append(j)
        test_img[count]=test
        count+=1
    j+=1

# garbage collection        
gc.collect()
    
for i in range(1,26):
    plt.figure('multiple_image_valid')
    emotion_text = emotion_labels[labels_Pritest[show[i]]]
    plt.subplot(5,5,i)
    plt.imshow(test_img[i],cmap='gray')
    plt.title(emotion_text)
    #plt.xticks([])
    #plt.yticks([])
    plt.axis('off')
plt.savefig('../images/multiple_image_valid.jpg')
    
   



    