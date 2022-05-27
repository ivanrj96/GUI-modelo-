# -*- coding: utf-8 -*-
"""
Created on Tue May 24 17:40:17 2022

@author: Ivan
"""

from utils.datasets import get_labels
import cv2
from keras.models import load_model


# parameters for loading data and images
emotion_model_path = './models/emotion_model.hdf5'
emotion_labels = get_labels('fer2013')

# hyper-parameters for bounding boxes shape
window_frame = 10
emotion_offsets = (20, 40)

# loading model
face_cascade = cv2.CascadeClassifier('./models/haarcascade_frontalface_default.xml')
emotion_classifier = load_model(emotion_model_path)

# getting input model shapes for inference
emotion_target_size = emotion_classifier.input_shape[1:3]

# starting lists for calculating modes
emotion_window = []
