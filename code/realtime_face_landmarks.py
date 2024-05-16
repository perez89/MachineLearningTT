# -*- coding: utf-8 -*-
"""
Created on Sun May  5 19:18:43 2024

@author: lplpe
"""

import face_recognition
from PIL import Image, ImageDraw
import cv2
import numpy as np

#capture video from default camera
webcam_video_stream = cv2.VideoCapture(0)
# capture from a video
#webcam_video_stream = cv2.VideoCapture("videos/paulo_video_test.mp4")

all_face_locations  = []

while True:
    ret, current_frame = webcam_video_stream.read()
    
    face_landmarks_list = face_recognition.face_landmarks(current_frame)
    
    #convert the numpy array image into pil image object
    pil_image = Image.fromarray(current_frame)
    #convert the pil image to draw object
    d = ImageDraw.Draw(pil_image)
     
    index = 0              
    #loop through every face
    while index < len(face_landmarks_list): 
        print(face_landmarks_list)
        #loop through face landmarks
        for face_landmark in face_landmarks_list:
            d.line(face_landmark['chin'], fill=(255,255,255), width=2)
            d.line(face_landmark['left_eyebrow'], fill=(255,255,255), width=2)
            d.line(face_landmark['right_eyebrow'], fill=(255,255,255), width=2)
            d.line(face_landmark['nose_bridge'], fill=(255,255,255), width=2)
            d.line(face_landmark['nose_tip'], fill=(255,255,255), width=2)
            d.line(face_landmark['left_eye'], fill=(255,255,255), width=2)
            d.line(face_landmark['right_eye'], fill=(255,255,255), width=2)
            d.line(face_landmark['top_lip'], fill=(255,255,255), width=2)
            d.line(face_landmark['bottom_lip'], fill=(255,255,255), width=2)
        index +=1
    
    #convert PIL image to RGB to show in opencv window
    rgb_image = pil_image.convert("RGB")
    rgb_open_cv_image = np.array(pil_image)
    
    #convert RGB to BGR
    bgr__open_cv_image = cv2.cvtColor(rgb_open_cv_image, cv2.COLOR_RGB2BGR)
    bgr__open_cv_image = bgr__open_cv_image[:, : , ::-1].copy()
    
    cv2.imshow("Webcam video", bgr__open_cv_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

webcam_video_stream.release()
cv2.destroyAllWindows()