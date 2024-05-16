# -*- coding: utf-8 -*-
"""
Created on Sun May  5 19:18:43 2024

@author: lplpe
"""

import face_recognition
from PIL import Image, ImageDraw

face_image = face_recognition.load_image_file("images/samples/paulo.jpg")

face_landmarks_list = face_recognition.face_landmarks(face_image)

print(face_landmarks_list)

for face_landmark in face_landmarks_list:
    #convert the numpy array image into pil image object
    pil_image = Image.fromarray(face_image)
    #convert the pil image to draw object
    d = ImageDraw.Draw(pil_image)
    
    d.line(face_landmark['chin'], fill=(255,255,255), width=2)
    d.line(face_landmark['left_eyebrow'], fill=(255,255,255), width=2)
    d.line(face_landmark['right_eyebrow'], fill=(255,255,255), width=2)
    d.line(face_landmark['nose_bridge'], fill=(255,255,255), width=2)
    d.line(face_landmark['nose_tip'], fill=(255,255,255), width=2)
    d.line(face_landmark['left_eye'], fill=(255,255,255), width=2)
    d.line(face_landmark['right_eye'], fill=(255,255,255), width=2)
    d.line(face_landmark['top_lip'], fill=(255,255,255), width=2)
    d.line(face_landmark['bottom_lip'], fill=(255,255,255), width=2)

pil_image.show()

pil_image.save("images/samples/paulo_landmarks.jpg")