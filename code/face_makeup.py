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
    d = ImageDraw.Draw(pil_image, "RGBA")
    
    #draw the shapes and fill the color
    d.line(face_landmark['chin'], fill=(255,255,255), width=2)
    
    #make left and right eyebrowns darker
    #polygon on top and line on bottom with dark color
    d.polygon(face_landmark['left_eyebrow'], fill=(68,54,39,128))    
    d.polygon(face_landmark['right_eyebrow'], fill=(68,54,39,128))
    d.line(face_landmark['left_eyebrow'], fill=(68,54,39,150), width=5)    
    d.line(face_landmark['right_eyebrow'], fill=(68,54,39,150), width=5)
    

    d.line(face_landmark['nose_bridge'], fill=(255,255,255), width=2)
    d.line(face_landmark['nose_tip'], fill=(255,255,255), width=2)
        
    d.polygon(face_landmark['left_eye'], fill=(255,0,0,100))
    d.polygon(face_landmark['right_eye'], fill=(255,0,0,100))    
    d.line(face_landmark['left_eye'] + [face_landmark['left_eye'][0]], fill=(0,0,0,110), width=6)
    d.line(face_landmark['right_eye'] + [face_landmark['right_eye'][0]], fill=(0,0,0,110), width=6)
    
    #add lipstick to top and bottom lips
    #using red polygons and lines filled with red
    d.polygon(face_landmark['top_lip'], fill=(150 , 0 , 0 , 128))
    d.polygon(face_landmark['bottom_lip'], fill=(150 , 0 , 0 , 128))
    d.line(face_landmark['top_lip'], fill=(150 , 0 , 0 , 64), width=8)
    d.line(face_landmark['bottom_lip'], fill=(150 , 0 , 0 , 64), width=8)
    
pil_image.show()

pil_image.save("images/samples/paulo_makeup.jpg")