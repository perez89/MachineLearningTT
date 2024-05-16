# -*- coding: utf-8 -*-
"""
Created on Sat May  4 00:06:29 2024

@author: lplpe
"""

#import required libraries
import cv2
import dlib

image_to_detect = cv2.imread('images/testing/trump-modi.jpg')

#cv2.imshow("test", image_to_detect)

#load the pretrained MMOD model
face_detection_classifier = dlib.cnn_face_detection_model_v1("models/mmod_human_face_detector.dat")


#detect all face locations using the MMOD classifier
all_face_locations = face_detection_classifier(image_to_detect, 1)

print("There are {0} number of faces in image".format(len(all_face_locations)))

for index, current_face_location in enumerate(all_face_locations):
    
    left_x, left_y, right_x, right_y  = current_face_location.rect.left(), current_face_location.rect.top(), current_face_location.rect.right(), current_face_location.rect.bottom()
        
    print("Found face {0} at top: {1} , right:{2}, bottom:{3}, left:{4}".format(index+1,left_y,right_x,right_y,left_x))
    
    current_face_image = image_to_detect[left_y:right_y, left_x:right_x]    
    cv2.imshow("Face number"+str(index+1), current_face_image)
    cv2.rectangle(image_to_detect, (left_x, left_y), (right_x, right_y), (0 , 255 , 0), 2)
    
cv2.imshow("faces in image", image_to_detect)
    
cv2.waitKey(0)
cv2.destroyAllWindows()