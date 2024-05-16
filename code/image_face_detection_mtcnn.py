# -*- coding: utf-8 -*-
"""
Created on Sat May  4 00:06:29 2024

@author: lplpe
"""

#import required libraries
import cv2
from  mtcnn.mtcnn import MTCNN
import matplotlib.pyplot as plt

image_to_detect = plt.imread('images/testing/trump-modi.jpg')

#cv2.imshow("test", image_to_detect)

#create instance mtcnn 
mtcnn_detector = MTCNN()

#load pre harr classifier model
all_face_locations = mtcnn_detector.detect_faces(image_to_detect)


print("There are {0} number of faces in image".format(len(all_face_locations)))
print(all_face_locations)

image_to_detect = cv2.cvtColor(image_to_detect, cv2.COLOR_BGR2RGB)

for index, current_face_location in enumerate(all_face_locations):
    x, y, width, height = current_face_location['box']
    
    left_x, left_y = x,y
    right_x, right_y = x+width, y+height
        
    #print the location of current face
    print("Found face {0} at top: {1} , right:{2}, bottom:{3}, left:{4}".format(index+1,left_y,right_x,right_y,left_x))
    
    #slicing the current face from main image
    current_face_image = image_to_detect[left_y:right_y, left_x:right_x]    
    #showing the current face with dynamic title
    cv2.imshow("Face number"+str(index+1), current_face_image)
    #draw bounding box around faces
    cv2.rectangle(image_to_detect, (left_x, left_y), (right_x, right_y), (0 , 255 , 0), 2)
    #dra circles for every face keypoints
    keypoints = current_face_location["keypoints"]
    cv2.circle(image_to_detect, (keypoints["left_eye"]), 5, (0,255,0), 1)
    cv2.circle(image_to_detect, (keypoints["right_eye"]), 5, (0,255,0), 1)
    cv2.circle(image_to_detect, (keypoints["mouth_left"]), 5, (0,255,0), 1)
    cv2.circle(image_to_detect, (keypoints["mouth_right"]), 5, (0,255,0), 1)
    cv2.circle(image_to_detect, (keypoints["nose"]), 5, (0,255,0), 1)
    
cv2.imshow("faces in image", image_to_detect)
    
cv2.waitKey(0)
cv2.destroyAllWindows()
