# -*- coding: utf-8 -*-
"""
Created on Sat May  4 00:06:29 2024

@author: lplpe
"""

#import required libraries
import cv2
import face_recognition

image_to_detect = cv2.imread('images/rosto3.jpg')

#cv2.imshow("test", image_to_detect)


all_face_locations = face_recognition.face_locations(image_to_detect, model="hog")

print("There are {0} number of faces in image".format(len(all_face_locations)))

for index, current_face_location in enumerate(all_face_locations):
    top_pos, right_pos, bottom_pos, left_pos = current_face_location
    print("Found face {0} at top: {1} , right:{2}, bottom:{3}, left:{4}".format(index+1,top_pos,right_pos,bottom_pos,left_pos))
    current_face_image = image_to_detect[top_pos:bottom_pos, left_pos:right_pos]    
    cv2.imshow("Face number"+str(index+1), current_face_image)

#cv2.waitKey(0)
#cv2.destroyAllWindows()