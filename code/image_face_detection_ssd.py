# -*- coding: utf-8 -*-
"""
Created on Sat May  4 00:06:29 2024

@author: lplpe
"""

#import required libraries
import cv2
import numpy as np

image_to_detect = cv2.imread('images/testing/trump-modi.jpg')
img_height = image_to_detect.shape[0]
img_width = image_to_detect.shape[1]
#cv2.imshow("test", image_to_detect)


#load pretrainned ssd classifier model
face_detection_classifier = cv2.dnn.readNetFromCaffe("models/deploy.prototxt.txt","models/res10_300x300_ssd_iter_140000.caffemodel")

#resize the image 300x300
resize_image = cv2.resize(image_to_detect, (300,300))
#create blob of the image
image_to_detect_blob = cv2.dnn.blobFromImage(resize_image, 1.0 , (300,300), (104, 177, 123))

#pass the blob as model input
face_detection_classifier.setInput(image_to_detect_blob)

#detect all face locations using the ssd classifiar
all_face_locations = face_detection_classifier.forward()
# 4-D array returned, 
# eg: all_face_locations[0, 0, index, 1] , 1 => will have the prediction class index
# 2 => will have confidence, 
# 3 to 7 => will have the bounding box co-ordinates

print(all_face_locations)
print("There are {0} number of faces in image".format(len(all_face_locations)))
no_of_detections = all_face_locations.shape[2]

#print("x = {0}".format(no_of_detections))

for index in range(no_of_detections):

    detection_confidence = all_face_locations[0,0,index,2]
    
    if detection_confidence > 0.5:
        current_face_location = all_face_locations[0,0,index, 3:7] * np.array([img_height, img_width, img_width, img_height])
                
        left_x, left_y, right_x, right_y = current_face_location.astype("int")
            
        print("Found face {0} at top: {1} , right:{2}, bottom:{3}, left:{4}".format(index+1,left_y,right_x,right_y,left_x))
        
        current_face_image = image_to_detect[left_y:right_y, left_x:right_x]    
        cv2.imshow("Face number"+str(index+1), current_face_image)
        cv2.rectangle(image_to_detect, (left_x, left_y), (right_x, right_y), (0 , 255 , 0), 2)
        
cv2.imshow("faces in image", image_to_detect)
    
cv2.waitKey(0)
cv2.destroyAllWindows()