# -*- coding: utf-8 -*-
"""
@author: abhilash
"""

#importing the required libraries
import cv2
import numpy as np

#capture the video from default camera 
webcam_video_stream = cv2.VideoCapture(0)

#load pretrainned ssd classifier model
face_detection_classifier = cv2.dnn.readNetFromCaffe("models/deploy.prototxt.txt","models/res10_300x300_ssd_iter_140000.caffemodel")

#initialize the array variable to hold all face locations in the frame
all_face_locations = []

#loop through every frame in the video
while True:
    #get the current frame from the video stream as an image
    ret,current_frame = webcam_video_stream.read()
    
    img_height = current_frame.shape[0]
    img_width = current_frame.shape[1]
    
    #resize the image 300x300
    resize_image = cv2.resize(current_frame, (300,300))
    #create blob of the image
    image_to_detect_blob = cv2.dnn.blobFromImage(resize_image, 1.0 , (300,300), (104, 177, 123))
    
    #pass the blob as model input
    face_detection_classifier.setInput(image_to_detect_blob)
    
    #detect all face locations using the ssd classifiar
    all_face_locations = face_detection_classifier.forward()
    
    no_of_detections = all_face_locations.shape[2]

    #looping through the face locations
    for index in range(no_of_detections):

        detection_confidence = all_face_locations[0,0,index,2]
        
        if detection_confidence > 0.5:
            
            current_face_location = all_face_locations[0,0,index, 3:7] * np.array([img_height, img_width, img_width, img_height])
                    
            left_x, left_y, right_x, right_y = current_face_location.astype("int")
            
            #crop video, slicing the current face from main image
            current_face_image = current_frame[left_y:right_y, left_x:right_x]    
            cv2.imshow("Face no"+str(index+1), current_face_image)
            
            #draw rectangle around the face detected
            cv2.rectangle(current_frame,(left_x,left_y),(right_x,right_y),(0,0,255),2)
            
    #showing the current face with rectangle drawn
    cv2.imshow("Realtime Video",current_frame)
        
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#release the stream and cam
#close all opencv windows open
webcam_video_stream.release()
cv2.destroyAllWindows()        










