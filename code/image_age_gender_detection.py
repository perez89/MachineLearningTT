# -*- coding: utf-8 -*-
"""
Created on Sat May  4 00:06:29 2024

@author: lplpe
"""

#import required libraries
import cv2
import face_recognition

gender_label_list = ["Male", "Female"]
gender_protext = "dataset/gender_deploy.prototxt"
gender_caffemodel = "dataset/gender_net.caffemodel"

age_label_list = ["(0-2)", "(4-6)", "(8-12)", "(15-20)", "(25-32)", "(38-43)", "(48-53)", "(60-100)"]
age_protext = "dataset/age_deploy.prototxt"
age_caffemodel = "dataset/age_net.caffemodel"

image_to_detect = cv2.imread('images/rosto3.jpg')

#cv2.imshow("test", image_to_detect)


all_face_locations = face_recognition.face_locations(image_to_detect, number_of_times_to_upsample=3, model="hog")

print("There are {0} number of faces in image".format(len(all_face_locations)))

for index, current_face_location in enumerate(all_face_locations):
    top_pos, right_pos, bottom_pos, left_pos = current_face_location
    print("Found face {0} at top: {1} , right:{2}, bottom:{3}, left:{4}".format(index+1,top_pos,right_pos,bottom_pos,left_pos))
    current_face_image = image_to_detect[top_pos:bottom_pos, left_pos:right_pos]    
    cv2.imshow("Face number"+str(index+1), current_face_image)
    
    AGE_GENDER_MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744,114.895847746)
    
    current_face_image_blob = cv2.dnn.blobFromImage(current_face_image, 1, (227,227), AGE_GENDER_MODEL_MEAN_VALUES, swapRB=False)
    
    #Predicting gender
    gender_cov_net = cv2.dnn.readNet(gender_caffemodel, gender_protext)
    gender_cov_net.setInput(current_face_image_blob)
    
    gender_predictions = gender_cov_net.forward()
    gender = gender_label_list[gender_predictions[0].argmax()]
    
    #Predicting age
    age_cov_net = cv2.dnn.readNet(age_caffemodel, age_protext)
    age_cov_net.setInput(current_face_image_blob)
    
    age_predictions = age_cov_net.forward()
    age = age_label_list[age_predictions[0].argmax()]        
    
    cv2.rectangle(image_to_detect, (left_pos, top_pos), (right_pos,bottom_pos), (0,0,255), 2)
    
    #display agr and gender as text in the image
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(image_to_detect, gender+ " "+ age + " years", (left_pos, bottom_pos), font, 0.5, (0,255, 0), 1)
    
cv2.imshow("Age and Gender video", image_to_detect)
#cv2.waitKey(0)
#cv2.destroyAllWindows()