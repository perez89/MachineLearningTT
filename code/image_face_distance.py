# -*- coding: utf-8 -*-
"""
Created on Sat May  4 00:06:29 2024

@author: lplpe
"""

#import required libraries
import cv2
import face_recognition

image_to_recognize_path = "images/testing/trump.jpg"
original_image = cv2.imread(image_to_recognize_path)

modi_image = face_recognition.load_image_file('images/samples/modi.jpg')
modi_face_encondings = face_recognition.face_encodings(modi_image)[0]


trump_image = face_recognition.load_image_file('images/samples/trump.jpg')
trump_face_encondings = face_recognition.face_encodings(trump_image)[0]

known_face_encoding = [modi_face_encondings, trump_face_encondings]
known_face_names = ["Modi", " Donal Trump"]


image_to_recognize = face_recognition.load_image_file(image_to_recognize_path)
image_to_recognize_encondings = face_recognition.face_encodings(image_to_recognize)[0]

face_distances = face_recognition.face_distance(known_face_encoding, image_to_recognize_encondings)

for i, face_distance in enumerate(face_distances):
    print("The calculated  face distanced is {:.2} from sample {}".format(face_distance, known_face_names[i]))
    print("The matching percentage is {} against the sample of {}".format(round(((1-float(face_distance))*100), 2), known_face_names[i]))