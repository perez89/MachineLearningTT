# -*- coding: utf-8 -*-
"""
Created on Sat May  4 00:06:29 2024

@author: lplpe
"""

#import required libraries
import cv2
import face_recognition

original_image = cv2.imread('images/testing/trump-modi-unknown.jpg')

modi_image = face_recognition.load_image_file('images/samples/modi.jpg')
modi_face_encondings = face_recognition.face_encodings(modi_image)[0]


trump_image = face_recognition.load_image_file('images/samples/trump.jpg')
trump_face_encondings = face_recognition.face_encodings(trump_image)[0]

known_face_encoding = [modi_face_encondings, trump_face_encondings]
known_face_names = ["Modi", " Donal Trump"]


image_to_recognize = face_recognition.load_image_file('images/testing/trump-modi-unknown.jpg')

#detect all faces in the image
all_faces_locations = face_recognition.face_locations(image_to_recognize, number_of_times_to_upsample=2, model="hog")

#detect face encodings for all faces detected
all_faces_encodings = face_recognition.face_encodings(image_to_recognize, all_faces_locations)

print("There are {0} number of faces in image".format(len(all_faces_locations)))

for current_face_location, current_face_encoding in zip(all_faces_locations, all_faces_encodings):
    top_pos, right_pos, bottom_pos, left_pos = current_face_location
    print("Found face at top: {0} , right:{1}, bottom:{2}, left:{3}".format(top_pos,right_pos,bottom_pos,left_pos))
    
    all_matches = face_recognition.compare_faces(known_face_encoding, current_face_encoding)
    
    name_of_person = "Unknow face"
    
    if True in all_matches:
        first_match_index = all_matches.index(True)
        name_of_person = known_face_names[first_match_index]
    
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(original_image, name_of_person, (left_pos, bottom_pos), font, 0.5, (255,255,255), 1)
    
    cv2.rectangle(original_image, (left_pos, top_pos), (right_pos,bottom_pos), (255, 0, 0), 2)

cv2.imshow("Identify faces", original_image)
#cv2.waitKey(0)
#cv2.destroyAllWindows()