# -*- coding: utf-8 -*-
"""
Created on Sat May  4 00:42:10 2024

@author: lplpe
"""

#import required libraries
import cv2
import face_recognition

webcam_video_stream = cv2.VideoCapture("videos/paulo_video_test.mp4")

modi_image = face_recognition.load_image_file('images/samples/modi.jpg')
modi_face_encondings = face_recognition.face_encodings(modi_image)[0]


trump_image = face_recognition.load_image_file('images/samples/trump.jpg')
trump_face_encondings = face_recognition.face_encodings(trump_image)[0]

paulo_image = face_recognition.load_image_file('images/samples/paulo.jpg')
paulo_face_encondings = face_recognition.face_encodings(paulo_image)[0]

known_face_encoding = [modi_face_encondings, trump_face_encondings, paulo_face_encondings]
known_face_names = ["Modi", " Donal Trump" , "Paulo"]

    
all_faces_locations  = []
all_faces_encodings = []

while True:
    ret, current_frame = webcam_video_stream.read()
    current_frame_small = cv2.resize(current_frame, (0, 0), fx = 0.25, fy= 0.25)
    all_faces_locations = face_recognition.face_locations(current_frame_small, number_of_times_to_upsample=2, model="hog")   

    
    #detect face encodings for all faces detected
    all_faces_encodings = face_recognition.face_encodings(current_frame_small, all_faces_locations)
    all_faces_names = []
    print("There are {0} number of faces in image".format(len(all_faces_locations)))

    for current_face_location, current_face_encoding in zip(all_faces_locations, all_faces_encodings):
        top_pos, right_pos, bottom_pos, left_pos = current_face_location
        top_pos = top_pos*4
        right_pos = right_pos*4
        bottom_pos = bottom_pos*4
        left_pos = left_pos*4
        print("Found face at top: {0} , right:{1}, bottom:{2}, left:{3}".format(top_pos,right_pos,bottom_pos,left_pos))
        
        all_matches = face_recognition.compare_faces(known_face_encoding, current_face_encoding)
        
        name_of_person = "Unknow face"
        
        if True in all_matches:
            first_match_index = all_matches.index(True)
            name_of_person = known_face_names[first_match_index]
        
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(current_frame, name_of_person, (left_pos, bottom_pos), font, 0.5, (255,255,255), 1)
        
        cv2.rectangle(current_frame, (left_pos, top_pos), (right_pos,bottom_pos), (255, 0, 0), 2)
           
    cv2.imshow("Webcam video", current_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

webcam_video_stream.release()
cv2.destroyAllWindows()