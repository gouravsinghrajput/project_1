import cv2 as cv 
import mediapipe as mp
import math 
import numpy as np 
import pygame as py 


cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("cannot open the camera")
    exit()
#---------- for hand module -------------
mp_hands = mp.solutions.hands 
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands = 4,
    min_detection_confidence = 0.7, 
    min_tracking_confidence = 0.6
)

while True:
    ret, frame = cap.read()
    if not ret:
        print("cannot collect the frame, exiting the program")
        break
    frame = cv.flip(frame, 1)
    frame = cv.resize(frame, (700, 500))

    results = hands.process(frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                frame, hand_landmarks, 
                # mp_hands.HAND_CONNECTIONS   #enable this to show lines between the points on the hands.
            )

            landmark_list = []
            for landmark_id, landmark in enumerate(hand_landmarks.landmark):
                h, w, c = frame.shape 
                cx, cy = int(landmark.x * w), int(landmark.y * h)
                landmark_list.append((landmark_id, (cx, cy)))
            
            
            #------logic for index finger up --------------
            if landmark_list:
                landmark_id, (cx, cy) = landmark_list[8]
                #the below line is to show the coordinates if index finger tip.
                # cv.putText(frame, f'({cx}, {cy})', (cx, cy + 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            if ((cy < landmark_list[5][1][1]) and
                ((cy < landmark_list[12][1][1])) and 
                ((cy < landmark_list[16][1][1])) and 
                ((cy < landmark_list[20][1][1]))):
                    cv.putText(frame, "Index Finger is up", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            #----------------------------------------------


            #------logic for middle finger up ---------------
            if landmark_list:
                landmark_id, (cx, cy) = landmark_list[10]
                # cv.putText(frame, f'({cx}, {cy})', (cx, cy + 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            if ((cy < landmark_list[4][1][1]) and
                ((cy < landmark_list[8][1][1])) and not 
                ((cy < landmark_list[11][1][1])) and 
                ((cy < landmark_list[16][1][1])) and 
                ((cy < landmark_list[20][1][1]))):
                    cv.putText(frame, "Middle Finger is up", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            #-----------------------------------------------
             

            #-----------for ring finger -------------------
            if landmark_list: 
                 landmark_id, (cx, cy) = landmark_list[14]
                # cv.putText(frame, f'({cx}, {cy})', (cx, cy + 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            if ((cy < landmark_list[4][1][1]) and
                ((cy < landmark_list[8][1][1])) and 
                ((cy < landmark_list[12][1][1])) and   
                ((cy > landmark_list[16][1][1])) and 
                ((cy < landmark_list[20][1][1]))):
                    cv.putText(frame, "Ring Finger is up", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            #-----------------------------------------------


                    


    # color = cv.cvtColor(frame, cv.COLOR_BGR2BGR555)
    cv.imshow('frame', frame)

    k = cv.waitKey(1)
    if k == ord("q"):
        break

cap.release()
cv.destroyAllWindows()