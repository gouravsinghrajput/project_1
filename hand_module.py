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
    # color = cv.cvtColor(frame, cv.COLOR_BGR2BGR555)
    cv.imshow('frame', frame)

    k = cv.waitKey(1)
    if k == ord("q"):
        break

cap.release()
cv.destroyAllWindows()