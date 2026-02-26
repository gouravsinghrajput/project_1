#----------Libraries-----------------------------
import cv2 as cv 
import mediapipe as mp 
import numpy as np 
import math 
import pygame as py
#------------------------------------------------


cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("cannot open the camera")
    exit()


# -------Enable this to make the window resizable by dragging the corners ---------
# cv.namedWindow('frame', cv.WINDOW_NORMAL)
# cv.resizeWindow('frame', 1000, 900)
#----------------------------------------------------------------------------------

#-----------for hands------------------------
mp_hands = mp.solutions.hands 
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands = 4,
    min_detection_confidence = 0.7, 
    min_tracking_confidence = 0.6
)
#-------------------------------------------
 

while True:
    ret, frame = cap.read()
    if not ret:
        print("cannot collect the frame, exiting the program")
        break
    frame = cv.flip(frame, 1)
    frame = cv.resize(frame, (700, 500)) 
    # color = cv.cvtColor(frame, cv.COLOR_BGR2GRAY) # for Grayscale video
    cv.imshow('frame', frame)

    k = cv.waitKey(1)
    if k == ord("q"):
        break

cap.release()
cv.destroyAllWindows()