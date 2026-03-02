import cv2 as cv 
import mediapipe as mp 
import numpy as num 
import math 
import pygame as py 


full_face_mesh = mp.solutions.face_mesh 
face_mesh = full_face_mesh.FaceMesh(refine_landmarks = True)


cap = cv.VideoCapture(0)

py.mixer.init()
py.mixer.music.load('arpit_bala_audio_trimmed.mp3')


RIGHT_EYE = [362, 385, 387, 263, 373, 380]
LEFT_EYE = [33, 160, 158, 133, 153, 144]


ear_threshold = 0.17
frame_threshold_for_blinking = 2
frame_threshold_for_sleep = 17
blink_counter = 0
total_number_of_blinks = 0


def ear_calculation(points_of_eye):
    vertical_1 = num.linalg.norm(num.array(points_of_eye[1]) - num.array(points_of_eye[5]))
    vertical_2 = num.linalg.norm(num.array(points_of_eye[2]) - num.array(points_of_eye[4]))
    horizontal = num.linalg.norm(num.array(points_of_eye[0]) - num.array(points_of_eye[3]))
    EAR = (vertical_1 + vertical_2) / (2 * horizontal)
    return EAR 



while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv.flip(frame, 1)

    