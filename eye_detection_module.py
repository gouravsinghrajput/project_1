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

    h, w, _ = frame.shape
    rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    result = face_mesh.process(rgb_frame)

    if result.multi_face_landmarks:
        face_landmarks = result.multi_face_landmarks[0]

        right_eye_points = [] 
        for pixel_position in RIGHT_EYE:
            x = int(face_landmarks.landmark[pixel_position].x * w)
            y = int(face_landmarks.landmark[pixel_position].y * h)
            right_eye_points.append((x,y))
            cv.circle(frame, (x, y), 2, (0, 0, 255), -1)

        left_eye_points = [] 
        for pixel_position in LEFT_EYE:
            x = int(face_landmarks.landmark[pixel_position].x * w)
            y = int(face_landmarks.landmark[pixel_position].y * h)
            left_eye_points.append((x,y))
            cv.circle(frame, (x, y), 2, (0, 0, 255), -1)

        
        ear_right = ear_calculation(right_eye_points)
        ear_left = ear_calculation(left_eye_points)


        if ear_right < ear_threshold and ear_left < ear_threshold:
            blink_counter += 1
        else:
            if blink_counter > frame_threshold_for_blinking:
                total_blinks +=1
                blink_counter = 0
            cv.putText(frame, "You are awake!!", (50, 50), cv.FONT_ITALIC, 2, (0, 255, 0), 4)
        if (ear_right < ear_threshold and ear_left < ear_threshold) and blink_counter > frame_threshold_for_sleep:
            cv.putText(frame, "UHTJAAA", (10, 200), cv.FONT_ITALIC, 2, (0, 0, 255), 4)
            print("Uthjaaaa....,  uthjaaaaa  oooooyyeeeeee")
            if not py.mixer.music.get_busy(): 
                py.mixer.music.play()
            else:
                py.mixer.music.stop()


        cv.putText(frame, f"EAR_RIGHT: {round(ear_right, 2)}", (10, 30), cv.FONT_ITALIC, 0.5, (255, 0, 0), 2)
        cv.putText(frame, f"EAR_LEFT: {round(ear_left, 2)}", (10, 60), cv.FONT_ITALIC, 0.5, (255, 0, 0), 2)
        cv.putText(frame, f"Blinks: {total_blinks}", (10, 120), cv.FONT_ITALIC, 0.75, (255, 0, 0), 2)


    cv.imshow('Eye Detection', frame)


    k = cv.waitKey(1)
    if k == ord('q'):
        break 

cap.release()
cv.destroyAllWindows()