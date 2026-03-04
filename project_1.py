import cv2 as cv 
import mediapipe as mp
import math 
import numpy as np 
import pygame as py 


cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("cannot open the camera")
    exit()


full_face_mesh = mp.solutions.face_mesh
face_mesh = full_face_mesh.FaceMesh(
    max_num_faces = 2,
    min_detection_confidence = 0.9, 
    min_tracking_confidence = 0.4,
    refine_landmarks = True)


ear_threshold = 0.21
frame_threshold_for_blinking = 1
frame_threshold_for_sleep = 12
blink_counter = 0
total_blinks = 0

py.init()
py.mixer.init()
py.mixer.music.load('arpit_bala_audio_trimmed.mp3')


RIGHT_EYE = [362, 385, 387, 263, 373, 380]
LEFT_EYE = [33, 160, 158, 133, 153, 144]


def EAR_calculation(points_of_the_eye):
    vertical_1 = np.linalg.norm(np.array(points_of_the_eye[1]) - np.array(points_of_the_eye[5]))
    vertical_2 = np.linalg.norm(np.array(points_of_the_eye[2]) - np.array(points_of_the_eye[4]))
    horizontal = np.linalg.norm(np.array(points_of_the_eye[0]) - np.array(points_of_the_eye[3]))
    ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
    return ear



#---------- for hand module -------------
mp_hands = mp.solutions.hands 
mp_draw = mp.solutions.drawing_utils


hands = mp_hands.Hands(
    max_num_hands = 4,
    min_detection_confidence = 0.9, 
    min_tracking_confidence = 0.4
)

# cv.namedWindow('frame', cv.WINDOW_NORMAL)
# cv.resizeWindow('frame', 1000, 900)

canvas = np.zeros((500, 700, 3), dtype=np.uint8)
x_old, y_old = 0, 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("cannot collect the frame, exiting the program")
        break
    frame = cv.flip(frame, 1)
    frame = cv.resize(frame, (700, 500))

    landmark_list = []
    results = hands.process(cv.cvtColor(frame, cv.COLOR_BGR2RGB))

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                frame, hand_landmarks, 
                mp_hands.HAND_CONNECTIONS
            )

            for landmark_id, landmark in enumerate(hand_landmarks.landmark):
                h, w, c = frame.shape 
                cx, cy = int(landmark.x * w), int(landmark.y * h)
                landmark_list.append((landmark_id, (cx, cy)))
            
            
            if landmark_list:
                    landmark_id, (cx, cy) = landmark_list[8]
                    # cv.putText(frame, f'({cx}, {cy})', (cx, cy + 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            if ((cy < landmark_list[5][1][1]) and
                    ((cy < landmark_list[12][1][1])) and 
                    ((cy < landmark_list[16][1][1])) and 
                    ((cy < landmark_list[20][1][1]))):
                        cv.putText(frame, "Index Finger is up", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2) 
            
            
            
            if landmark_list:
                landmark_id, (cx, cy) = landmark_list[10]
                # cv.putText(frame, f'({cx}, {cy})', (cx, cy + 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            if ((cy < landmark_list[4][1][1]) and
                ((cy < landmark_list[8][1][1])) and not 
                ((cy < landmark_list[11][1][1])) and 
                ((cy < landmark_list[16][1][1])) and 
                ((cy < landmark_list[20][1][1]))):
                    cv.putText(frame, "Middle Finger is up", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2) 



            if landmark_list: 
                 landmark_id, (cx, cy) = landmark_list[14]
                # cv.putText(frame, f'({cx}, {cy})', (cx, cy + 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            if ((cy < landmark_list[4][1][1]) and
                ((cy < landmark_list[8][1][1])) and 
                ((cy < landmark_list[12][1][1])) and   
                ((cy > landmark_list[16][1][1])) and 
                ((cy < landmark_list[20][1][1]))):
                    cv.putText(frame, "Ring Finger is up", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


            if landmark_list:
                landmark_id, (cx, cy) = landmark_list[18]
                # cv.putText(frame, f'({cx}, {cy})', (cx, cy + 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            if ((cy < landmark_list[8][1][1]) and
                ((cy < landmark_list[12][1][1])) and 
                ((cy < landmark_list[16][1][1])) and  
                ((cy > landmark_list[20][1][1]))):
                    cv.putText(frame, "Pinky Finger is up", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    if landmark_list:
        (x, y) = landmark_list[8][1]
        (p, q) = landmark_list[12][1]


        if ((landmark_list[8][1][1] < landmark_list[5][1][1]) and
                    ((landmark_list[8][1][1] < landmark_list[12][1][1])) and 
                    ((landmark_list[8][1][1] < landmark_list[16][1][1])) and 
                    ((landmark_list[8][1][1] < landmark_list[20][1][1]))):
            if x_old == 0 and y_old == 0:
                x_old, y_old = x, y

            cv.line(canvas, (x_old, y_old), (x, y), (0, 0, 255), 5)
            x_old, y_old = x, y
        else:
            x_old, y_old = 0, 0


        if (landmark_list[8][1][1]   >  landmark_list[5][1][1] and 
            landmark_list[12][1][1]  > landmark_list[5][1][1] and
            landmark_list[16][1][1]  > landmark_list[5][1][1] and 
            landmark_list[20][1][1]  > landmark_list[5][1][1]):
            cv.circle(canvas, (p, q), 50, (0, 0, 0), -1)
        
            
            


    # gray = cv.cvtColor(canvas, cv.COLOR_BGR2GRAY)
    # _, inverted = cv.threshold(gray, 50, 255, cv.THRESH_BINARY_INV)
    # inverted = cv.cvtColor(inverted, cv.COLOR_GRAY2BGR)

    # frame = cv.bitwise_and(frame, inverted)
    # frame = cv.bitwise_or(frame, canvas)





    h, w, _ = frame.shape
    rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0]


        right_eye_points = []
        for pixel_position in RIGHT_EYE:
            x = int(face_landmarks.landmark[pixel_position].x * w)
            y = int(face_landmarks.landmark[pixel_position].y * h)
            right_eye_points.append((x, y))
            cv.circle(frame, (x, y), 2, (0, 0, 255), -1)

        left_eye_points = []
        for pixel_position in LEFT_EYE:
            x = int(face_landmarks.landmark[pixel_position].x * w)
            y = int(face_landmarks.landmark[pixel_position].y * h)
            left_eye_points.append((x, y))
            cv.circle(frame, (x, y), 2, (0, 0, 255), -1)      


      
        ear_right = EAR_calculation(right_eye_points)
        ear_left = EAR_calculation(left_eye_points)

        if ear_right < ear_threshold and ear_left < ear_threshold:
            blink_counter +=1
        else:
            if blink_counter > frame_threshold_for_blinking:
                total_blinks += 1
                blink_counter = 0
            cv.putText(frame, 'you are awake', (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)    

        if (ear_right < ear_threshold and ear_left < ear_threshold) and (blink_counter > frame_threshold_for_sleep):
            cv.putText(frame, 'you are sleeping', (10, 170), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            if not py.mixer.music.get_busy():
                py.mixer.music.play(-1)

        if ear_right > ear_threshold and ear_left > ear_threshold:
            py.mixer.music.stop()


        cv.putText(frame, f'ear_right: {ear_right:.2f}', (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv.putText(frame, f'ear_left: {ear_left:.2f}', (10, 60), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv.putText(frame, f'total_blinks: {total_blinks}', (10, 90), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

# ---------- HOVER DISPLAY ----------
    canvas_display = canvas.copy()

    if landmark_list:
        index_tip = landmark_list[8][1]
        middle_tip = landmark_list[12][1]

        cv.circle(canvas_display, index_tip, 2, (0, 255, 0), -1)   
        cv.circle(canvas_display, middle_tip, 4, (255, 0, 0), -1)  
        cv.circle(canvas_display, middle_tip, 50, (255, 255, 255), 1)  
        

    cv.imshow('frame', frame)
    cv.imshow('canvas', canvas_display)
    # \cv.imshow('frame', frame)
    # cv.imshow('canvas', canvas)

    k = cv.waitKey(1)
    if k == ord("q"):
        break



cap.release()
cv.destroyAllWindows()

