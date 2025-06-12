import cv2
import mediapipe
import math
import numpy as np
import time
import json


MINIMUM_EAR = 0.3 #Researchers normally choose 0.2 or 0.3 as the EAR threshold
MAXIMUM_FRAME_COUNT = 10
TTL_TIMER=300 #5min

userdict={}
def check_for_inactive_id():
    global userdict
    to_be_deleted_id=[]
    temp_cur_time=time.time()
    for id in userdict.keys():
        if userdict[id]['TTL']<temp_cur_time:
            to_be_deleted_id.append(id)
    for i in to_be_deleted_id:
        userdict.pop(id, None)       
    print("drowsiness DELETD IDS -> ",to_be_deleted_id)


# landmarks from mesh_map.jpg
LEFT_EYE = [ 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385,384, 398 ]
RIGHT_EYE = [ 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161 , 246 ]

face_mesh = mediapipe.solutions.face_mesh.FaceMesh(static_image_mode=True,max_num_faces=1, min_detection_confidence =0.6, min_tracking_confidence=0.7)


def landmarksDetection(image, results):
    image_height, image_width= image.shape[:2]
    mesh_coordinates = [(int(point.x * image_width), int(point.y * image_height)) for point in results.multi_face_landmarks[0].landmark]
    return mesh_coordinates

# Blinking Ratio
def blinkRatio( landmarks, right_indices, left_indices):

    right_eye_landmark_h1 = landmarks[right_indices[0]]
    right_eye_landmark_h2 = landmarks[right_indices[8]]
    
    right_eye_landmark_v1 = landmarks[right_indices[12]]
    right_eye_landmark_v2 = landmarks[right_indices[4]]

    left_eye_landmark_h1 = landmarks[left_indices[0]]
    left_eye_landmark_h2 = landmarks[left_indices[8]]

    left_eye_landmark_v1 = landmarks[left_indices[12]]
    left_eye_landmark_v2 = landmarks[left_indices[4]]

    right_eye_horizontal_distance = math.dist(right_eye_landmark_h1, right_eye_landmark_h2)
    right_eye_vertical_distance = math.dist(right_eye_landmark_v1, right_eye_landmark_v2)

    left_eye_vertical_distance = math.dist(left_eye_landmark_v1, left_eye_landmark_v2)
    left_eye_horizobtal_distance = math.dist(left_eye_landmark_h1, left_eye_landmark_h2)

    right_eye_ratio = right_eye_vertical_distance/right_eye_horizontal_distance
    left_eye_ratio = left_eye_vertical_distance/left_eye_horizobtal_distance

    eyes_ratio = (right_eye_ratio+left_eye_ratio)/2

    return eyes_ratio


def detect_drowsiness(userid,frame):
    global MINIMUM_EAR,MAXIMUM_FRAME_COUNT,TTL_TIMER,userdict,LEFT_EYE,RIGHT_EYE ,face_mesh

    if userid not in userdict:
        userdict[userid]={"EYE_CLOSED_COUNTER" : 0, 'TTL': time.time()+TTL_TIMER}
    else:
         userdict[userid]["TTL"]+=TTL_TIMER
    
    check_for_inactive_id()

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    try:
        results  = face_mesh.process(rgb_frame)
    except Exception as e:
        print("Drowsiness Detection line 81: ",e)



    output=0   #0 -> No Drowsiness  1-> drowsiness detected  -1-> No face detected
    if results.multi_face_landmarks:
        mesh_coordinatess = landmarksDetection(frame, results)

        eyes_ratio = blinkRatio(mesh_coordinatess, RIGHT_EYE, LEFT_EYE)

        # cv2.putText(frame, "Please blink your eyes",(int(frame_height/2), 100), FONT, 1, (0, 255, 0), 2)

        if eyes_ratio < MINIMUM_EAR:
            userdict[userid]['EYE_CLOSED_COUNTER'] += 1

        else:
            userdict[userid]['EYE_CLOSED_COUNTER'] = 0

        if userdict[userid]['EYE_CLOSED_COUNTER'] >= MAXIMUM_FRAME_COUNT:
            output=1
             
    else:
        output=-1
    
    jsonString=json.dumps(output)
    return jsonString    

if __name__=="__main__":
    video_capture = cv2.VideoCapture(0)

    while True:
        ret, frame = video_capture.read()

        jsonstring=json.loads(detect_drowsiness('abcd',frame))
        text=''

        print(jsonstring,type(jsonstring))
        if jsonstring==0:
            text='AWAKE'
        elif jsonstring==1:
            text='DROWSY'
        elif jsonstring==-1:
            text='NO FACE'
        else:
            text='Undefined'    

        cv2.putText(frame, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow('Liveness Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    video_capture.release()