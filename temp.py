import cv2
import json

from concurrent.futures import ThreadPoolExecutor
import asyncio
from Drowsiness_detection import detect_drowsiness

executor_dash_wake = ThreadPoolExecutor(max_workers=3)
loop = asyncio.get_event_loop()

user_id='abcd'

#JsonDrowsinessString = loop.run_in_executor(executor_dash_wake, detect_drowsiness,user_id,img)

video_capture = cv2.VideoCapture(0)


async def testfunc(frame):
    jsonstring = await loop.run_in_executor(executor_dash_wake, detect_drowsiness,'abcd',frame)
    print(jsonstring)

while True:
    ret, frame1 = video_capture.read()
    ret, frame2 = video_capture.read()
    #jsonstring = testfunc(frame2)
    #jsonstring = testfunc(frame1)
    jsonstring=detect_drowsiness('abcd',frame2)
    jsonstring=detect_drowsiness('abcd',frame1)
    
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

    #cv2.putText(frame, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow('Liveness Detection', frame2)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
video_capture.release()