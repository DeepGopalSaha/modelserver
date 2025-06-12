from fastapi import FastAPI, UploadFile, File, Request, Query
from fastapi.responses import Response, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import cv2

import asyncio
from concurrent.futures import ThreadPoolExecutor

from processor import process_frame_from_api

import custom_map_navigation
import placesofinterest
from Drowsiness_detection import detect_drowsiness

from logger import logger




frame_processor = FastAPI()

# Define Pydantic model for input JSON
class LocationRequest(BaseModel):
    lat: float
    lon: float

executor_dash_nav = ThreadPoolExecutor(max_workers=4)
executor_dash_wake = ThreadPoolExecutor(max_workers=3)

# CORS settings
frame_processor.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@frame_processor.get("/")
def home():
    return {"message": "Backend API home page"}

@frame_processor.post("/wakeup-backend")
async def check_backend_availability():
    return JSONResponse(content='True',status_code=200)

@frame_processor.post("/process-frame")
async def process_frame(
    frame: UploadFile = File(...),
    scaleh=-1,
    scalew=-1
):
    try:
        # Read frame
        image_bytes = await frame.read()
        np_arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # Run processing in separate thread for speed
        loop = asyncio.get_event_loop()
        processed_frame = await loop.run_in_executor(
            executor_dash_nav, process_frame_from_api, img,[float(scaleh),float(scalew)] 
        )

        #print(processed_frame, "@")

        #_, encoded_image = cv2.imencode('.jpg', processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        return Response(content=processed_frame, media_type="application/json")

    except Exception as e:
        logger.error(f"Error processing frame: {str(e)}", exc_info=True)
        return JSONResponse(content={"error": str(e)}, status_code=500)


@frame_processor.post("/get-directions")
async def get_direction_distance(curlat, curlng,id):
    latitude = float(curlat)
    longitude = float(curlng)

    jsonDirectionString = custom_map_navigation.check_map([latitude, longitude],id=id)

    return jsonDirectionString

@frame_processor.post("/new-map")
async def get_new_map(startlat, startlng, endlat, endlng, id):
    #receive order longitude, latitude
    startloc_list=list(map(float,[startlat, startlng]))
    endloc_list=list(map(float,[endlat, endlng]))
    #start_latitude = startlocation.lat
    #start_longitude = startlocation.lon
    #end_latitude = endlocation.lat
    #end_longitude = endlocation.lon

    jsonWaypointsString = custom_map_navigation.create_map(startloc_list, endloc_list, id= id)

    return jsonWaypointsString

@frame_processor.post("/clear-map")
async def clear_map(id=None):
    custom_map_navigation.clear_map(id)

@frame_processor.get("/get-poi")
async def get_poi(curlat, curlng,id=-1, response_type="%2A", dist =5000, limit=10):
    latitude = float(curlat)
    longitude = float(curlng)
    print(curlat,curlng,id,response_type,dist,limit)

    try:
        response_status,jsonPOIString = placesofinterest.poi(curlat=latitude, curlng=longitude,id=int(id),responsetype=str(response_type), dist=min(30000,int(dist)), limit=min(50,int(limit)))
    except Exception as e:
        print(r"/get-poi -> ",e)

    return JSONResponse(content=jsonPOIString, status_code=response_status)


@frame_processor.post("/detect-drowsiness")
async def process_frame(
    frame: UploadFile = File(...),
    user_id = 0
    ):
    try:
        # Read frame
        image_bytes = await frame.read()
        np_arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # Run processing in separate thread for speed
        loop = asyncio.get_event_loop()
        JsonDrowsinessString = await loop.run_in_executor(executor_dash_wake, detect_drowsiness,user_id,img)

        '''
        JsonDrowsinessString    0 -> no drowsiness
                                1 -> Drowsiness Detected
                               -1 -> No Face Detected
        '''
        return JSONResponse(content=JsonDrowsinessString, status_code=200)
    
    except Exception as e:
        logger.error(f"Error processing frame: {str(e)}", exc_info=True)
        return JSONResponse(content={"error": str(e)}, status_code=500)