from fastapi import FastAPI, UploadFile, File, Request, Query
from fastapi.responses import Response, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2

import asyncio
from concurrent.futures import ThreadPoolExecutor

from processor import process_frame_from_api
from logger import logger

frame_processor = FastAPI()

executor = ThreadPoolExecutor(max_workers=10)

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
    return {"message": "Frame processing API home page"}


@frame_processor.post("/process-frame")
async def process_frame(
    frame: UploadFile = File(...),
):
    try:
        # Read frame
        image_bytes = await frame.read()
        np_arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # Run processing in separate thread for speed
        loop = asyncio.get_event_loop()
        processed_frame = await loop.run_in_executor(
            executor, process_frame_from_api, img, "best.tflite"
        )

        _, encoded_image = cv2.imencode('.jpg', processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        return Response(content=encoded_image.tobytes(), media_type="image/jpeg")

    except Exception as e:
        logger.error(f"Error processing frame: {str(e)}", exc_info=True)
        return JSONResponse(content={"error": str(e)}, status_code=500)

