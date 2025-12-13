from fastapi import FastAPI, WebSocket,File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.handlers.aiSearch_handler import aiSearch_websocket_handler
from src.websocket.ai_search import run_ai_search
from src.store_s3.video_storage import upload_video_to_s3




from concurrent.futures import ThreadPoolExecutor

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- Session Stores ----------------

ai_search={}

detection_executor = ThreadPoolExecutor(max_workers=10)
storage_executor = ThreadPoolExecutor(max_workers=5)


#--------------------------------------------------------------------------- WebSocket for all Models ------------------------------------------------------------------------------#



# ---------------- Theft Detection WebSocket ----------------
@app.websocket("/ws/ai_search/{client_id}")
async def websocket_people(ws: WebSocket, client_id: str):
    await aiSearch_websocket_handler(detection_executor,storage_executor,ws, client_id, ai_search, run_ai_search, "AiSearch")











#--------------------------------------------------------------  Api for video upload -------------------------------------------------------------


# ------------------- Video upload for ai Search -------------------
@app.post("/upload_ai_search_video")
async def upload_ai_search_video(
    video: UploadFile = File(...)      
):  
    """
    API endpoint to upload a video for AI search
    """
    if not video.filename.lower().endswith((".mp4", ".avi", ".mov", ".mkv",".webm")):
        raise HTTPException(status_code=400, detail="Invalid file type. Only video files are allowed.")

    url = await upload_video_to_s3(video)
    return {"message": "Video uploaded successfully", "s3_url": url}