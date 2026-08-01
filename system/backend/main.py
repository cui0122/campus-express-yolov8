"""
FastAPI 推理服务
启动：uvicorn main:app --reload --port 8000
"""
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from inference import predict_image

app = FastAPI(title="校园快递包裹分类识别 API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    image_bytes = await file.read()
    detections, (width, height) = predict_image(image_bytes)
    return {
        "filename": file.filename,
        "image_width": width,
        "image_height": height,
        "count": len(detections),
        "detections": detections,
    }
