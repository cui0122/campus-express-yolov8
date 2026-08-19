"""
FastAPI 推理服务
启动：uvicorn main:app --reload --port 8000
"""
import logging
import traceback

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from inference import predict_image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("detect-api")

app = FastAPI(title="校园快递包裹分类识别 API")

# 注意：CORSMiddleware 只会给"正常返回"的响应加 CORS 头。
# 如果视图函数内部抛出未捕获异常，Starlette 会在 CORSMiddleware
# 包装响应之前就把异常转成 500 响应返回，导致该响应缺少
# Access-Control-Allow-Origin 头 —— 浏览器就会把真实的 500 错误
# 显示成一条"CORS policy blocked"的误导性报错。
# 因此这里额外加一个全局异常处理器，确保出错时也带上 CORS 头，
# 并把真实错误信息返回给前端，方便排查。
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request, exc: Exception):
    logger.error("请求处理异常: %s\n%s", exc, traceback.format_exc())
    origin = request.headers.get("origin", "*")
    return JSONResponse(
        status_code=500,
        content={"error": str(exc), "type": type(exc).__name__},
        headers={
            "Access-Control-Allow-Origin": origin,
            "Access-Control-Allow-Credentials": "true",
        },
    )


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        return JSONResponse(
            status_code=400,
            content={"error": f"文件类型不支持: {file.content_type}，请上传图片文件"},
        )

    image_bytes = await file.read()
    if not image_bytes:
        return JSONResponse(status_code=400, content={"error": "上传的文件为空"})

    try:
        detections, (width, height) = predict_image(image_bytes)
    except FileNotFoundError as e:
        # 模型权重文件缺失，这是最常见的 500 成因
        logger.error("模型权重加载失败: %s", e)
        return JSONResponse(status_code=500, content={"error": str(e)})
    except Exception as e:
        logger.error("推理失败: %s\n%s", e, traceback.format_exc())
        return JSONResponse(
            status_code=500,
            content={"error": f"推理过程出错: {e}"},
        )

    return {
        "filename": file.filename,
        "image_width": width,
        "image_height": height,
        "count": len(detections),
        "detections": detections,
    }