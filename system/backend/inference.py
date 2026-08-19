"""
模型加载与推理封装，供 FastAPI 后端调用。
"""
import io
import os

from PIL import Image
from ultralytics import YOLO

# 默认权重路径改为基于本文件所在目录的绝对路径，
# 避免因为 uvicorn 启动时的当前工作目录不同而导致找不到文件。
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_PATH = os.environ.get(
    "MODEL_WEIGHTS", os.path.join(_THIS_DIR, "models", "best.pt")
)
CONF_THRES = float(os.environ.get("CONF_THRES", 0.35))

_model = None


def get_model():
    global _model
    if _model is None:
        if not os.path.exists(WEIGHTS_PATH):
            raise FileNotFoundError(
                f"未找到模型权重: {WEIGHTS_PATH}，请把训练好的 best.pt 拷贝到该路径，"
                "或设置环境变量 MODEL_WEIGHTS 指向权重文件。"
            )
        _model = YOLO(WEIGHTS_PATH)
    return _model


def predict_image(image_bytes: bytes):
    """输入图像字节流，返回检测结果列表：
    [{"cls": 类别名, "conf": 置信度, "box": [x1,y1,x2,y2]}, ...]
    """
    model = get_model()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    results = model.predict(image, conf=CONF_THRES, verbose=False)
    result = results[0]

    detections = []
    names = result.names
    for box in result.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
        detections.append(
            {
                "cls": names[cls_id],
                "conf": round(conf, 4),
                "box": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
            }
        )
    return detections, image.size