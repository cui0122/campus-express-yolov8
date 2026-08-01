"""
离线数据增强：只对 train 集做增强（val/test 保持原始，避免评估失真）。
重点针对小尺寸包裹目标：随机缩放贴近远景、HSV扰动、亮度/对比度变化、水平翻转。
YOLOv8 训练时自带在线 Mosaic/HSV 等增强，这里的离线增强是作为"数据量不够时"的补充，
如果你的数据集已经有 2000+ 张，可以跳过这一步，只依赖 YOLOv8 内置在线增强。
"""
import os
import random

import albumentations as A
import cv2
from tqdm import tqdm

IMG_DIR = "data/dataset/images/train"
LBL_DIR = "data/dataset/labels/train"
AUG_PER_IMAGE = 2  # 每张原图生成几张增强图

transform = A.Compose(
    [
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=25, val_shift_limit=15, p=0.5),
        A.RandomScale(scale_limit=(-0.3, 0.1), p=0.4),  # 偏向缩小，模拟远景小目标
        A.MotionBlur(blur_limit=3, p=0.2),
        A.GaussNoise(p=0.2),
    ],
    bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]),
)


def read_yolo_labels(path):
    boxes, labels = [], []
    if not os.path.exists(path):
        return boxes, labels
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls, x, y, w, h = parts
            boxes.append([float(x), float(y), float(w), float(h)])
            labels.append(int(float(cls)))
    return boxes, labels


def main():
    img_files = [f for f in os.listdir(IMG_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    print(f"待增强图像数: {len(img_files)}")

    for img_file in tqdm(img_files, desc="数据增强中", unit="张"):
        stem, ext = os.path.splitext(img_file)
        img_path = os.path.join(IMG_DIR, img_file)
        lbl_path = os.path.join(LBL_DIR, stem + ".txt")

        image = cv2.imread(img_path)
        if image is None:
            continue
        boxes, labels = read_yolo_labels(lbl_path)
        if not boxes:
            continue

        for i in range(AUG_PER_IMAGE):
            try:
                augmented = transform(image=image, bboxes=boxes, class_labels=labels)
            except Exception:
                continue

            new_stem = f"{stem}_aug{i}"
            cv2.imwrite(os.path.join(IMG_DIR, new_stem + ext), augmented["image"])

            lines = [
                f"{cls} {b[0]:.6f} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f}"
                for cls, b in zip(augmented["class_labels"], augmented["bboxes"])
            ]
            with open(os.path.join(LBL_DIR, new_stem + ".txt"), "w", encoding="utf-8") as f:
                f.write("\n".join(lines))

    total = len([f for f in os.listdir(IMG_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png"))])
    print(f"[OK] 增强完成，train 集图像总数: {total}")


if __name__ == "__main__":
    random.seed(42)
    main()