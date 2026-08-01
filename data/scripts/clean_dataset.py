"""
数据清洗：
1. 剔除无法打开/损坏的图像
2. 剔除空标注（没有任何框）的样本（除非明确想留作负样本）
3. 基于文件哈希去重
"""
import hashlib
import os

from PIL import Image

SRC_DIR = "data/merged"          # merge_datasets.py 的输出
IMG_DIR = os.path.join(SRC_DIR, "images")
LBL_DIR = os.path.join(SRC_DIR, "labels")

DROP_EMPTY_LABELS = True


def file_hash(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def main():
    seen_hash = set()
    removed_broken, removed_dup, removed_empty = 0, 0, 0

    for img_file in list(os.listdir(IMG_DIR)):
        img_path = os.path.join(IMG_DIR, img_file)
        stem = os.path.splitext(img_file)[0]
        lbl_path = os.path.join(LBL_DIR, stem + ".txt")

        try:
            with Image.open(img_path) as im:
                im.verify()
        except Exception:
            os.remove(img_path)
            if os.path.exists(lbl_path):
                os.remove(lbl_path)
            removed_broken += 1
            continue

        h = file_hash(img_path)
        if h in seen_hash:
            os.remove(img_path)
            if os.path.exists(lbl_path):
                os.remove(lbl_path)
            removed_dup += 1
            continue
        seen_hash.add(h)

        if DROP_EMPTY_LABELS:
            if not os.path.exists(lbl_path) or os.path.getsize(lbl_path) == 0:
                os.remove(img_path)
                if os.path.exists(lbl_path):
                    os.remove(lbl_path)
                removed_empty += 1
                continue

    remaining = len(os.listdir(IMG_DIR))
    print(f"[OK] 清洗完成。剩余 {remaining} 张图像。")
    print(f"  剔除损坏图像: {removed_broken}")
    print(f"  剔除重复图像: {removed_dup}")
    print(f"  剔除空标注: {removed_empty}")


if __name__ == "__main__":
    main()
