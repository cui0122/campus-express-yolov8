"""
把「网页手动下载并解压的 Roboflow 数据集」整理成本项目统一的 YOLO 格式（images/ + labels/ + classes.txt）。

使用前提（全程点鼠标，不需要 API Key）：
1. 打开 Roboflow Universe 上的数据集页面
2. 点 "Download Dataset" -> 格式选 "YOLOv8" -> 下载方式选 "download zip to computer"
3. 解压后把整个文件夹放到 data/public_raw/ 下（保留它自带的 data.yaml 和
   train/valid/test 子目录结构，不需要手动整理）

运行：
python prepare_public_dataset.py --src data/public_raw --out data/public_yolo
"""
import argparse
import os
import shutil

import yaml


def find_data_yaml(src_dir):
    for root, _, files in os.walk(src_dir):
        for f in files:
            if f in ("data.yaml", "data.yml"):
                return os.path.join(root, f)
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default="data/public_raw", help="解压后的 Roboflow 数据集目录")
    parser.add_argument("--out", default="data/public_yolo", help="整理后的输出目录")
    args = parser.parse_args()

    yaml_path = find_data_yaml(args.src)
    if not yaml_path:
        raise SystemExit(
            f"在 {args.src} 下没找到 data.yaml，请确认已经把 Roboflow 下载的 zip 完整解压到这里。"
        )
    base_dir = os.path.dirname(yaml_path)

    with open(yaml_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    classes = cfg["names"]
    if isinstance(classes, dict):  # 有的导出格式是 {0: 'box', 1: 'envelope'}
        classes = [classes[i] for i in sorted(classes)]
    print(f"检测到类别: {classes}")

    out_img = os.path.join(args.out, "images")
    out_lbl = os.path.join(args.out, "labels")
    os.makedirs(out_img, exist_ok=True)
    os.makedirs(out_lbl, exist_ok=True)

    total = 0
    for split in ("train", "valid", "test", "val"):
        split_img_dir = os.path.join(base_dir, split, "images")
        split_lbl_dir = os.path.join(base_dir, split, "labels")
        if not os.path.isdir(split_img_dir):
            continue

        for img_file in os.listdir(split_img_dir):
            if not img_file.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            stem = os.path.splitext(img_file)[0]
            new_stem = f"{split}_{stem}"

            shutil.copy(
                os.path.join(split_img_dir, img_file),
                os.path.join(out_img, new_stem + os.path.splitext(img_file)[1]),
            )
            lbl_file = os.path.join(split_lbl_dir, stem + ".txt")
            if os.path.exists(lbl_file):
                shutil.copy(lbl_file, os.path.join(out_lbl, new_stem + ".txt"))
            else:
                # 没有标注文件说明这张图没有目标框，补一个空txt（负样本）
                open(os.path.join(out_lbl, new_stem + ".txt"), "w").close()
            total += 1

    with open(os.path.join(args.out, "classes.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(classes))

    print(f"[OK] 共整理 {total} 张图像 -> {args.out}")
    print("下一步：打开 data/scripts/merge_datasets.py，按上面打印出的类别名配置 CLASS_MAP。")


if __name__ == "__main__":
    main()
