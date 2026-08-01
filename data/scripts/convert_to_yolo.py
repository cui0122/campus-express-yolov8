"""
【备用脚本】把 VOC(xml) 或 COCO(json) 格式标注转换成 YOLO txt 格式。
如果你在 Roboflow 下载时直接选择了 "YOLOv8" 导出格式，不需要跑这个脚本，
直接用 prepare_public_dataset.py 即可。

用法：
python convert_to_yolo.py --src data/public_raw --format voc --out data/public_yolo
python convert_to_yolo.py --src data/public_raw --format coco --out data/public_yolo
"""
import argparse
import json
import os
import shutil
import xml.etree.ElementTree as ET

from tqdm import tqdm


def convert_voc(src_dir, out_dir):
    """期望结构: src_dir/Annotations/*.xml, src_dir/JPEGImages/*.jpg"""
    ann_dir = os.path.join(src_dir, "Annotations")
    img_dir = os.path.join(src_dir, "JPEGImages")
    out_img = os.path.join(out_dir, "images")
    out_lbl = os.path.join(out_dir, "labels")
    os.makedirs(out_img, exist_ok=True)
    os.makedirs(out_lbl, exist_ok=True)

    classes = set()
    xml_files = [f for f in os.listdir(ann_dir) if f.endswith(".xml")]
    for f in xml_files:
        tree = ET.parse(os.path.join(ann_dir, f))
        for obj in tree.findall("object"):
            classes.add(obj.find("name").text.strip())
    classes = sorted(classes)
    cls2id = {c: i for i, c in enumerate(classes)}
    print(f"检测到类别: {cls2id}")

    for f in tqdm(xml_files, desc="VOC->YOLO"):
        tree = ET.parse(os.path.join(ann_dir, f))
        root = tree.getroot()
        size = root.find("size")
        w, h = int(size.find("width").text), int(size.find("height").text)

        lines = []
        for obj in root.findall("object"):
            cls = obj.find("name").text.strip()
            box = obj.find("bndbox")
            xmin = float(box.find("xmin").text)
            ymin = float(box.find("ymin").text)
            xmax = float(box.find("xmax").text)
            ymax = float(box.find("ymax").text)
            xc = (xmin + xmax) / 2 / w
            yc = (ymin + ymax) / 2 / h
            bw = (xmax - xmin) / w
            bh = (ymax - ymin) / h
            lines.append(f"{cls2id[cls]} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

        stem = os.path.splitext(f)[0]
        with open(os.path.join(out_lbl, stem + ".txt"), "w") as fo:
            fo.write("\n".join(lines))

        for ext in (".jpg", ".jpeg", ".png"):
            src_img = os.path.join(img_dir, stem + ext)
            if os.path.exists(src_img):
                shutil.copy(src_img, os.path.join(out_img, stem + ext))
                break

    with open(os.path.join(out_dir, "classes.txt"), "w") as f:
        f.write("\n".join(classes))


def convert_coco(src_dir, out_dir):
    """期望结构: src_dir/_annotations.coco.json + 图像在同目录"""
    json_path = None
    for f in os.listdir(src_dir):
        if f.endswith(".json"):
            json_path = os.path.join(src_dir, f)
            break
    if not json_path:
        raise SystemExit(f"未在 {src_dir} 找到 coco json 标注文件")

    with open(json_path) as f:
        coco = json.load(f)

    cat_id2name = {c["id"]: c["name"] for c in coco["categories"]}
    cat_ids_sorted = sorted(cat_id2name.keys())
    catid2yoloid = {cid: i for i, cid in enumerate(cat_ids_sorted)}
    print(f"检测到类别: {[cat_id2name[c] for c in cat_ids_sorted]}")

    img_id2info = {img["id"]: img for img in coco["images"]}
    img_id2anns = {}
    for ann in coco["annotations"]:
        img_id2anns.setdefault(ann["image_id"], []).append(ann)

    out_img = os.path.join(out_dir, "images")
    out_lbl = os.path.join(out_dir, "labels")
    os.makedirs(out_img, exist_ok=True)
    os.makedirs(out_lbl, exist_ok=True)

    for img_id, info in tqdm(img_id2info.items(), desc="COCO->YOLO"):
        w, h = info["width"], info["height"]
        stem = os.path.splitext(info["file_name"])[0]
        lines = []
        for ann in img_id2anns.get(img_id, []):
            x, y, bw, bh = ann["bbox"]
            xc = (x + bw / 2) / w
            yc = (y + bh / 2) / h
            nbw = bw / w
            nbh = bh / h
            yolo_cls = catid2yoloid[ann["category_id"]]
            lines.append(f"{yolo_cls} {xc:.6f} {yc:.6f} {nbw:.6f} {nbh:.6f}")

        with open(os.path.join(out_lbl, stem + ".txt"), "w") as fo:
            fo.write("\n".join(lines))

        src_img = os.path.join(src_dir, info["file_name"])
        if os.path.exists(src_img):
            shutil.copy(src_img, os.path.join(out_img, info["file_name"]))

    with open(os.path.join(out_dir, "classes.txt"), "w") as f:
        f.write("\n".join(cat_id2name[c] for c in cat_ids_sorted))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True)
    parser.add_argument("--format", choices=["voc", "coco"], required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    if args.format == "voc":
        convert_voc(args.src, args.out)
    else:
        convert_coco(args.src, args.out)
    print(f"[OK] 转换完成，输出目录: {args.out}")


if __name__ == "__main__":
    main()
