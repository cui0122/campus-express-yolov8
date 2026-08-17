"""
把 Roboflow 数据集整理成本项目统一的 YOLO 格式
增加功能：自动清理/跳过 public_raw 中混入的 foambox 与 plasticbag 前缀文件
"""
import argparse
import os
import re
import shutil
from collections import Counter

import yaml
from tqdm import tqdm

FINAL_CLASSES = ["纸箱", "塑料袋", "泡沫箱"]

# 需要过滤/剔除的文件前缀（自动转小写比对）
EXCLUDE_PREFIXES = ("foambox", "plasticbag")

CLASS_MAP = {
    "boxes": "纸箱",
    "parcel": "纸箱",
    "good-parcel": "纸箱",
    "package": "纸箱",
    "box": "纸箱",
    "cardboard": "纸箱",
    "cardboard box": "纸箱",
    "corrugated carton": "纸箱",
    "plastic bag": "塑料袋",
    "single-use carrier bag": "塑料袋",
    "polypropylene bag": "塑料袋",
    "courier bag": "塑料袋",
    "poly mailer": "塑料袋",
    "foam food container": "泡沫箱",
    "styrofoam": "泡沫箱",
    "styrofam piece": "泡沫箱",
    "foam box": "泡沫箱",
    "label": None,
    "person": None,
}


def normalize(name):
    return re.sub(r"[\s_\-]+", " ", str(name).strip().lower())


def find_data_yaml(src_dir):
    for root, _, files in os.walk(src_dir):
        for f in files:
            if f in ("data.yaml", "data.yml"):
                return os.path.join(root, f)
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default="data/public_raw", help="数据集目录")
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
    src_classes = cfg["names"]
    if isinstance(src_classes, dict):
        src_classes = [src_classes[i] for i in sorted(src_classes)]
    print(f"检测到原始类别: {src_classes}")

    norm_map = {normalize(k): v for k, v in CLASS_MAP.items()}
    final_id = {c: i for i, c in enumerate(FINAL_CLASSES)}

    old_id_to_new_id = {}
    unmapped = []
    for old_id, old_name in enumerate(src_classes):
        mapped = norm_map.get(normalize(old_name))
        if mapped is None or mapped not in final_id:
            unmapped.append(old_name)
            continue
        old_id_to_new_id[old_id] = final_id[mapped]

    print(f"类别映射结果: { {src_classes[i]: FINAL_CLASSES[old_id_to_new_id[i]] for i in old_id_to_new_id} }")
    if unmapped:
        print(f"⚠️ 以下原始类别未在 CLASS_MAP 中映射到任何最终类别，对应的框会被丢弃: {unmapped}")

    out_img = os.path.join(args.out, "images")
    out_lbl = os.path.join(args.out, "labels")
    os.makedirs(out_img, exist_ok=True)
    os.makedirs(out_lbl, exist_ok=True)

    total = 0
    deleted_prefix_count = 0
    dropped_empty_after_remap = 0
    remapped_box_counter = Counter()

    for split in ("train", "valid", "test", "val"):
        split_img_dir = os.path.join(base_dir, split, "images")
        split_lbl_dir = os.path.join(base_dir, split, "labels")
        if not os.path.isdir(split_img_dir):
            continue

        img_files = [f for f in os.listdir(split_img_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        for img_file in tqdm(img_files, desc=f"整理 {split} 子集"):
            stem = os.path.splitext(img_file)[0]
            
            # --- 新增逻辑：检测并清理混入的前缀文件 ---
            if img_file.lower().startswith(EXCLUDE_PREFIXES):
                src_img_path = os.path.join(split_img_dir, img_file)
                src_lbl_path = os.path.join(split_lbl_dir, stem + ".txt")
                
                # 从磁盘中彻底删除这些混入的源图片与标注
                if os.path.exists(src_img_path):
                    os.remove(src_img_path)
                if os.path.exists(src_lbl_path):
                    os.remove(src_lbl_path)
                    
                deleted_prefix_count += 1
                continue  # 跳过不复制到 public_yolo
            # ---------------------------------------

            new_stem = f"{split}_{stem}"
            lbl_file = os.path.join(split_lbl_dir, stem + ".txt")
            new_lines = []
            
            if os.path.exists(lbl_file):
                with open(lbl_file, encoding="utf-8") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) != 5:
                            continue
                        old_cid = int(float(parts[0]))
                        if old_cid not in old_id_to_new_id:
                            continue
                        new_cid = old_id_to_new_id[old_cid]
                        new_lines.append(f"{new_cid} {' '.join(parts[1:])}")
                        remapped_box_counter[new_cid] += 1

            if os.path.exists(lbl_file) and not new_lines:
                dropped_empty_after_remap += 1

            shutil.copy(
                os.path.join(split_img_dir, img_file),
                os.path.join(out_img, new_stem + os.path.splitext(img_file)[1]),
            )
            with open(os.path.join(out_lbl, new_stem + ".txt"), "w", encoding="utf-8") as fo:
                fo.write("\n".join(new_lines))
            total += 1

    with open(os.path.join(args.out, "classes.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(FINAL_CLASSES))

    print(f"\n[清理过滤统计]")
    print(f"🗑️ 自动彻底删除/剔除混入的前缀图片及标注: {deleted_prefix_count} 张")

    print(f"\n[各最终类别实际保留框数统计]")
    for i, name in enumerate(FINAL_CLASSES):
        n = remapped_box_counter.get(i, 0)
        mark = "✅" if n > 0 else "⚠️ 0个"
        print(f"  {name}: {n}  {mark}")

    print(f"\n[OK] 共整理有效图片 {total} 张 -> {args.out}")


if __name__ == "__main__":
    main()