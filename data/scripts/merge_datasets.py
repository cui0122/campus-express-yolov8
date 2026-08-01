"""
合并「公开数据集（YOLO格式）」+「校园自采数据集（YOLO格式）」，统一类别映射。

★★★ 使用前必须修改下面的 CLASS_MAP ★★★
key 是原始数据集里出现的类别名（不区分大小写），value 是你项目最终想用的类别名。
把不需要的类别映射成 None，脚本会自动丢弃这些框（如果一张图的所有框都被丢弃，
这张图会被跳过，除非 KEEP_NEGATIVE_IMAGES=True）。

最终类别顺序由 FINAL_CLASSES 决定，训练用的 class id 以这个列表的下标为准。
"""
import os
import shutil

# ---------------- 配置区，按你的实际情况修改 ----------------

FINAL_CLASSES = ["纸箱", "文件袋", "塑料袋", "泡沫箱"]

# 公开数据集类别名 -> 项目类别名（None 表示丢弃）
PUBLIC_CLASS_MAP = {
    "box": "纸箱",
    "boxes": "纸箱",
    "cardboard": "纸箱",
    "package": "纸箱",
    "parcel": "纸箱",
    "good-parcel": "纸箱",
    "envelope": "文件袋",
    "label": None,
    "person": None,
}

# 校园自采数据集本身标注时就应直接用中文类名（对应 data/campus_raw/classes.txt），
# 这里同样做一层保险映射（一般恒等映射即可）
CAMPUS_CLASS_MAP = {c: c for c in FINAL_CLASSES}

KEEP_NEGATIVE_IMAGES = False  # 是否保留"该图所有框都被丢弃"的图（作为负样本）

PUBLIC_DIR = "data/public_yolo"      # prepare_public_dataset.py 产出的目录，需含 images/ labels/ classes.txt
CAMPUS_DIR = "data/campus_raw"       # 你自己采集标注的数据，需含 images/ labels/ classes.txt
OUT_DIR = "data/merged"              # 合并后输出目录

# -------------------------------------------------------------


def load_classes(dataset_dir):
    classes_path = os.path.join(dataset_dir, "classes.txt")
    if not os.path.exists(classes_path):
        raise SystemExit(f"未找到 {classes_path}，请确认该数据集已是 YOLO 格式并带 classes.txt")
    with open(classes_path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def remap_one_dataset(dataset_dir, class_map, final_classes, out_dir, prefix):
    src_classes = load_classes(dataset_dir)
    img_dir = os.path.join(dataset_dir, "images")
    lbl_dir = os.path.join(dataset_dir, "labels")
    out_img_dir = os.path.join(out_dir, "images")
    out_lbl_dir = os.path.join(out_dir, "labels")
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_lbl_dir, exist_ok=True)

    final_id = {c: i for i, c in enumerate(final_classes)}
    kept, skipped = 0, 0

    for lbl_file in os.listdir(lbl_dir):
        if not lbl_file.endswith(".txt"):
            continue
        stem = os.path.splitext(lbl_file)[0]

        new_lines = []
        with open(os.path.join(lbl_dir, lbl_file), encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                old_id = int(parts[0])
                if old_id >= len(src_classes):
                    continue
                old_name = src_classes[old_id]
                mapped = class_map.get(old_name.lower(), class_map.get(old_name))
                if mapped is None or mapped not in final_id:
                    continue
                new_id = final_id[mapped]
                new_lines.append(f"{new_id} {' '.join(parts[1:])}")

        if not new_lines and not KEEP_NEGATIVE_IMAGES:
            skipped += 1
            continue

        img_path = None
        for ext in (".jpg", ".jpeg", ".png"):
            candidate = os.path.join(img_dir, stem + ext)
            if os.path.exists(candidate):
                img_path = candidate
                break
        if img_path is None:
            skipped += 1
            continue

        new_stem = f"{prefix}_{stem}"
        shutil.copy(img_path, os.path.join(out_img_dir, new_stem + os.path.splitext(img_path)[1]))
        with open(os.path.join(out_lbl_dir, new_stem + ".txt"), "w", encoding="utf-8") as fo:
            fo.write("\n".join(new_lines))
        kept += 1

    print(f"[{prefix}] 保留 {kept} 张，跳过 {skipped} 张")


def main():
    if os.path.exists(OUT_DIR):
        shutil.rmtree(OUT_DIR)

    if os.path.exists(PUBLIC_DIR):
        remap_one_dataset(PUBLIC_DIR, PUBLIC_CLASS_MAP, FINAL_CLASSES, OUT_DIR, prefix="pub")
    else:
        print(f"[跳过] 未找到公开数据集目录: {PUBLIC_DIR}")

    if os.path.exists(CAMPUS_DIR):
        remap_one_dataset(CAMPUS_DIR, CAMPUS_CLASS_MAP, FINAL_CLASSES, OUT_DIR, prefix="campus")
    else:
        print(f"[跳过] 未找到校园自采数据目录: {CAMPUS_DIR}")

    with open(os.path.join(OUT_DIR, "classes.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(FINAL_CLASSES))

    print(f"[OK] 合并完成 -> {OUT_DIR}（下一步跑 clean_dataset.py）")


if __name__ == "__main__":
    main()