"""
按 7:2:1 划分 train/val/test，并生成最终 data/dataset/ 目录 + data.yaml
"""
import os
import random
import shutil

from sklearn.model_selection import train_test_split

SRC_DIR = "data/merged"
OUT_DIR = "data/dataset"
SPLIT = {"train": 0.7, "val": 0.2, "test": 0.1}
SEED = 42


def main():
    random.seed(SEED)
    img_dir = os.path.join(SRC_DIR, "images")
    lbl_dir = os.path.join(SRC_DIR, "labels")
    classes_path = os.path.join(SRC_DIR, "classes.txt")

    stems = [os.path.splitext(f)[0] for f in os.listdir(img_dir)]
    train_stems, temp_stems = train_test_split(
        stems, train_size=SPLIT["train"], random_state=SEED
    )
    val_ratio_in_temp = SPLIT["val"] / (SPLIT["val"] + SPLIT["test"])
    val_stems, test_stems = train_test_split(
        temp_stems, train_size=val_ratio_in_temp, random_state=SEED
    )

    split_map = {"train": train_stems, "val": val_stems, "test": test_stems}

    for split_name, split_stems in split_map.items():
        out_img = os.path.join(OUT_DIR, "images", split_name)
        out_lbl = os.path.join(OUT_DIR, "labels", split_name)
        os.makedirs(out_img, exist_ok=True)
        os.makedirs(out_lbl, exist_ok=True)
        for stem in split_stems:
            for ext in (".jpg", ".jpeg", ".png"):
                src_img = os.path.join(img_dir, stem + ext)
                if os.path.exists(src_img):
                    shutil.copy(src_img, os.path.join(out_img, stem + ext))
                    break
            src_lbl = os.path.join(lbl_dir, stem + ".txt")
            if os.path.exists(src_lbl):
                shutil.copy(src_lbl, os.path.join(out_lbl, stem + ".txt"))
        print(f"{split_name}: {len(split_stems)} 张")

    with open(classes_path, encoding="utf-8") as f:
        classes = [line.strip() for line in f if line.strip()]

    data_yaml = f"""# 自动生成，可手动调整
path: {os.path.abspath(OUT_DIR)}
train: images/train
val: images/val
test: images/test

nc: {len(classes)}
names: {classes}
"""
    with open(os.path.join(OUT_DIR, "data.yaml"), "w", encoding="utf-8") as f:
        f.write(data_yaml)

    print(f"[OK] 划分完成，data.yaml 已生成: {os.path.join(OUT_DIR, 'data.yaml')}")


if __name__ == "__main__":
    main()