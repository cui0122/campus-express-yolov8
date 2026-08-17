"""
按 7:2:1 划分 train/val/test，并生成最终 data/dataset/ 目录 + data.yaml
"""
import json
import os
import random
import shutil
from tqdm import tqdm

SRC_DIR = "data/merged"
OUT_DIR = "data/dataset"
SPLIT = {"train": 0.7, "val": 0.2, "test": 0.1}
SEED = 11


def group_based_split(stems, groups, split_ratios, seed):
    random.seed(seed)
    stem_to_group = {s: groups.get(s, s) for s in stems}
    group_members = {}
    for s, g in stem_to_group.items():
        group_members.setdefault(g, []).append(s)

    group_ids = list(group_members.keys())
    random.shuffle(group_ids)

    total = len(stems)
    targets = {k: v * total for k, v in split_ratios.items()}
    counts = {k: 0 for k in split_ratios}
    result = {k: [] for k in split_ratios}

    for g in group_ids:
        members = group_members[g]
        deficits = {k: targets[k] - counts[k] for k in split_ratios}
        target_split = max(deficits, key=deficits.get)
        result[target_split].extend(members)
        counts[target_split] += len(members)

    return result["train"], result["val"], result["test"]


def random_split(stems, split_ratios, seed):
    from sklearn.model_selection import train_test_split

    train_stems, temp_stems = train_test_split(
        stems, train_size=split_ratios["train"], random_state=seed
    )
    val_ratio_in_temp = split_ratios["val"] / (split_ratios["val"] + split_ratios["test"])
    val_stems, test_stems = train_test_split(
        temp_stems, train_size=val_ratio_in_temp, random_state=seed
    )
    return train_stems, val_stems, test_stems


def main():
    img_dir = os.path.join(SRC_DIR, "images")
    lbl_dir = os.path.join(SRC_DIR, "labels")
    classes_path = os.path.join(SRC_DIR, "classes.txt")
    groups_path = os.path.join(SRC_DIR, "photo_groups.json")

    stems = [os.path.splitext(f)[0] for f in os.listdir(img_dir)]

    if os.path.exists(groups_path):
        with open(groups_path, "r", encoding="utf-8") as f:
            groups = json.load(f)
        n_groups = len(set(groups.get(s, s) for s in stems))
        print(f"✅ 检测到相似分组信息 photo_groups.json，共 {n_groups} 组，将按组划分")
        train_stems, val_stems, test_stems = group_based_split(stems, groups, SPLIT, SEED)
    else:
        print("⚠️ 未找到 photo_groups.json，退化为纯随机划分")
        train_stems, val_stems, test_stems = random_split(stems, SPLIT, SEED)

    split_map = {"train": train_stems, "val": val_stems, "test": test_stems}

    for split_name, split_stems in split_map.items():
        out_img = os.path.join(OUT_DIR, "images", split_name)
        out_lbl = os.path.join(OUT_DIR, "labels", split_name)
        os.makedirs(out_img, exist_ok=True)
        os.makedirs(out_lbl, exist_ok=True)
        
        for stem in tqdm(split_stems, desc=f"写入 {split_name} 集"):
            for ext in (".jpg", ".jpeg", ".png"):
                src_img = os.path.join(img_dir, stem + ext)
                if os.path.exists(src_img):
                    shutil.copy(src_img, os.path.join(out_img, stem + ext))
                    break
            src_lbl = os.path.join(lbl_dir, stem + ".txt")
            if os.path.exists(src_lbl):
                shutil.copy(src_lbl, os.path.join(out_lbl, stem + ".txt"))
        
        actual_ratio = len(split_stems) / len(stems) if stems else 0
        print(f"{split_name}: {len(split_stems)} 张（占比 {actual_ratio:.1%}，目标 {SPLIT[split_name]:.0%}）")

    with open(classes_path, 'r', encoding='utf-8') as f:
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