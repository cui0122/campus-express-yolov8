"""
数据清洗：
1. 剔除无法打开/损坏的图像
2. 剔除空标注（没有任何框）的样本（除非明确想留作负样本）
3. 基于感知哈希（pHash）去重
4. 对"相似但不完全重复"的图像做分组
"""
import json
import os

import imagehash
from PIL import Image
from tqdm import tqdm

SRC_DIR = "data/merged"
IMG_DIR = os.path.join(SRC_DIR, "images")
LBL_DIR = os.path.join(SRC_DIR, "labels")

DROP_EMPTY_LABELS = True
DEDUP_THRESHOLD = 5
GROUP_THRESHOLD = 15


class UnionFind:
    def __init__(self, items):
        self.parent = {x: x for x in items}

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def compute_hashes(img_files):
    hashes = {}
    broken = []
    for img_file in tqdm(img_files, desc="1/4 计算感知哈希"):
        img_path = os.path.join(IMG_DIR, img_file)
        try:
            with Image.open(img_path) as im:
                im.verify()
            with Image.open(img_path) as im:
                hashes[img_file] = imagehash.phash(im)
        except Exception:
            broken.append(img_file)
    return hashes, broken


def main():
    img_files = [f for f in os.listdir(IMG_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    print(f"开始清洗，原始图像数: {len(img_files)}")

    # 1) 计算感知哈希，顺便识别损坏图像
    hashes, broken = compute_hashes(img_files)
    removed_broken = 0
    for img_file in broken:
        stem = os.path.splitext(img_file)[0]
        img_path = os.path.join(IMG_DIR, img_file)
        lbl_path = os.path.join(LBL_DIR, stem + ".txt")
        if os.path.exists(img_path):
            os.remove(img_path)
        if os.path.exists(lbl_path):
            os.remove(lbl_path)
        removed_broken += 1
    print(f"剔除损坏图像: {removed_broken}")

    valid_files = list(hashes.keys())

    # 2) 两两比较汉明距离，做去重聚类 + 相似分组聚类
    n = len(valid_files)
    dedup_uf = UnionFind(valid_files)
    group_uf = UnionFind(valid_files)

    for i in tqdm(range(n), desc="2/4 相似度比较与分组"):
        for j in range(i + 1, n):
            fi, fj = valid_files[i], valid_files[j]
            dist = hashes[fi] - hashes[fj]
            if dist <= DEDUP_THRESHOLD:
                dedup_uf.union(fi, fj)
                group_uf.union(fi, fj)
            elif dist <= GROUP_THRESHOLD:
                group_uf.union(fi, fj)

    # 3) 去重：每个去重簇只保留一张
    dedup_clusters = {}
    for f in valid_files:
        root = dedup_uf.find(f)
        dedup_clusters.setdefault(root, []).append(f)

    removed_dup = 0
    kept_files = []
    for root, members in tqdm(dedup_clusters.items(), desc="3/4 执行感知哈希去重"):
        members.sort()
        keep = members[0]
        kept_files.append(keep)
        for f in members[1:]:
            stem = os.path.splitext(f)[0]
            img_path = os.path.join(IMG_DIR, f)
            lbl_path = os.path.join(LBL_DIR, stem + ".txt")
            if os.path.exists(img_path):
                os.remove(img_path)
            if os.path.exists(lbl_path):
                os.remove(lbl_path)
            removed_dup += 1
    print(f"剔除重复/近似重复图像(感知哈希去重): {removed_dup}")

    # 4) 剔除空标注
    removed_empty = 0
    final_kept = []
    for f in tqdm(kept_files, desc="4/4 检查并剔除空标注"):
        stem = os.path.splitext(f)[0]
        lbl_path = os.path.join(LBL_DIR, stem + ".txt")
        if DROP_EMPTY_LABELS and (not os.path.exists(lbl_path) or os.path.getsize(lbl_path) == 0):
            img_path = os.path.join(IMG_DIR, f)
            if os.path.exists(img_path):
                os.remove(img_path)
            if os.path.exists(lbl_path):
                os.remove(lbl_path)
            removed_empty += 1
            continue
        final_kept.append(f)
    print(f"剔除空标注: {removed_empty}")

    # 5) 保存相似分组信息
    photo_groups = {}
    for f in final_kept:
        stem = os.path.splitext(f)[0]
        group_root = group_uf.find(f)
        photo_groups[stem] = group_root

    n_groups = len(set(photo_groups.values()))
    with open(os.path.join(SRC_DIR, "photo_groups.json"), "w", encoding="utf-8") as fp:
        json.dump(photo_groups, fp, ensure_ascii=False, indent=2)

    remaining = len(final_kept)
    print(f"\n[OK] 清洗完成。剩余 {remaining} 张图像，分成 {n_groups} 个相似分组。")
    print(f"分组信息已保存到: {os.path.join(SRC_DIR, 'photo_groups.json')}")


if __name__ == "__main__":
    main()