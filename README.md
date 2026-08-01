<<<<<<< HEAD
# campus-express-yolov8
=======
# 基于YOLOv8的校园快递包裹分类识别 — 工程代码包

本包实现「公开数据集 + 校园自采数据集 → 合并 → 迁移学习训练 YOLOv8 → 推理服务」的完整流程。

## 目录结构

```
campus-express-yolov8/
├── requirements.txt
├── data/
│   ├── scripts/
│   │   ├── prepare_public_dataset.py    # 整理网页手动下载的公开包裹数据集（Roboflow zip）
│   │   ├── convert_to_yolo.py           # 【备用】若公开集是 VOC/COCO 格式，转换为 YOLO txt
│   │   ├── merge_datasets.py            # 合并 公开数据集 + 校园自采数据集，统一类别映射
│   │   ├── clean_dataset.py             # 去重、无效图像/空标注剔除
│   │   ├── split_dataset.py             # 划分 train/val/test
│   │   └── augment.py                   # 离线数据增强（重点强化小目标）
│   ├── campus_raw/                      # 你自己采集标注的校园数据放这里
│   ├── dataset/                         # 合并划分后的最终 YOLO 数据集（脚本自动生成）
│   └── data.yaml                        # 数据集配置（脚本自动生成/也可手动改）
├── training/
│   ├── configs/
│   │   ├── stage1_public_pretrain.yaml  # 阶段1：在公开包裹数据集上先训一版
│   │   └── stage2_campus_finetune.yaml  # 阶段2：在校园自采数据集上迁移微调
│   ├── train.py                         # 训练入口，支持两阶段迁移学习
│   └── eval.py                          # 在测试集上算 P/R/mAP/推理速度，导出报告
└── system/
    ├── backend/
    │   ├── main.py                      # FastAPI 服务
    │   └── inference.py                 # 模型加载与推理封装
    └── frontend/
        └── index.html                   # 极简上传+可视化页面（纯前端，调用后端API）
docs/
├── 系统架构图.drawio                     # 对应大纲 5.2 总体设计，draw.io 可编辑源文件
├── 数据集说明.md                         # 对应大纲 6.1，数据来源/类别/划分/标注规范模板
└── 实验记录表.xlsx                       # 对应大纲 6.4，训练/对比/真实场景测试记录模板
```

## 迁移学习策略（两阶段）

1. **阶段1：公开数据集预训练**
   推荐使用 Roboflow Universe 上的公开包裹检测数据集（例如 `public.roboflow.com/object-detection/packages-dataset`，箱子/信封/大件包裹标注），在 YOLOv8 官方 COCO 预训练权重基础上先训练一版通用"包裹检测"模型，让模型先学会"什么是包裹"这个大类特征。

2. **阶段2：校园自采数据集微调**
   用阶段1得到的权重作为初始权重，在你自己采集、标注的校园快递包裹数据集（纸箱/文件袋/塑料袋/泡沫箱等细分类别）上继续训练（更小学习率、更少 epoch），让模型学会你关心的**细分类别**。

这样比直接用 COCO 权重从零微调收敛更快、小样本下泛化更好，也符合开题报告里"利用迁移学习策略在自建数据集上训练"的表述。

## 获取公开数据集（网页操作，不需要写代码/API Key）

1. 打开 Roboflow Universe（https://universe.roboflow.com），搜 "package detection" / "parcel detection" 之类关键词，挑一个图片量大、类别接近的项目
2. 点 **"Download Dataset"** → 格式选 **YOLOv8** → 下载方式选 **"download zip to computer"**
3. 解压后，把整个文件夹放到项目的 `data/public_raw/` 目录下（保留它自带的 `data.yaml` 和 `train/valid/test` 结构，不用自己整理）

## 使用步骤

```bash
# 0. 安装依赖
pip install -r requirements.txt

# 1. 整理刚才手动下载解压的公开数据集（自动读取 data.yaml 和 train/valid/test）
python data/scripts/prepare_public_dataset.py --src data/public_raw --out data/public_yolo

# 2. 【备用，一般用不到】如果你找到的数据集不是YOLOv8格式而是VOC/COCO，用这个转换
python data/scripts/convert_to_yolo.py --src data/public_raw --format voc --out data/public_yolo

# 3. 把你自己采集并标注好的数据放到 data/campus_raw/{images,labels}，
#    然后合并公开数据集 + 校园数据集，并统一类别映射（在脚本顶部 CLASS_MAP 里配置）
python data/scripts/merge_datasets.py

# 4. 清洗 + 划分 + 增强
python data/scripts/clean_dataset.py
python data/scripts/split_dataset.py
python data/scripts/augment.py

# 5. 两阶段训练
python training/train.py --stage 1 --config training/configs/stage1_public_pretrain.yaml
python training/train.py --stage 2 --config training/configs/stage2_campus_finetune.yaml \
       --weights training/runs/stage1/weights/best.pt

# 6. 评估
python training/eval.py --weights training/runs/stage2/weights/best.pt --data data/dataset/data.yaml

# 7. 启动推理服务
cd system/backend && uvicorn main:app --reload --port 8000
# 打开 system/frontend/index.html 上传图片测试
```

## 说明

- 公开数据集类别名和你的校园数据集类别名往往不一致（比如公开集只有 `box`/`envelope`，你需要 `纸箱/文件袋/塑料袋/泡沫箱`），`merge_datasets.py` 里的 `CLASS_MAP` 就是做这层映射，映射不上的类别会被丢弃或映射到"其他"，用前务必按你实际情况改这张表（`prepare_public_dataset.py` 跑完会直接把检测到的类别名打印出来，照着改就行）。
- 只要最终整理成 `images/ + labels/`（YOLO txt 格式）+ `classes.txt`，就能接入后续 `merge_datasets.py` 流程，不局限于 Roboflow，任何来源的公开数据集都适用。
>>>>>>> 9fe9be6 (第一次提交)
