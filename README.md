# 基于YOLOv8的校园快递包裹分类识别 — 工程代码包

本包实现「公开数据集 → 合并 → 迁移学习训练 YOLOv8 → 推理服务」的完整流程。

## 目录结构

```
campus-express-yolov8/
├── requirements.txt
├── data/
│   ├── scripts/
│   │   ├── prepare_public_dataset.py    # 整理网页手动下载的公开包裹数据集（Roboflow zip）
│   │   ├── convert_to_yolo.py           # 【备用】若公开集是 VOC/COCO 格式，转换为 YOLO txt
│   │   ├── merge_datasets.py            # 合并 公开数据集，统一类别映射
│   │   ├── clean_dataset.py             # 感知哈希(pHash)去重 + 相似图片分组、无效图像/空标注剔除
│   │   ├── split_dataset.py             # 按相似分组划分 train/val/test（避免同源图片跨集泄漏）
│   │   └── augment.py                   # 离线数据增强（重点强化小目标）
│   ├── dataset/                         # 合并划分后的最终 YOLO 数据集（脚本自动生成）
│   └── data.yaml                        # 数据集配置（脚本自动生成/也可手动改）
├── training/
│   ├── configs/
│   │   ├── stage1_public_pretrain.yaml  # 阶段1：在公开包裹数据集上先训一版
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

## 迁移学习策略

1. **阶段1：公开数据集预训练**
   推荐使用 Roboflow Universe 上的公开包裹检测数据集（例如 `public.roboflow.com/object-detection/packages-dataset`，箱子/信封/大件包裹标注），在 YOLOv8 官方 COCO 预训练权重基础上先训练一版通用"包裹检测"模型，让模型先学会"什么是包裹"这个大类特征。


## 使用步骤（本地）

# 0. 安装依赖
pip install -r requirements.txt

# 1. 整理刚才手动下载解压的公开数据集（自动读取 data.yaml 和 train/valid/test）
python data/scripts/prepare_public_dataset.py --src data/public_raw --out data/public_yolo

# 2. 【备用，一般用不到】如果你找到的数据集不是YOLOv8格式而是VOC/COCO，用这个转换
python data/scripts/convert_to_yolo.py --src data/public_raw --format voc --out data/public_yolo

# 3. 把你自己采集并标注好的数据放到 data/campus_raw/{images,labels}，
#    然后合并公开数据集 + 校园数据集()，并统一类别映射（在脚本顶部 CLASS_MAP 里配置）
python data/scripts/merge_datasets.py

# 3.5【可选】如果类别严重不平衡（比如纸箱远多于塑料袋/泡沫箱），下采样多数类
python data/scripts/downsample_majority_class.py --src data/merged --majority-class 纸箱 --target 2500

# 4. 清洗 + 划分 + 增强
python data/scripts/clean_dataset.py
python data/scripts/split_dataset.py
python data/scripts/augment.py

# 5. 两阶段训练
python training/train.py --stage 1 --config training/configs/stage1_public_pretrain.yaml


# 6. 评估
python training/eval.py --weights training/runs/stage2/weights/best.pt --data data/dataset/data.yaml

# 7. 启动推理服务
cd system/backend && uvicorn main:app --reload --port 8000
# 打开 system/frontend/index.html 上传图片测试
```

## 说明

- 公开数据集类别名和你的校园数据集类别名往往不一致（比如公开集只有 `box`/`plastic bag`，你需要 `纸箱/塑料袋/泡沫箱`），`merge_datasets.py` 里的 `CLASS_MAP` 就是做这层映射，映射不上的类别会被丢弃，用前务必按你实际情况改这张表（`prepare_public_dataset.py` 跑完会直接把检测到的类别名打印出来，照着改就行）。
- **类别不平衡处理**：如果纸箱类样本量远超塑料袋/泡沫箱（比如 1万+ vs 几百），建议在 `merge_datasets.py` 和 `clean_dataset.py` 之间加一步下采样：
  ```bash
  python data/scripts/downsample_majority_class.py --src data/merged --majority-class 纸箱 --target 2500
  ```
  只会随机删除"纯纸箱、不含其他类别"的图片，混合图片和少数类图片全部保留，不会误删稀缺样本。经验上不平衡比例控制在 10:1 以内，模型才能正常学到少数类特征。
- **数据同源性与防泄漏**：如果补充数据集和公开数据集本身来自相近的网络渠道（而非独立的线下采集），单纯用 MD5 精确去重无法识别"同一张图的不同压缩/裁剪/转存版本"。`clean_dataset.py` 改用感知哈希（pHash）按内容相似度去重，并把"相似但不完全重复"的图片记录为同一分组（`data/merged/photo_groups.json`）；`split_dataset.py` 会按这份分组信息划分数据集，保证同一分组不会被拆到不同的 train/val/test 里，避免测试集"偷看"训练集内容导致评估结果虚高。
- 只要最终整理成 `images/ + labels/`（YOLO txt 格式）+ `classes.txt`，就能接入后续 `merge_datasets.py` 流程，不局限于 Roboflow，任何来源的公开数据集都适用。
