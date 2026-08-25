# 基于YOLOv8的快递包裹识别 — 工程代码包

本包实现「公开数据集 + 自标注补充数据 → 合并 → 单阶段训练 YOLOv8 → 推理服务」的完整流程。

## 目录结构

```
campus-express-yolov8/
├── requirements.txt
├── data/
│   ├── scripts/
│   │   ├── prepare_public_dataset.py       # 整理网页手动下载的公开包裹数据集
│   │   ├── merge_datasets.py               # 合并 公开数据集 + own_labeled 补充数据，统一类别映射样
│   │   ├── clean_dataset.py                # 感知哈希(pHash)去重 + 相似图片分组、无效图像/空标注剔除
│   │   ├── split_dataset.py                # 分层 + 按相似分组划分 train/val/test，含巨型分组拆分保护
│   │   └── augment.py                      # 离线数据增强（重点强化小目标）
│   ├── public_raw/                         # 网页下载解压的公开数据集原始文件
│   ├── own_labeled/                        # 手动标注的另外两类数据（YOLO格式）
│   ├── merged/                             # merge_datasets.py 产出的合并数据集
│   ├── dataset/                            # 最终划分好的 YOLO 数据集
│   └── public_yolo/                        # prepare_public_dataset.py 输出的指定目录
├── training/
│   ├── configs/
│   │   └── train_config.yaml               # 训练配置
│   ├── train.py                            # 训练入口
│   └── eval.py                             # 在测试集上算 P/R/mAP/推理速度，导出报告
└── system/
    ├── backend/
    │   ├── main.py                         # FastAPI 服务
    │   └── inference.py                    # 模型加载与推理封装
    └── frontend/
        └── index.html                      # 极简上传+可视化页面（纯前端，调用后端API）
```

## 训练策略

不再区分"公开数据预训练"和"校园数据微调"两个阶段——补充数据（塑料袋、泡沫箱等细分类别）本身也来自网络公开渠道人工标注整理，与主体公开数据集同源，人为拆成两阶段没有实际意义。现在是**单阶段训练**：把所有数据源合并成一个数据集，一次性训练。

数据来源：
1. **公开数据集**：从 Roboflow 等平台下载的包裹检测数据集，覆盖纸箱等主要类别
2. **own_labeled 补充数据**：针对公开数据集里样本量不足的细分类别（塑料袋、泡沫箱等），额外收集图片并用 labelme 人工标注并进行转换

## 使用步骤（本地）

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 整理网页下载解压的公开数据集（自动读取 data.yaml 和 train/valid/test）
python data/scripts/prepare_public_dataset.py --src data/public_raw --out data/public_yolo

# 3. 合并公开数据集 + own_labeled 补充数据，统一类别映射（在脚本顶部 CLASS_MAP 里配置）
python data/scripts/merge_datasets.py

# 4. 清洗 + 划分 + 增强
python data/scripts/clean_dataset.py
python data/scripts/split_dataset.py
python data/scripts/augment.py

# 5. 训练（单阶段）
python training/train.py --config training/configs/train_config.yaml

# 6. 评估
python training/eval.py --weights training/runs/stage1/weights/best.pt --data data/dataset/data.yaml

# 7. 启动推理服务
cd system/backend && uvicorn main:app --reload --port 8000
# 打开 system/frontend/index.html 上传图片测试
```
