"""
训练入口，支持两阶段迁移学习：
  阶段1：公开数据集预训练（从 COCO 权重开始）
  阶段2：校园数据集微调（从阶段1权重开始）

用法：
python ./training/train.py --stage 1 --config ./training/configs/stage1_public_pretrain.yaml
python ./training/train.py --stage 2 --config ./training/configs/stage2_campus_finetune.yaml \
                 --weights ./training/runs/stage1/weights/best.pt
"""
import argparse

import yaml
from ultralytics import YOLO


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=int, choices=[1, 2], required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--weights", default=None, help="覆盖配置文件里的初始权重路径（阶段2常用）")
    args = parser.parse_args()

    cfg = load_config(args.config)
    model_path = args.weights or cfg.pop("model")
    data_path = cfg.pop("data")

    print(f"[Stage {args.stage}] 使用初始权重: {model_path}")
    print(f"[Stage {args.stage}] 数据集配置: {data_path}")

    model = YOLO(model_path)
    results = model.train(data=data_path, **cfg)

    print(f"[OK] 训练完成，最佳权重: {results.save_dir}/weights/best.pt")


if __name__ == "__main__":
    main()
