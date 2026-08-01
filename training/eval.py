"""
在测试集上评估模型：Precision / Recall / mAP50 / mAP50-95 / 单张推理耗时
输出一份 CSV + 控制台汇总，方便直接贴进论文第六章实验结果表格。

用法：
python eval.py --weights training/runs/stage2/weights/best.pt --data data/dataset/data.yaml
"""
import argparse
import time

import pandas as pd
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--out-csv", default="training/eval_report.csv")
    args = parser.parse_args()

    model = YOLO(args.weights)

    metrics = model.val(data=args.data, imgsz=args.imgsz, split="test")

    import glob

    sample_imgs = glob.glob("data/dataset/images/test/*.jpg") + glob.glob(
        "data/dataset/images/test/*.png"
    )
    infer_time_ms = None
    if sample_imgs:
        model.predict(sample_imgs[0], imgsz=args.imgsz, verbose=False)
        t0 = time.time()
        n = min(20, len(sample_imgs))
        for img in sample_imgs[:n]:
            model.predict(img, imgsz=args.imgsz, verbose=False)
        infer_time_ms = (time.time() - t0) / n * 1000

    result_row = {
        "weights": args.weights,
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "mAP50": float(metrics.box.map50),
        "mAP50-95": float(metrics.box.map),
        "avg_inference_ms": infer_time_ms,
    }

    df = pd.DataFrame([result_row])
    df.to_csv(args.out_csv, index=False, encoding="utf-8-sig")

    print("\n===== 评估结果（可直接用于论文第六章表格）=====")
    for k, v in result_row.items():
        print(f"{k}: {v}")
    print(f"\n[OK] 报告已保存: {args.out_csv}")


if __name__ == "__main__":
    main()
