"""UAV-Crack Benchmark 本地评估脚本。

计算比赛口径的 6 个指标:
    mIoU, Crack F1, Crack IoU, Crack Precision, Crack Recall, aAcc

预测与真值都必须是二值 mask (像素值 0=背景, 1=裂缝)。脚本会兼容
0/255 或 0/1 两种保存方式, 并统一二值化成 0/1 后再统计。

用法:
    # 用训练集切出来的本地验证集评估
    python tools/eval_metrics.py \
        --pred-dir work_dirs/mask2former_swin-t/val_pred \
        --gt-dir   UAV-Crack-dataset/gtFine/train/UAV-CrackX4 \
        --pred-ext .png --gt-ext .png

    # 只看文件名重合的部分, 并导出逐图指标
    python tools/eval_metrics.py --pred-dir preds --gt-dir gt --json metrics.json
"""
import argparse
import json
import os
import sys

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None


def load_binary_mask(path):
    """读取任意格式的 mask 并二值化成 uint8 {0, 1}。

    返回 (mask, raw_max, raw_uniq_cnt):
        raw_max / raw_uniq_cnt 用于判断原文件是否已经是 0/1 规范格式。
    """
    with Image.open(path) as im:
        arr = np.array(im.convert("L"))
    raw_max = int(arr.max())
    raw_uniq_cnt = int(np.unique(arr).size)
    if raw_max <= 1:
        mask = (arr > 0).astype(np.uint8)
    else:
        # 0/255 之类的编码, 按 128 阈值二值化
        mask = (arr >= 128).astype(np.uint8)
    return mask, raw_max, raw_uniq_cnt


def confusion(gt, pred):
    """返回 2x2 混淆矩阵, cm[i, j] = gt==i 且 pred==j 的像素数。"""
    k = gt.astype(np.int64).ravel() * 2 + pred.astype(np.int64).ravel()
    return np.bincount(k, minlength=4).reshape(2, 2).astype(np.int64)


def metrics_from_cm(cm):
    """由混淆矩阵推导全部指标。类别 0=背景, 1=裂缝。"""
    tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
    eps = 1e-12

    iou_bg = tn / (tn + fn + fp + eps)
    iou_crack = tp / (tp + fn + fp + eps)
    prec = tp / (tp + fp + eps)
    rec = tp / (tp + fn + eps)
    f1 = 2 * prec * rec / (prec + rec + eps)
    total = cm.sum()
    acc = (tp + tn) / (total + eps)

    return {
        "mIoU": float((iou_bg + iou_crack) / 2),
        "Crack F1": float(f1),
        "Crack IoU": float(iou_crack),
        "Crack Precision": float(prec),
        "Crack Recall": float(rec),
        "aAcc": float(acc),
        "IoU(bg)": float(iou_bg),
        # 逐图取平均时才有意义的宏平均版本, 留作参考
        "TP": int(tp),
        "FP": int(fp),
        "FN": int(fn),
        "TN": int(tn),
        "gt_crack_pixels": int(tp + fn),
        "pred_crack_pixels": int(tp + fp),
    }


def list_stems(d, ext):
    if not os.path.isdir(d):
        raise SystemExit(f"[FATAL] 目录不存在: {d}")
    exts = {ext.lower()} if ext else None
    out = []
    for f in sorted(os.listdir(d)):
        stem, e = os.path.splitext(f)
        if exts is None or e.lower() in exts:
            out.append((stem, os.path.join(d, f)))
    return out


def fmt_row(name, glob_v, perimg_v):
    g = f"{glob_v:.4f}" if glob_v is not None else "-"
    p = f"{perimg_v:.4f}" if perimg_v is not None else "-"
    return f"  {name:<18}{g:>12}{p:>16}"


def main():
    ap = argparse.ArgumentParser(description="UAV-Crack 二值分割指标评估")
    ap.add_argument("--pred-dir", required=True, help="预测 mask 目录")
    ap.add_argument(
        "--gt-dir",
        default=None,
        help="真值 mask 目录 (例如 gtFine/train/UAV-CrackX4); 给了 --split-file 时可省略",
    )
    ap.add_argument("--pred-ext", default=".png", help="预测文件扩展名, 传空串表示不过滤")
    ap.add_argument("--gt-ext", default=".png", help="真值文件扩展名, 传空串表示不过滤")
    ap.add_argument(
        "--split-file",
        default=None,
        help="由 tools/split_train_val.py 生成的 val.txt; 给出后忽略 --gt-dir, "
        "按文件里的真值列表评估 (支持 X4/X8/X16 多尺度混合验证集)",
    )
    ap.add_argument("--data-root", default="UAV-Crack-dataset", help="split-file 的相对路径基准")
    ap.add_argument("--json", default=None, help="把结果写到该 json 文件")
    ap.add_argument(
        "--check-size",
        action="store_true",
        help="检查逐预测/真值尺寸是否一致 (略慢, 建议提交前打开)",
    )
    args = ap.parse_args()

    if args.split_file:
        gt_items = []
        with open(args.split_file, encoding="utf-8") as fh:
            for line in fh:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                gt_path = os.path.join(args.data_root, parts[1])
                gt_items.append((os.path.splitext(os.path.basename(parts[1]))[0], gt_path))
        args.gt_dir = f"{args.split_file} ({len(gt_items)} 条记录)"
    else:
        if not args.gt_dir:
            ap.error("需要 --gt-dir 或 --split-file 之一")
        gt_items = list_stems(args.gt_dir, args.gt_ext)

    pred_items = list_stems(args.pred_dir, args.pred_ext)
    if not gt_items:
        raise SystemExit(f"[FATAL] 真值目录为空: {args.gt_dir}")
    if not pred_items:
        raise SystemExit(f"[FATAL] 预测目录为空: {args.pred_dir}")

    pred_map = {s: p for s, p in pred_items}
    gt_map = {s: p for s, p in gt_items}
    gt_stems = [s for s, _ in gt_items]

    missing = [s for s in gt_stems if s not in pred_map]
    extra = sorted(set(pred_map) - set(gt_stems))
    matched = [s for s in gt_stems if s in pred_map]

    print("=" * 66)
    print("UAV-Crack 评估")
    print("=" * 66)
    print(f"  真值目录 : {args.gt_dir}  ({len(gt_items)} 个文件)")
    print(f"  预测目录 : {args.pred_dir}  ({len(pred_items)} 个文件)")
    print(f"  匹配成功 : {len(matched)}")
    if missing:
        print(f"  [WARN] 缺少预测 {len(missing)} 个, 例: {missing[:5]}")
    if extra:
        print(f"  [WARN] 多余预测 {len(extra)} 个, 例: {extra[:5]}")
        print("         (比赛要求 result 文件夹里只能有 300 张, 多余文件会导致评测异常)")
    if not matched:
        raise SystemExit("[FATAL] 没有任何文件名匹配成功, 请检查命名规则")

    cm_global = np.zeros((2, 2), dtype=np.int64)
    per_image = {}
    nonstandard_pred = []
    size_mismatch = []

    for stem in matched:
        gt_path = gt_map[stem]
        pred_path = pred_map[stem]

        gt, _, _ = load_binary_mask(gt_path)
        pred, raw_max, raw_uniq = load_binary_mask(pred_path)

        if raw_max > 1 or raw_uniq > 2:
            nonstandard_pred.append((stem, raw_max, raw_uniq))

        if gt.shape != pred.shape:
            size_mismatch.append((stem, gt.shape, pred.shape))
            continue

        cm = confusion(gt, pred)
        cm_global += cm
        per_image[stem] = metrics_from_cm(cm)

    if size_mismatch:
        print(f"\n  [ERROR] {len(size_mismatch)} 张预测与真值尺寸不一致, 已跳过:")
        for stem, gs, ps in size_mismatch[:10]:
            print(f"          {stem}: gt{gs[::-1]} vs pred{ps[::-1]}")

    if nonstandard_pred:
        print(
            f"\n  [WARN] {len(nonstandard_pred)} 张预测不是规范的 0/1 像素值 "
            f"(已按阈值二值化): "
        )
        for stem, mx, uc in nonstandard_pred[:5]:
            print(f"          {stem}: max={mx}, 唯一值个数={uc}")
        print("         比赛要求像素值为 0/1, 交付前请用 tools/make_submission.py 规范化")

    glob = metrics_from_cm(cm_global)

    # 逐图指标的算术平均 (aAcc 在部分 benchmark 里指这个口径, 一并给出)
    keys = ["mIoU", "Crack F1", "Crack IoU", "Crack Precision", "Crack Recall", "aAcc"]
    perimg = {}
    for k in keys:
        vals = [v[k] for v in per_image.values()]
        perimg[k] = float(np.mean(vals)) if vals else float("nan")

    print()
    print("-" * 66)
    print(f"  {'指标':<16}{'全局(像素级)':>14}{'逐图平均':>16}")
    print("-" * 66)
    for k in keys:
        print(fmt_row(k, glob[k], perimg[k]))
    print("-" * 66)
    print(fmt_row("IoU(背景)", glob["IoU(bg)"], None))
    print()
    print(f"  参与评估图片数   : {len(per_image)}")
    print(f"  裂缝像素总数(gt) : {glob['gt_crack_pixels']:,}")
    print(f"  裂缝像素总数(pred): {glob['pred_crack_pixels']:,}")
    print(f"  TP={glob['TP']:,}  FP={glob['FP']:,}  FN={glob['FN']:,}  TN={glob['TN']:,}")

    if args.check_size and not size_mismatch:
        print("  尺寸检查         : 全部通过")

    print("=" * 66)

    if glob["pred_crack_pixels"] == 0:
        print("\n  提示: 预测里没有任何裂缝像素, Crack F1/IoU/Recall 全为 0。")

    if args.json:
        out = {
            "pred_dir": args.pred_dir,
            "gt_dir": args.gt_dir,
            "num_images": len(per_image),
            "num_missing": len(missing),
            "num_extra": len(extra),
            "global": glob,
            "per_image_mean": perimg,
            "per_image": per_image,
        }
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"  结果已写入: {args.json}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
