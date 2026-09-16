"""按无人机源图 ID 切分 UAV-CrackX 训练集, 生成 mmseg 可用的 train/val 列表。

为什么要按源图切分:
    官方 val 的 142 张原图里有 137 张和训练集重合, 说明它是"同一批航拍原图的
    另一些 patch"。因此本地切分也必须以源图 (xxx_Z.JPG) 为单位, 同一源图的
    所有 patch 只能落在同一侧, 否则本地分数会明显虚高。

用法:
    python tools/split_train_val.py --val-ratio 0.2 --seed 42
输出:
    splits/train.txt       mmseg 格式, 单列 "UAV-CrackX4/937956_..._11_z"
    splits/val.txt         同上
    splits/val_pairs.txt   两列完整相对路径, 给 tools/eval_metrics.py 用
"""
import argparse
import os
import random
import re
import sys
from collections import defaultdict

SCALES = ["UAV-CrackX4", "UAV-CrackX8", "UAV-CrackX16"]
SRC_RE = re.compile(r"^(.*_Z)\.JPG_", re.IGNORECASE)


def source_id(filename):
    """937956_DJI_20231015154707_0002_Z.JPG_12_z.jpg -> 937956_DJI_..._0002_Z"""
    m = SRC_RE.match(filename)
    return m.group(1) if m else os.path.splitext(filename)[0]


def main():
    ap = argparse.ArgumentParser(description="按源图切分 UAV-CrackX 训练集")
    ap.add_argument("--data-root", default="UAV-Crack-dataset")
    ap.add_argument("--val-ratio", type=float, default=0.2, help="验证集源图占比")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default="splits")
    args = ap.parse_args()

    img_root = os.path.join(args.data_root, "leftImg8bit", "train")
    gt_root = os.path.join(args.data_root, "gtFine", "train")
    if not os.path.isdir(img_root):
        raise SystemExit(f"[FATAL] 找不到 {img_root}")

    pairs = defaultdict(list)  # source_id -> [(img_rel, gt_rel)]
    for scale in SCALES:
        img_dir = os.path.join(img_root, scale)
        gt_dir = os.path.join(gt_root, scale)
        if not os.path.isdir(img_dir):
            print(f"  [skip] 缺少 {img_dir}")
            continue
        for f in sorted(os.listdir(img_dir)):
            stem, ext = os.path.splitext(f)
            if ext.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            gt_name = stem + ".png"
            if not os.path.exists(os.path.join(gt_dir, gt_name)):
                print(f"  [WARN] 缺少真值: {scale}/{gt_name}")
                continue
            # 统一用正斜杠写进列表文件, Windows / Linux 都能直接读
            pairs[source_id(f)].append(
                (f"leftImg8bit/train/{scale}/{f}", f"gtFine/train/{scale}/{gt_name}")
            )

    if not pairs:
        raise SystemExit("[FATAL] 没有找到任何 图像/真值 配对")

    src_ids = sorted(pairs.keys())

    # 按"该源图 patch 最多的尺度"分层, 避免某个尺度在验证集里几乎消失
    groups = defaultdict(list)
    for s in src_ids:
        hist = defaultdict(int)
        for img_rel, _ in pairs[s]:
            hist[img_rel.split("/")[2]] += 1
        groups[max(hist, key=hist.get)].append(s)

    rng = random.Random(args.seed)
    val_src = set()
    for scale in SCALES:
        ids = groups.get(scale, [])
        if not ids:
            continue
        rng.shuffle(ids)
        n_val = max(1, int(round(len(ids) * args.val_ratio)))
        val_src.update(ids[:n_val])

    train_lines = [p for s in src_ids if s not in val_src for p in pairs[s]]
    val_lines = [p for s in src_ids if s in val_src for p in pairs[s]]

    # 同源泄漏自检
    leak = {source_id(os.path.basename(i)) for i, _ in train_lines} & {
        source_id(os.path.basename(i)) for i, _ in val_lines
    }
    if leak:
        print(f"  [ERROR] 存在同源泄漏 {len(leak)} 个源图, 切分逻辑有 bug")

    os.makedirs(args.out_dir, exist_ok=True)

    def write_mmseg(name, lines):
        """mmseg BaseSegDataset 只认单列的 "相对子路径/文件名（不含扩展名）"。

        它会自己拼成
            <data_root>/<img_path>/<这一列>.jpg
            <data_root>/<seg_map_path>/<这一列>.png
        所以这里写 UAV-CrackX4/937956_..._11_z 这种形式, 尺度子目录就被带上了。
        """
        path = os.path.join(args.out_dir, name)
        prefix = "leftImg8bit/train/"
        with open(path, "w", encoding="utf-8") as f:
            for img_rel, _ in lines:
                stem = os.path.splitext(img_rel)[0]      # 去掉 .jpg
                assert stem.startswith(prefix), stem
                f.write(f"{stem[len(prefix):]}\n")       # 只留 UAV-CrackX4/xxx

    write_mmseg("train.txt", train_lines)
    write_mmseg("val.txt", val_lines)

    # 额外交一份两列的完整路径, 给 tools/eval_metrics.py 用
    with open(os.path.join(args.out_dir, "val_pairs.txt"), "w", encoding="utf-8") as f:
        for img_rel, gt_rel in val_lines:
            f.write(f"{img_rel} {gt_rel}\n")

    def scale_hist(lines):
        h = defaultdict(int)
        for img_rel, _ in lines:
            h[img_rel.split("/")[2]] += 1
        return dict(sorted(h.items()))

    print("=" * 62)
    print(f"  源图总数   : {len(src_ids)}  (seed={args.seed})")
    print(f"  train 源图 : {len(src_ids) - len(val_src):>4}   图片 {len(train_lines):>5} 张  {scale_hist(train_lines)}")
    print(f"  val   源图 : {len(val_src):>4}   图片 {len(val_lines):>5} 张  {scale_hist(val_lines)}")
    print(f"  输出       : {args.out_dir}/train.txt , {args.out_dir}/val.txt")
    print("=" * 62)
    return 0


if __name__ == "__main__":
    sys.exit(main())
