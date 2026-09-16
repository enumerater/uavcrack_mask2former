"""UAV-Crack Benchmark 提交打包脚本。

做三件事:
  1. 逐张校验预测 mask 的 文件名 / 分辨率 / 像素值
  2. 规范化成 单通道 L 模式、像素值 {0, 1} 的 png, 输出到 result/
  3. 把 result/ 打成 result.zip (zip 内路径为 result/xxx.png)

用法:
    # 正常打包
    python tools/make_submission.py \
        --pred-dir work_dirs/mask2former_swin-t/test_pred \
        --val-img-dir UAV-Crack-dataset/leftImg8bit/val \
        --out-dir result --zip result.zip

    # 生成一份全 0 的占位提交, 先验证提交流程与线上评分接口是否正常
    python tools/make_submission.py --val-img-dir UAV-Crack-dataset/leftImg8bit/val --dummy

    # 只校验不打包
    python tools/make_submission.py --pred-dir preds --val-img-dir ... --dry-run
"""
import argparse
import os
import shutil
import sys
import zipfile

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None


def _aliases(name):
    """生成可能的命名变体: xxx.jpg / xxx.jpg.png / xxx.png 都尽量能对上。"""
    yield name
    low = name.lower()
    for e in (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"):
        if low.endswith(e):
            yield name[: -len(e)]
            break


def build_lookup(pred_dir, pred_ext):
    """建立 文件名 -> 路径 的索引, 兼容 'xxx.png' 与 'xxx.jpg.png' 两种保存习惯。"""
    lookup = {}
    if pred_dir is None:
        return lookup
    for f in sorted(os.listdir(pred_dir)):
        if not os.path.isfile(os.path.join(pred_dir, f)):
            continue
        if pred_ext and not f.lower().endswith(pred_ext.lower()):
            continue
        full = os.path.join(pred_dir, f)
        for key in _aliases(f):
            lookup.setdefault(key, full)
    return lookup


def count_files(pred_dir, pred_ext):
    if pred_dir is None or not os.path.isdir(pred_dir):
        return 0
    return sum(
        1
        for f in os.listdir(pred_dir)
        if os.path.isfile(os.path.join(pred_dir, f))
        and (not pred_ext or f.lower().endswith(pred_ext.lower()))
    )


def to_binary_u8(path):
    """读入任意 mask, 返回 (uint8 {0,1} 数组, raw_max, 是否已是规范 0/1)。"""
    with Image.open(path) as im:
        arr = np.array(im.convert("L"))
    raw_max = int(arr.max())
    uniq = np.unique(arr)
    if raw_max <= 1:
        return (arr > 0).astype(np.uint8), raw_max, True
    return (arr >= 128).astype(np.uint8), raw_max, False


def main():
    ap = argparse.ArgumentParser(description="UAV-Crack 提交打包")
    ap.add_argument("--pred-dir", default=None, help="预测 mask 所在目录")
    ap.add_argument(
        "--val-img-dir",
        default="UAV-Crack-dataset/leftImg8bit/val",
        help="官方验证集图片目录 (决定文件名清单与目标分辨率)",
    )
    ap.add_argument("--pred-ext", default=".png", help="预测文件扩展名过滤")
    ap.add_argument("--out-dir", default="result", help="规范化输出目录")
    ap.add_argument("--zip", dest="zip_path", default="result.zip", help="输出 zip 路径")
    ap.add_argument(
        "--dummy",
        action="store_true",
        help="忽略 --pred-dir, 生成全 0 占位提交 (用于验证提交流程)",
    )
    ap.add_argument(
        "--dry-run", action="store_true", help="只做校验, 不写文件也不打包"
    )
    ap.add_argument(
        "--allow-missing",
        action="store_true",
        help="允许缺失预测并继续打包 (默认缺失会终止, 因为比赛要求正好 300 张)",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="输出目录已存在且非空时, 先清空再写入",
    )
    args = ap.parse_args()

    if not args.dummy and not args.pred_dir:
        raise SystemExit("[FATAL] 需要 --pred-dir, 或者加 --dummy 生成占位提交")

    val_files = sorted(
        f
        for f in os.listdir(args.val_img_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )
    if not val_files:
        raise SystemExit(f"[FATAL] 验证集目录为空: {args.val_img_dir}")

    lookup = build_lookup(args.pred_dir, args.pred_ext)

    print("=" * 70)
    print("UAV-Crack 提交打包检查")
    print("=" * 70)
    print(f"  验证集图片 : {args.val_img_dir}  ({len(val_files)} 张)")
    if args.dummy:
        print("  模式       : dummy (全部输出 0)")
    else:
        print(f"  预测目录   : {args.pred_dir}  ({count_files(args.pred_dir, args.pred_ext)} 个候选文件)")

    problems = {"missing": [], "size": [], "nonstandard": [], "empty": []}
    records = []

    for f in val_files:
        stem = os.path.splitext(f)[0]
        out_name = stem + ".png"

        with Image.open(os.path.join(args.val_img_dir, f)) as im:
            want_w, want_h = im.size

        if args.dummy:
            mask = np.zeros((want_h, want_w), dtype=np.uint8)
            records.append((out_name, mask))
            continue

        src = lookup.get(f) or lookup.get(stem)
        if src is None:
            problems["missing"].append(stem)
            continue

        mask, raw_max, standard = to_binary_u8(src)
        h, w = mask.shape
        if (w, h) != (want_w, want_h):
            problems["size"].append((stem, (w, h), (want_w, want_h)))
        if not standard:
            problems["nonstandard"].append((stem, raw_max))
        if mask.sum() == 0:
            problems["empty"].append(stem)

        records.append((out_name, mask))

    print()
    print("-" * 70)
    n_ok = len(records)
    print(f"  可打包数量 : {n_ok} / {len(val_files)}")
    if problems["missing"]:
        print(f"  [ERROR] 缺失 {len(problems['missing'])} 张, 例: {problems['missing'][:5]}")
    if problems["size"]:
        print(f"  [ERROR] 分辨率不符 {len(problems['size'])} 张:")
        for stem, got, want in problems["size"][:5]:
            print(f"          {stem}: 预测{got} 期望{want}")
    if problems["nonstandard"]:
        print(
            f"  [WARN] 非 0/1 像素值 {len(problems['nonstandard'])} 张 "
            f"(已规范化, 例: {problems['nonstandard'][:3]})"
        )
    if problems["empty"]:
        print(
            f"  [WARN] 预测为全背景(无裂缝) {len(problems['empty'])} 张, "
            "若占比过高会拖垮 Crack Recall"
        )
    if not any(problems.values()):
        print("  校验结果   : 全部通过")
    print("-" * 70)

    fatal = bool(problems["missing"] or problems["size"])
    if fatal and not args.allow_missing:
        print("\n[FATAL] 存在缺失或尺寸错误, 已终止打包。")
        print("        确认无误可加 --allow-missing 强制继续。")
        return 1
    if n_ok == 0:
        print("\n[FATAL] 没有任何可打包的预测。")
        return 1

    if args.dry_run:
        print("\n[dry-run] 未写入任何文件。")
        return 0

    # ---- 写入规范化结果 ----
    if os.path.isdir(args.out_dir):
        existing = os.listdir(args.out_dir)
        if existing:
            if not args.force:
                raise SystemExit(
                    f"[FATAL] 输出目录 {args.out_dir} 已存在且非空 "
                    f"({len(existing)} 个文件)。确认可以覆盖请加 --force。"
                )
            shutil.rmtree(args.out_dir)
    os.makedirs(args.out_dir, exist_ok=True)

    for name, mask in records:
        # 单通道 uint8 -> PIL 自动为 'L' 模式, 像素值保持 {0, 1}
        Image.fromarray(mask).save(os.path.join(args.out_dir, name), optimize=True)

    # ---- 打包 ----
    if os.path.exists(args.zip_path):
        os.remove(args.zip_path)
    base = os.path.basename(os.path.normpath(args.out_dir))
    with zipfile.ZipFile(args.zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, _ in records:
            zf.write(os.path.join(args.out_dir, name), arcname=f"{base}/{name}")

    # ---- 复核 zip 内容 ----
    with zipfile.ZipFile(args.zip_path) as zf:
        names = zf.namelist()
    roots = {n.split("/")[0] for n in names}
    non_png = [n for n in names if not n.lower().endswith(".png")]

    print()
    print(f"  输出目录   : {args.out_dir}  ({len(records)} 个 png)")
    print(f"  压缩包     : {args.zip_path}  ({os.path.getsize(args.zip_path)/1024:.0f} KB)")
    print(f"  zip 顶层   : {sorted(roots)}")
    print(f"  zip 内文件 : {len(names)}")
    if non_png:
        print(f"  [ERROR] zip 内含非 png 文件: {non_png[:5]}")
    if len(names) != len(val_files):
        print(f"  [WARN] zip 内文件数 {len(names)} != 验证集图片数 {len(val_files)}")
    if roots != {base}:
        print(f"  [ERROR] zip 顶层应为单一目录 {base}, 实际为 {sorted(roots)}")
    print("=" * 70)
    print("\n提示: 本地可先自查 zip 命名, 例如")
    print(f"      python -c \"import zipfile;print(zipfile.ZipFile('{args.zip_path}').namelist()[:3])\"")
    return 0


if __name__ == "__main__":
    sys.exit(main())
