"""用训好的 Mask2Former 权重, 对官方那 300 张 val 图跑推理并导出提交用 mask。

输出严格遵循比赛要求:
    - 单通道 png, 像素值 0(背景) / 1(裂缝)
    - 分辨率与输入一致, 672x378
    - 文件名 = 输入 jpg 换扩展名, 保留中间的 .JPG_xx_z

之后用 tools/make_submission.py 打包即可 (本脚本也可以直接 --zip 一步到位)。

用法:
    # 用 best 权重推理官方 300 张
    python tools/infer_submit.py \
        --config configs/uavcrack_mask2former_swin-t.py \
        --checkpoint work_dirs/uavcrack_mask2former_swin-t/best_Crack_F1_iter_XXXXX.pth

    # 推理 + 直接打包成 result.zip
    python tools/infer_submit.py -c ... -k ... --zip

    # 想调分割阈值 (默认 0.5), mask 概率大于该值判为裂缝
    python tools/infer_submit.py -c ... -k ... --thr 0.4

    # 在本地验证集上复评某个权重, 再交给 eval_metrics.py 核对
    python tools/infer_submit.py -c ... -k ... \
        --split-file splits/val.txt --out-dir val_pred
    python tools/eval_metrics.py --pred-dir val_pred --split-file splits/val_pairs.txt
"""
import argparse
import os
import os.path as osp
import sys

import numpy as np
import torch
from PIL import Image

PROJ_ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)


def parse_args():
    ap = argparse.ArgumentParser(description='UAV-Crack 推理并导出提交 mask')
    ap.add_argument('-c', '--config', required=True, help='配置文件')
    ap.add_argument('-k', '--checkpoint', required=True, help='模型权重 .pth')
    ap.add_argument(
        '--img-dir',
        default='UAV-Crack-dataset/leftImg8bit/val',
        help='待推理图片目录 (官方 300 张)。给了 --split-file 时忽略此项')
    ap.add_argument(
        '--split-file',
        default=None,
        help='本地验证集清单 (如 splits/val.txt)。给了它就在本地 val 上推理, '
        '配合 tools/eval_metrics.py 复评权重')
    ap.add_argument(
        '--data-root',
        default='UAV-Crack-dataset/leftImg8bit/train',
        help='--split-file 模式下的图片根目录')
    ap.add_argument('--img-suffix', default='.jpg', help='--split-file 模式下的图片后缀')
    ap.add_argument('--out-dir', default='test_pred', help='预测 mask 输出目录')
    ap.add_argument(
        '--thr',
        type=float,
        default=0.5,
        help='裂缝判定阈值, sigmoid 概率 > thr 记为 1。默认 0.5')
    ap.add_argument('--zip', action='store_true', help='推理完直接打包 result.zip')
    ap.add_argument('--device', default='cuda')
    return ap.parse_args()


def build_model(cfg_path, ckpt_path, device):
    from mmengine.config import Config
    from mmengine.runner import load_checkpoint
    from mmseg.registry import MODELS
    from mmseg.utils import register_all_modules
    import uavcrack  # noqa: F401

    register_all_modules()
    cfg = Config.fromfile(cfg_path)
    model = MODELS.build(cfg.model)
    load_checkpoint(model, ckpt_path, map_location='cpu', strict=False)
    model.to(device).eval()
    return cfg, model


def build_pipeline(cfg):
    """只保留 LoadImageFromFile -> Resize -> PackSegInputs。

    config 里的 test_pipeline 带 LoadAnnotations (需要真值), 官方 val 没有标注,
    所以这里单独拼一个不含标注的版本, 其余参数与训练时保持一致。
    """
    from mmengine.dataset import Compose

    transforms = []
    for t in cfg.test_pipeline:
        if t['type'] == 'LoadAnnotations':
            continue
        transforms.append(t)
    return Compose(transforms)


@torch.no_grad()
def main():
    args = parse_args()
    os.chdir(PROJ_ROOT)

    if not osp.exists(args.checkpoint):
        raise SystemExit(f'[FATAL] 权重文件不存在: {args.checkpoint}')

    cfg, model = build_model(args.config, args.checkpoint, args.device)
    pipeline = build_pipeline(cfg)

    # 两种输入模式:
    #   官方提交  -> --img-dir 平铺目录, 文件名原样保留
    #   本地复评  -> --split-file 清单, 路径由 data_root + 清单行拼出
    if args.split_file:
        with open(args.split_file, encoding='utf-8') as f:
            stems = [
                line.strip().split()[0] for line in f if line.strip()
            ]
        if not stems:
            raise SystemExit(f'[FATAL] 清单为空: {args.split_file}')
        sources = []
        for stem in stems:
            p = osp.join(args.data_root, stem + args.img_suffix)
            if not osp.exists(p):
                raise SystemExit(f'[FATAL] 清单里的图片不存在: {p}')
            sources.append((osp.basename(stem), p))
        src_desc = f'{args.split_file}  ({len(sources)} 条)'
    else:
        names = sorted(
            f for f in os.listdir(args.img_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png')))
        if not names:
            raise SystemExit(f'[FATAL] 图片目录为空: {args.img_dir}')
        sources = [
            (osp.splitext(f)[0], osp.join(args.img_dir, f)) for f in names
        ]
        src_desc = f'{args.img_dir}  ({len(sources)} 张)'

    os.makedirs(args.out_dir, exist_ok=True)

    print('=' * 70)
    print('UAV-Crack 推理')
    print('=' * 70)
    print(f'  权重     : {args.checkpoint}')
    print(f'  输入     : {src_desc}')
    print(f'  输出     : {args.out_dir}')
    print(f'  阈值     : {args.thr}')
    print(f'  设备     : {args.device}')
    print('=' * 70)

    crack_ratio_sum = 0.0
    all_zero = 0
    shape_bad = []

    for i, (stem, path) in enumerate(sources, 1):
        data = pipeline(dict(img_path=path))
        data_samples = [data['data_samples']]
        # EncoderDecoder.predict 不会自己调 data_preprocessor, 得手动走一遍
        # (BGR->RGB、按数据集均值方差归一化、pad 到 32 的倍数)
        pre = model.data_preprocessor(
            dict(inputs=data['inputs'].unsqueeze(0), data_samples=data_samples),
            training=False)
        inputs = pre['inputs'].to(args.device)

        ori_hw = tuple(data_samples[0].metainfo['ori_shape'][:2])
        pred = model.predict(inputs, data_samples)
        logits = pred[0].pred_sem_seg.data  # (H, W)
        if logits.dim() == 3:
            logits = logits.squeeze(0)

        # 模型是在 pad 到 384 高的图上算的, 这里裁回原始 378x672
        if tuple(logits.shape) != ori_hw:
            logits = logits[:ori_hw[0], :ori_hw[1]]

        mask = (logits > 0).to(torch.uint8).cpu().numpy()
        if mask.shape != ori_hw:
            shape_bad.append((name, mask.shape, ori_hw))

        ratio = float(mask.mean())
        crack_ratio_sum += ratio
        if mask.sum() == 0:
            all_zero += 1

        Image.fromarray(mask).save(osp.join(args.out_dir, f'{stem}.png'))

        if i % 50 == 0 or i == len(sources):
            print(f'  [{i}/{len(sources)}]  平均裂缝占比 {crack_ratio_sum / i:.2%}')

    print('-' * 70)
    print(f'  平均预测裂缝占比 : {crack_ratio_sum / len(sources):.2%}')
    print('  参考: 训练集真值裂缝占比约 6.2%, 越接近说明尺度越对')
    if all_zero:
        print(f'  [WARN] 有 {all_zero} 张预测为全背景 (无裂缝)')
    if shape_bad:
        print(f'  [ERROR] {len(shape_bad)} 张尺寸不对: {shape_bad[:3]}')
    print('=' * 70)

    if args.zip and args.split_file:
        raise SystemExit(
            '[FATAL] --zip 只能用于官方 300 张 val。本地验证集请改用 '
            'tools/eval_metrics.py 对比真值。')

    if args.zip:
        cmd = (f'"{sys.executable}" tools/make_submission.py '
               f'--pred-dir {args.out_dir} --out-dir result --zip result.zip '
               f'--force')
        print(f'\n打包提交: {cmd}\n')
        os.system(cmd)


if __name__ == '__main__':
    main()
