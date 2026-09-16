"""UAV-Crack Mask2Former 训练入口。

相比 mmseg 官方 tools/train.py, 这里多做三件事:
  1. 自动 chdir 到项目根目录, 让 config 里的相对路径 (data_root、splits/...)
     不受当前终端所在目录影响
  2. 把项目根目录加进 sys.path, 让 config 的 custom_imports 能 import 到
     uavcrack.metrics (比赛口径的验证指标)
  3. 启动前打印显存与配置摘要, OOM 时给出下一步该调哪个开关

用法:
    python tools/train.py configs/uavcrack_mask2former_swin-t.py
    python tools/train.py configs/uavcrack_mask2former_swin-t.py --work-dir work_dirs/exp1
    python tools/train.py configs/uavcrack_mask2former_swin-t.py --resume
    # 临时改配置 (不写进文件):
    python tools/train.py configs/xxx.py --cfg-options train_dataloader.batch_size=1
"""
import argparse
import logging
import os
import os.path as osp
import sys
import warnings

PROJ_ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)


def parse_args():
    parser = argparse.ArgumentParser(description='Train UAV-Crack segmentor')
    parser.add_argument('config', help='配置文件路径')
    parser.add_argument('--work-dir', help='日志与权重保存目录')
    parser.add_argument(
        '--resume',
        action='store_true',
        default=False,
        help='从 work-dir 里最新的 checkpoint 续训')
    parser.add_argument(
        '--amp',
        action='store_true',
        default=False,
        help='开启自动混合精度 (本项目的 config 默认已经开了)')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        help='覆盖配置项, 形如 a.b=1 c="[x,y]"')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none')
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()

    # ---- 关键: 先切到项目根目录, 再 import mmseg 相关模块 ----
    os.chdir(PROJ_ROOT)

    import torch
    from mmengine.config import Config
    from mmengine.logging import print_log
    from mmengine.runner import Runner
    from mmseg.registry import RUNNERS
    import uavcrack  # noqa: F401  注册 UAVCrackMetric

    if args.cfg_options:
        parsed = {}
        for item in args.cfg_options:
            key, _, val = item.partition('=')
            parsed[key.strip()] = _literal(val.strip())
        args.cfg_options = parsed

    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        cfg.work_dir = osp.join(
            './work_dirs', osp.splitext(osp.basename(args.config))[0])

    if args.amp is True:
        if cfg.optim_wrapper.type == 'AmpOptimWrapper':
            print_log(
                '配置里已经开了 AMP, --amp 忽略。',
                logger='current',
                level=logging.WARNING)
        else:
            cfg.optim_wrapper.type = 'AmpOptimWrapper'
            cfg.optim_wrapper.loss_scale = 'dynamic'

    if 'auto_scale_lr' in cfg and cfg.auto_scale_lr.get('enable', False):
        warnings.warn(
            'auto_scale_lr 已启用, 会按 base_batch_size 自动缩放 lr, '
            '注意它不感知 accumulative_counts。')

    if args.resume:
        cfg.resume = True
        cfg.load_from = None

    _print_banner(cfg, torch)

    runner = (RUNNERS.build(cfg) if 'runner_type' not in cfg
              else RUNNERS.get(cfg.runner_type)(**cfg))
    runner.train()
    return runner


def _literal(text):
    """把命令行字符串还原成 int/float/bool/list, 失败就原样返回。"""
    low = text.lower()
    if low in ('true', 'false'):
        return low == 'true'
    if low in ('none', 'null'):
        return None
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        pass
    if text.startswith('[') or text.startswith('('):
        try:
            import ast
            return ast.literal_eval(text)
        except Exception:
            pass
    return text


def _print_banner(cfg, torch):
    bs = cfg.train_dataloader.batch_size
    accum = cfg.optim_wrapper.get('accumulative_counts', 1)
    crop = cfg.get('crop_size', '?')
    n_train = '?'
    try:
        with open(osp.join(cfg.data_root, cfg.train_dataloader.dataset.ann_file),
                  encoding='utf-8') as f:
            n_train = sum(1 for line in f if line.strip())
    except Exception:
        pass

    total_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print('\n' + '=' * 74)
    print('  UAV-Crack / Mask2Former 训练')
    print('=' * 74)
    print(f'  配置        : {cfg.filename}')
    print(f'  工作目录    : {cfg.work_dir}')
    print(f'  GPU         : {torch.cuda.get_device_name(0)}  ({total_mem:.1f} GB)')
    print(f'  AMP         : {cfg.optim_wrapper.type}')
    print(f'  裁剪尺寸    : {crop}')
    print(f'  训练图数    : {n_train}')
    print(f'  batch_size  : {bs}   x 累积 {accum}  = 等效 {bs * accum}')
    print(f'  max_iters   : {cfg.train_cfg.max_iters}  (val 每 '
          f'{cfg.train_cfg.val_interval} iter)')
    print(f'  lr          : {cfg.optim_wrapper.optimizer.lr}')
    print(f'  save_best   : {cfg.default_hooks.checkpoint.save_best}')
    print('-' * 74)
    print('  显存不够时按这个顺序调 (重跑即可):')
    print('    1) backbone.with_cp=True                     省 ~1.5G, 慢约 30%')
    print('    2) train_dataloader.batch_size=1             省 ~2G, 同时把')
    print('       optim_wrapper.accumulative_counts 翻倍到 16 保持等效 batch')
    print('    3) decode_head.num_queries=30                省 mask loss 显存')
    print('=' * 74 + '\n')


if __name__ == '__main__':
    main()
