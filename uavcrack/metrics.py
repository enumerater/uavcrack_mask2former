"""与 UAV-Crack 官方评测口径完全一致的指标。

mmseg 自带的 IoUMetric 只输出"各类平均"(mIoU / mFscore ...)。本比赛 90% 的
像素是背景, 背景类的指标几乎恒定, 取平均会把裂缝类的真实变化稀释掉一半;
而且 mFscore 是"背景 F1 与裂缝 F1 的均值", 不等于官方计分用的 Crack F1。

所以这里单独实现, 直接输出官方那 6 个数字, 让 `save_best` 盯的指标
和排行榜打分的指标是同一个东西。

类别约定: 0 = 背景 (normal road surface), 1 = 裂缝 (crack)。
"""
from collections import OrderedDict
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger, print_log
from mmengine.utils import is_main_process, mkdir_or_exist
from PIL import Image

from mmseg.registry import METRICS

Image.MAX_IMAGE_PIXELS = None


@METRICS.register_module()
class UAVCrackMetric(BaseMetric):
    """二值裂缝分割指标。

    Args:
        ignore_index (int): 该像素不参与统计。默认 255, 与 mmseg 的
            ``seg_pad_val`` 一致, 这样 padding 出来的区域会被自动排除。
        output_dir (str, optional): 若给出, 则把预测 mask 按
            ``<原文件名>.png`` 存到该目录, 像素值为 0/1, 可直接用于提交。
        format_only (bool): 只导出预测不做统计。官方那 300 张 val 没有标注,
            推理时用 True。
        collect_device (str): 多卡聚合设备, 单卡不用管。
        prefix (str, optional): 指标名前缀。
    """

    def __init__(self,
                 ignore_index: int = 255,
                 output_dir: Optional[str] = None,
                 format_only: bool = False,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.ignore_index = ignore_index
        self.output_dir = output_dir
        self.format_only = format_only
        if self.output_dir and is_main_process():
            mkdir_or_exist(self.output_dir)

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        for data_sample in data_samples:
            pred = data_sample['pred_sem_seg']['data'].squeeze()
            pred = (pred == 1).to(torch.uint8)

            if self.output_dir is not None:
                img_path = data_sample.get('img_path')
                if img_path is None:
                    raise KeyError(
                        'data_sample 里没有 img_path, 无法确定导出文件名')
                basename = _split_ext(_basename(str(img_path)))
                out = (pred.cpu().numpy() * 1).astype(np.uint8)
                # 单通道 {0,1}; 比赛要求像素值就是 0/1, 不是 0/255
                Image.fromarray(out).save(
                    f'{self.output_dir}/{basename}.png')

            if self.format_only:
                continue

            gt = data_sample['gt_sem_seg']['data'].squeeze().to(pred)
            valid = gt != self.ignore_index
            g = gt[valid].to(torch.int64)
            p = pred[valid].to(torch.int64)

            self.results.append(
                dict(
                    tp=int(((p == 1) & (g == 1)).sum()),
                    fp=int(((p == 1) & (g == 0)).sum()),
                    fn=int(((p == 0) & (g == 1)).sum()),
                    tn=int(((p == 0) & (g == 0)).sum()),
                ))

    def compute_metrics(self, results: List[dict]) -> Dict[str, float]:
        logger: MMLogger = MMLogger.get_current_instance()
        if self.format_only:
            print_log(f'预测已导出到 {self.output_dir}', logger=logger)
            return OrderedDict()

        eps = 1e-12
        tp = sum(r['tp'] for r in results)
        fp = sum(r['fp'] for r in results)
        fn = sum(r['fn'] for r in results)
        tn = sum(r['tn'] for r in results)

        iou_bg = tn / (tn + fn + fp + eps)
        iou_crack = tp / (tp + fn + fp + eps)
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)
        acc = (tp + tn) / (tp + tn + fp + fn + eps)

        print_log(
            f'裂缝像素: tp={tp:,} fp={fp:,} fn={fn:,} | '
            f'预测裂缝占比={100 * (tp + fp) / (tp + fp + tn + fn + eps):.2f}% | '
            f'真实裂缝占比={100 * (tp + fn) / (tp + fp + tn + fn + eps):.2f}%',
            logger=logger)

        return OrderedDict([
            ('mIoU', float((iou_bg + iou_crack) / 2)),
            ('aAcc', float(acc)),
            ('Crack_F1', float(f1)),
            ('Crack_IoU', float(iou_crack)),
            ('Crack_Precision', float(precision)),
            ('Crack_Recall', float(recall)),
        ])


def _basename(path: str) -> str:
    return path.replace('\\', '/').rsplit('/', 1)[-1]


def _split_ext(name: str) -> str:
    i = name.rfind('.')
    return name[:i] if i > 0 else name
