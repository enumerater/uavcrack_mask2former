# UAV-Crack 无人机路面裂缝分割

参加 UAV-Crack Benchmark 的比赛方案：**Mask2Former + Swin-T**（基于 mmsegmentation 1.2.2），
单卡 RTX 4060 Laptop 8G 可训。

任务是把无人机航拍的路面图做**二值分割**：`0 = 正常路面`，`1 = 裂缝`。

---

## 目录

- [一、环境](#一环境)
- [二、数据](#二数据)
- [三、快速开始](#三快速开始)
- [四、项目结构](#四项目结构)
- [五、配置说明](#五配置说明)
- [六、指标口径](#六指标口径)
- [七、踩过的坑](#七踩过的坑)

---

## 一、环境

conda 环境 `uavcrack`，Python 3.10。

| 包 | 版本 | 备注 |
|---|---|---|
| torch | 2.1.0+cu118 | 4060 是 sm_89，cu118 支持 |
| torchvision | 0.16.0+cu118 | 必须与 torch 配对 |
| mmcv | 2.1.0 | 用官方 Windows 预编译轮子 |
| mmengine | 0.10.4 | |
| mmdet | 3.3.0 | **必需**，mmseg 的 Mask2Former 继承自它 |
| mmsegmentation | 1.2.2 | 源码安装在 `d:\1offer\changan_u_seg\mmsegmentation` |
| numpy | **< 2** | mmcv 2.x 不兼容 numpy 2 |

### 为什么必须是这几个版本

这不是随便选的最新版，是一串版本死锁逼出来的唯一解：

```
mmseg 1.2.2  ┐
              ├─ 都要求  mmcv >= 2.0.0rc4 且 < 2.2.0
mmdet 3.3.0  ┘
```

而 Windows 上 mmcv 的官方预编译轮子**最高只到 `cu118/torch2.3.0`**，那个目录下恰好只有
mmcv 2.2.0 —— 被上限卡死。往下退一档到 `cu118/torch2.1.0`，才有满足条件的 mmcv 2.1.0。
所以 torch 必须跟着降到 2.1.0。

> 别试图"顺手升级"其中任何一项。改一个，整条链就断。

### 安装

```bat
conda create -n uavcrack python=3.10 -y
conda activate uavcrack

python -m pip install -U pip setuptools==75.8.0 wheel -i https://pypi.tuna.tsinghua.edu.cn/simple

pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118

pip install mmengine==0.10.4 opencv-python matplotlib scipy prettytable "numpy<2" -i https://pypi.tuna.tsinghua.edu.cn/simple

pip install "https://download.openmmlab.com/mmcv/dist/cu118/torch2.1.0/mmcv-2.1.0-cp310-cp310-win_amd64.whl" "numpy<2"

pip install mmdet==3.3.0 --no-deps
pip install pycocotools shapely terminaltables tqdm -i https://pypi.tuna.tsinghua.edu.cn/simple

cd /d d:\1offer\changan_u_seg\mmsegmentation
pip install -e . --no-build-isolation

cd /d d:\1offer\changan_u_seg\maskformer
pip install ftfy regex -i https://pypi.tuna.tsinghua.edu.cn/simple
```

几个不能省的细节：

- mmcv 必须用**直连 wheel 地址**。写 `pip install mmcv==2.1.0` 会让 pip 去拉源码编译然后失败
- mmdet 必须加 `--no-deps`，否则它会去动 mmcv / mmengine / numpy
- 装完 mmseg 还缺 `ftfy` 和 `regex` —— `mmseg/utils/tokenizer.py` 无条件 import 了它们，
  但没写进 runtime 依赖
- mmseg 用 `--no-build-isolation` 配合 setuptools 75.8.0。setuptools 78+ 会因为
  `setup.py` 里过期的 License classifier 直接报错

### 环境自检

```bat
python -c "import torch,mmcv,mmdet,mmseg,mmengine,numpy;                                        \
print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0));             \
print('mmcv',mmcv.__version__,'mmdet',mmdet.__version__,'mmengine',mmengine.__version__,         \
      'mmseg',mmseg.__version__,'numpy',numpy.__version__)"
```

期望：`2.1.0+cu118 True NVIDIA GeForce RTX 4060 Laptop GPU` 以及各版本与上表一致。

---

## 二、数据

```
UAV-Crack-dataset/
├── leftImg8bit/
│   ├── train/{UAV-CrackX4, UAV-CrackX8, UAV-CrackX16}/   各 400 张 .jpg
│   └── val/                                              300 张 .jpg（无标注，用于提交）
└── gtFine/
    └── train/{UAV-CrackX4, UAV-CrackX8, UAV-CrackX16}/   各 400 张 .png
```

| 项 | 实测值 |
|---|---|
| 训练图像 | **1200 张**（X4/X8/X16 各 400） |
| 测试图像 | 300 张，无标注 |
| 分辨率 | 全部 672×378 |
| 标注格式 | 单通道 `L` 模式 PNG，像素值严格 `{0, 1}` |
| 裂缝像素占比 | 5.5% ~ 6.5%，**没有纯背景图** |

### 本地验证集怎么来的

官方那 300 张没有标注，无法在本地算任何指标 —— 只能上传等排行榜。所以从 1200 张
训练图里划出 20% 当本地考场：

```
splits/train.txt       979 张 / 158 个源图
splits/val.txt         221 张 /  39 个源图      (X4:87  X8:54  X16:80)
splits/val_pairs.txt   同上，两列完整路径，给 eval_metrics.py 用
```

**关键：切分是按"源图"而不是按 patch 随机的。**

文件名 `937956_DJI_20231015154707_0002_Z.JPG_12_z.jpg` 里，`937956_..._0002_Z` 是源图 ID，
`_12_z` 是它切出来的第 12 个 patch。实测官方 val 的 142 个源图里有 **137 个和训练集重源**。
如果按 patch 随机切，同一个源图的相邻 patch 会一个进训练一个进验证，路面纹理几乎一样，
本地分数会明显虚高，导致选错超参。

重新生成（seed 固定，结果可复现）：

```bat
python tools/split_train_val.py --val-ratio 0.2 --seed 42
```

---

## 三、快速开始

### 1. 训练

```bat
cd /d d:\1offer\changan_u_seg\maskformer
python tools/train.py configs/uavcrack_mask2former_swin-t.py
```

启动后会先打印配置摘要，核对 GPU / batch / lr / max_iters 无误后让它跑。

预计 **约 4.4 小时**（30000 iter，实测 0.53 秒/iter）。中途随时 `Ctrl+C`，
`best_Crack_F1_*.pth` 已经落盘不会丢。

续训：

```bat
python tools/train.py configs/uavcrack_mask2former_swin-t.py --resume
```

换实验目录 / 临时改配置：

```bat
python tools/train.py configs/uavcrack_mask2former_swin-t.py --work-dir work_dirs/exp1
python tools/train.py configs/uavcrack_mask2former_swin-t.py --cfg-options train_dataloader.batch_size=4
```

### 2. 看训练进度

日志在 `work_dirs/uavcrack_mask2former_swin-t/<时间戳>/<时间戳>.log`。

三种关键行：

```
Iter(train) [150/30000]  lr: 4.99e-05  time: 0.53  memory: 3444  loss: 52.31  grad_norm: 183.42
```
- `loss` 应从 ~90 持续下降
- `grad_norm` 必须是**有限值**且在下降。出现 `nan` / `inf` 立刻停下
- `memory` 约 3400 正常

```
Iter(val) [221/221]  mIoU: 0.5xxx  aAcc: 0.9xxx  Crack_F1: 0.3xxx
                     Crack_IoU: 0.2xxx  Crack_Precision: 0.xxxx  Crack_Recall: 0.xxxx
                          ← 这就是记分板，每 1000 iter（约 9 分钟）一行
```

```
The best checkpoint with 0.4123 Crack_F1 at 12000 iter is saved to best_Crack_F1_iter_12000.pth
```

判断是否在正常学习：`Crack_F1` 应在前 5000 iter 内从 0.1 爬到 0.3 以上。
到 8000 iter 还趴在 0.15 以下就是有问题。

另外会打印一行裂缝像素占比，可以提前发现模型退化：

```
裂缝像素: tp=... fp=... fn=... | 预测裂缝占比=6.4% | 真实裂缝占比=6.2%
```

预测占比应逐渐收敛到接近 6.2%。如果是 90% 或 0.5%，说明还没学好或学歪了。

### 3. 模型在哪

```
work_dirs/uavcrack_mask2former_swin-t/
├── best_Crack_F1_iter_12000.pth   ← 最终要用的是这个（验证集 Crack F1 最高的一次）
├── iter_28000.pth                 ← 最近 3 个快照，旧的自动删
├── iter_29000.pth
├── iter_30000.pth
├── last_checkpoint                ← 文本，内容指向最近一次的权重名
└── 20260916_223000/
    ├── 20260916_223000.log        ← 完整日志
    └── vis_data/                  ← 训练曲线
```

单个权重约 570MB（含优化器状态），跑完全程目录占用约 2~3GB。

### 4. 评估

**（a）训练中自动评估 —— 主要手段，无需额外操作**

每 1000 iter 自动在本地验证集（221 张独立源图）上算全部 6 个指标打进日志。
`save_best='Crack_F1'` 会自动保留最佳权重。你要做的就是**盯 Crack_F1 什么时候不再涨**，
不涨了就可以停。

**（b）单独复评某个权重**

比如对比 `iter_20000.pth` 和 `best_Crack_F1_iter_12000.pth` 哪个更好，
或者提交前最后确认一次：

```bat
python tools/infer_submit.py -c configs/uavcrack_mask2former_swin-t.py ^
    -k work_dirs/uavcrack_mask2former_swin-t/best_Crack_F1_iter_12000.pth ^
    --split-file splits/val.txt --out-dir val_pred

python tools/eval_metrics.py --pred-dir val_pred --split-file splits/val_pairs.txt --check-size
```

**（c）交叉验证**

`eval_metrics.py` 是一条完全独立的代码路径。它的数字应该和训练日志里的**完全一致**：

```
训练日志     Crack_F1: 0.4123
eval_metrics Crack F1: 0.4123     ← 对不上就说明有一边口径有问题
```

### 5. 生成提交

```bat
python tools/infer_submit.py -c configs/uavcrack_mask2former_swin-t.py ^
    -k work_dirs/uavcrack_mask2former_swin-t/best_Crack_F1_iter_12000.pth ^
    --out-dir test_pred --zip
```

`--zip` 会顺带调 `make_submission.py`，自动校验 300 张、分辨率、像素值 0/1、
文件名、zip 顶层目录，任何一项不对都会报错拦下来。产物是 `result.zip`。

调分割阈值（默认 0.5）：

```bat
python tools/infer_submit.py -c configs/... -k ... --out-dir test_pred --thr 0.4 --zip
```

---

## 四、项目结构

```
maskformer/
├── configs/
│   └── uavcrack_mask2former_swin-t.py   主配置
├── uavcrack/
│   ├── __init__.py
│   └── metrics.py                       UAVCrackMetric：与排行榜口径一致
├── tools/
│   ├── train.py                         训练入口
│   ├── infer_submit.py                  推理 + 导出提交 mask
│   ├── make_submission.py               格式校验 + 打包 result.zip
│   ├── eval_metrics.py                  独立复算 6 个指标
│   └── split_train_val.py               按源图切分本地验证集
├── splits/                              切分清单（已生成）
├── result/ + result.zip                 占位提交，格式已验证
├── work_dirs/                           训练产物（已 gitignore）
└── UAV-Crack-dataset/                   数据集（已 gitignore）
```

`uavcrack/` 是本项目的插件目录，通过 `tools/train.py` 把项目根目录加进 `sys.path`
后再由 config 的 `custom_imports` 导入。**不改 mmseg 源码**，好处是 mmseg 更新不会覆盖掉。

---

## 五、配置说明

配置文件：`configs/uavcrack_mask2former_swin-t.py`

### 常改的几项

| 想改什么 | 改哪个 |
|---|---|
| 训练总时长 | `MAX_ITERS`（单位是 micro-batch，见下） |
| 显存不够 | `train_dataloader.batch_size` 降到 1，同时 `accumulative_counts` 翻倍 |
| 显存还有富余想提速 | `batch_size` 提到 4（显存 5.9G），吞吐 +30% |
| 加快训练 | `backbone.with_cp=True` 省 1.5G 但慢 30%；或把 `MAX_ITERS` 调小 |
| 二分类 query 太多 | `decode_head.num_queries` 从 100 降到 30 |

### 当前参数

```
crop_size        384 x 672      贴合原图 378x672 的 16:9，高是 32 的倍数
num_classes      2              0=背景 1=裂缝
backbone         Swin-T         ImageNet 预训练
num_queries      100
batch_size       2      x  梯度累积 4  = 等效 batch 8
lr               5e-5           官方 1e-4 对应 batch 16，按比例折半
optimizer        AdamW  weight_decay 0.05  clip_grad max_norm 0.01
精度             fp32           【不要开 AMP，见第七节】
MAX_ITERS        30000          ≈ 7500 优化步 ≈ 60000 张图 ≈ 61 轮 ≈ 4.4 小时
val_interval     1000           约 9 分钟验证一次
save_best        Crack_F1       直接盯比赛真正打分的指标
```

### 类别不平衡的处理

裂缝只占 6%，纯 CE 会退化成"全预测背景"。Mask2Former 自带两道防线：

1. `loss_dice`（`naive_dice=True`）—— Dice 与类别比例无关，天然抗不平衡
2. 点采样用 `importance_sample_ratio=0.75` 的 importance sampling —— 偏向难样本和前景

在此之上把分类损失的裂缝类权重上调到 2.0（`class_weight=[1.0, 2.0, 0.1]`，最后一位是
"无目标"的权重）。如果验证时发现 **Recall 上不去**，优先把 Dice 权重从 5.0 往上调。

### 归一化

`mean/std` 是在本地 1200 张训练图上实测的，不是 ImageNet 的默认值：

```
RGB mean [138.08, 131.10, 120.74]     std [37.63, 33.72, 30.23]
```

配置里 `bgr_to_rgb=True` 表示先 BGR→RGB 再减均值，所以填的是 **RGB 顺序**。

---

## 六、指标口径

比赛 6 个指标，本项目的 `UAVCrackMetric` 与 `eval_metrics.py` 用两套独立代码实现，
互相校验。

设 `tp/fp/fn/tn` 为裂缝类的混淆矩阵元素（背景=0，裂缝=1）：

```
Crack IoU        = tp / (tp + fp + fn)
Crack Precision  = tp / (tp + fp)
Crack Recall     = tp / (tp + fn)
Crack F1         = 2PR / (P + R)
IoU(背景)         = tn / (tn + fp + fn)
mIoU             = (Crack IoU + IoU(背景)) / 2
aAcc             = (tp + tn) / 总数
```

### 优化目标是 Crack F1 / Crack IoU，不是 aAcc

实测全 0 提交的分数：

| 指标 | 全 0 预测 |
|---|---|
| **Crack F1** | **0.0000** |
| **Crack IoU** | **0.0000** |
| **aAcc** | **0.9381** |
| mIoU | 0.4691 |

aAcc 高达 93.8%，但裂缝一个都没找到。**按 aAcc 优化会得到一个交了等于没交的模型。**

### 为什么不用 mmseg 自带的 IoUMetric

它只输出"各类平均"（`mIoU` / `mFscore` / ...）。90% 的像素是背景，背景类指标几乎恒定，
取平均会把裂缝类的真实变化稀释掉一半；而且 `mFscore` 是"背景 F1 与裂缝 F1 的均值"，
不等于官方计分的 Crack F1。

### 提交格式的四个硬要求

`make_submission.py` 会自动校验，任何一条不满足都会报错：

1. 像素值必须是 `0` 和 `1`，**不是 0 和 255**（因为官方 GT 就是 0/1）
2. 文件名保留中间的 `.JPG_12_z`，只把末尾的 `.jpg` 换成 `.png`
3. zip 里必须有一层 `result/` 目录
4. 正好 300 张，多一个文件都不行

---

## 七、踩过的坑

### 1. fp16 / AMP 会让训练静默崩溃 ❌

```
ValueError: cost matrix is infeasible     ← mmdet HungarianAssigner
grad_norm: nan
```

Mask2Former 的匈牙利匹配代价矩阵在 fp16 下会产生 NaN。这个是硬伤，**不要开 AMP**。

实际上也不需要 —— 纯 fp32 在 384×672 / batch 2 下只占 **3.4G** 显存，8G 完全放得下。
改成 fp32 后：

```
grad_norm: 725.77 → 606.04 → 424.49 → 250.49     有限且稳步下降
loss:       92.52 →  76.70 →  64.92 →  58.95     正常收敛
```

### 2. `max_iters` 数的是 micro-batch，不是优化步数 ⚠️

`mmengine/runner/loops.py` 里每次 `run_iter` 只调一次 `train_step`，梯度要攒够
`accumulative_counts` 次才真正 step 一次优化器。所以：

```
优化步数 = MAX_ITERS / accumulative_counts
训练张数 = MAX_ITERS * batch_size
```

当前配置 30000 / 4 = 7500 个优化步。如果按"30000 个优化步"来估时间会差 4 倍。

### 3. 数据切分必须按源图，不能按 patch ⚠️

见 [第二节](#本地验证集怎么来的)。按 patch 随机切会让本地分数虚高。

### 4. Windows 的 DataLoader 必须用 spawn

config 里 `env_cfg.mp_cfg.mp_start_method='spawn'`。Windows 不支持 `fork`，
写成 `fork` 会直接崩。

### 5. `MSDeformAttn` 是 mmdet 的符号名

mmcv 里对应的叫 `MultiScaleDeformableAttention`。冒烟测试时写错名字会误判成环境坏了。

### 6. 显存不够时按这个顺序调

1. `backbone.with_cp=True` —— 省 ~1.5G，慢约 30%
2. `train_dataloader.batch_size=1` —— 省 ~2G，同时把 `accumulative_counts` 翻倍到 8
   保持等效 batch 不变
3. `decode_head.num_queries=30` —— 省 mask loss 显存

### 7. 磁盘

单个权重 570MB。`max_keep_ckpts=3` 已经把中间快照限制在 3 个，
加上 `best_*` 和 `last_*`，整个 `work_dirs/` 约 2~3GB。

---

## 后续可优化方向

- [ ] **调分割阈值**。默认 0.5，在本地验证集上扫 0.3~0.6 通常能再榨出一点 F1
- [ ] 测试时增强（TTA）：多尺度 + 水平翻转
- [ ] 调整 `loss_dice` 权重改善 Recall
- [ ] 多个 checkpoint 做权重平均（SWA / 模型集成）
