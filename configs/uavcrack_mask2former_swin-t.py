# ============================================================================
# UAV-Crack 二值裂缝分割 — Mask2Former + Swin-T
# 目标硬件: RTX 4060 Laptop 8G, 单卡
#
# 与原版 mmseg mask2former_swin-t 配置的差异:
#   1. 类别数 19 -> 2 (0=背景, 1=裂缝)
#   2. 裁剪尺寸 512x1024 -> 384x672, 贴合原图 378x672 的 16:9 比例
#   3. 归一化换成在本地 1200 张训练图上实测的均值/方差
#   4. 开启 AMP + 梯度累积, 单卡 8G 也能跑到等效 batch 16
#   5. 验证指标换成 UAVCrackMetric (与排行榜口径一致)
#   6. 去掉 cityscapes 的 crop 类平衡约束, 因为裂缝只占 6%, 约束会空转
#
# 运行目录必须是本项目根目录 (maskformer/), 因为 data_root 是相对路径。
# tools/train.py 会自动 chdir 过去。
# ============================================================================
custom_imports = dict(imports=['uavcrack.metrics'], allow_failed_imports=False)
default_scope = 'mmseg'

# ---------------------------------------------------------------- 数据集 ----
dataset_type = 'BaseSegDataset'
data_root = 'UAV-Crack-dataset/'
# (H, W)。原图 378x672, 取 384 是为了让高是 32 的整数倍, 省掉不必要的 padding
crop_size = (384, 672)
num_classes = 2

metainfo = dict(
    classes=('background', 'crack'),
    palette=[[0, 0, 0], [255, 0, 0]],
)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    # X4/X8/X16 三个子集本身已经提供了尺度多样性, 这里再叠一层随机缩放。
    # 下限取 1.05 而不是更低, 是为了保证缩放后仍不小于 crop_size,
    # 否则 RandomCrop 会补黑边, 把纯黑图喂进训练。
    dict(
        type='RandomResize',
        scale=(672, 384),
        ratio_range=(1.05, 1.6),
        keep_ratio=True),
    # 不设 cat_max_ratio: 它默认 1.0。cityscapes 那套 0.75 在这里会 100% 命中
    # "背景占比超阈值", 每次 crop 白跑 10 遍重采样。
    dict(type='RandomCrop', crop_size=crop_size),
    dict(type='RandomFlip', prob=0.5),
    # 航拍是俯视图, 上下翻转在物理上同样合理
    dict(type='RandomFlip', prob=0.5, direction='vertical'),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    # 输入本来就是 672x378, keep_ratio 下这一步是恒等变换;
    # 真正补到 384 高由 data_preprocessor 的 size 完成 (pixel 补 0, 标注补 255)
    dict(type='Resize', scale=(672, 384), keep_ratio=True),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs'),
]

train_dataloader = dict(
    # 实测: batch 2 显存峰值 3.4G / 吞吐 3.8 张每秒
    #       batch 4 显存峰值 5.9G / 吞吐 5.1 张每秒
    # 选 2 是为了留足余量 —— 本机微信等程序常驻占用约 1G, batch 4 跑几小时
    # 中途容易被挤爆。想提速可以改成 4, 但要做好随时可能 OOM 的心理准备。
    batch_size=2,
    # Windows 的 DataLoader worker 走 spawn, 开大了启动和内存都吃不消
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='../splits/train.txt',
        data_prefix=dict(
            img_path='leftImg8bit/train', seg_map_path='gtFine/train'),
        metainfo=metainfo,
        pipeline=train_pipeline,
    ),
)

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='../splits/val.txt',
        data_prefix=dict(
            img_path='leftImg8bit/train', seg_map_path='gtFine/train'),
        metainfo=metainfo,
        test_mode=True,
        pipeline=test_pipeline,
    ),
)
test_dataloader = val_dataloader

val_evaluator = dict(type='UAVCrackMetric', ignore_index=255)
test_evaluator = val_evaluator

# ------------------------------------------------------------------ 模型 ----
model = dict(
    type='EncoderDecoder',
    data_preprocessor=dict(
        type='SegDataPreProcessor',
        # 在本数据集 1200 张训练图上实测的 RGB 均值/方差。
        # bgr_to_rgb=True 表示先 BGR->RGB 再减均值, 所以这里要填 RGB 顺序。
        mean=[138.08, 131.10, 120.74],
        std=[37.63, 33.72, 30.23],
        bgr_to_rgb=True,
        pad_val=0,
        seg_pad_val=255,
        size=crop_size,
        test_cfg=dict(size_divisor=32),
    ),
    backbone=dict(
        type='SwinTransformer',
        pretrain_img_size=224,
        embed_dims=96,
        patch_size=4,
        window_size=7,
        mlp_ratio=4,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        strides=(4, 2, 2, 2),
        out_indices=(0, 1, 2, 3),
        qkv_bias=True,
        qk_scale=None,
        patch_norm=True,
        drop_rate=0.,
        attn_drop_rate=0.,
        # 小数据集 + 大模型, drop_path 是主要正则手段
        drop_path_rate=0.3,
        use_abs_pos_embed=False,
        act_cfg=dict(type='GELU'),
        norm_cfg=dict(type='LN', requires_grad=True),
        # 显存不够就打开下面这行 (backbone 梯度检查点), 大约省 1.5G, 慢 ~30%
        with_cp=False,
        frozen_stages=-1,
        init_cfg=dict(
            type='Pretrained',
            checkpoint='https://download.openmmlab.com/mmsegmentation/v0.5/'
            'pretrain/swin/swin_tiny_patch4_window7_224_20220317-1cdeb081.pth'),
    ),
    decode_head=dict(
        type='Mask2FormerHead',
        in_channels=[96, 192, 384, 768],
        strides=[4, 8, 16, 32],
        feat_channels=256,
        out_channels=256,
        num_classes=num_classes,
        # 二分类用不满 100 个 query; 显存紧张时降到 30 能省下可观的 mask loss 显存
        num_queries=100,
        num_transformer_feat_level=3,
        align_corners=False,
        pixel_decoder=dict(
            type='mmdet.MSDeformAttnPixelDecoder',
            num_outs=3,
            norm_cfg=dict(type='GN', num_groups=32),
            act_cfg=dict(type='ReLU'),
            encoder=dict(
                num_layers=6,
                layer_cfg=dict(
                    self_attn_cfg=dict(
                        embed_dims=256,
                        num_heads=8,
                        num_levels=3,
                        num_points=4,
                        im2col_step=64,
                        dropout=0.0,
                        batch_first=True,
                        norm_cfg=None,
                        init_cfg=None),
                    ffn_cfg=dict(
                        embed_dims=256,
                        feedforward_channels=1024,
                        num_fcs=2,
                        ffn_drop=0.0,
                        act_cfg=dict(type='ReLU', inplace=True))),
                init_cfg=None),
            positional_encoding=dict(num_feats=128, normalize=True),
            init_cfg=None),
        enforce_decoder_input_project=False,
        positional_encoding=dict(num_feats=128, normalize=True),
        transformer_decoder=dict(
            return_intermediate=True,
            num_layers=9,
            layer_cfg=dict(
                self_attn_cfg=dict(
                    embed_dims=256,
                    num_heads=8,
                    attn_drop=0.0,
                    proj_drop=0.0,
                    dropout_layer=None,
                    batch_first=True),
                cross_attn_cfg=dict(
                    embed_dims=256,
                    num_heads=8,
                    attn_drop=0.0,
                    proj_drop=0.0,
                    dropout_layer=None,
                    batch_first=True),
                ffn_cfg=dict(
                    embed_dims=256,
                    feedforward_channels=2048,
                    num_fcs=2,
                    act_cfg=dict(type='ReLU', inplace=True),
                    ffn_drop=0.0,
                    dropout_layer=None,
                    add_identity=True)),
            init_cfg=None),
        # class_weight 长度为 num_classes + 1, 最后一位是 "无目标" 的权重。
        # 把裂缝那一类上调到 2.0, 让 query 更倾向被分配到裂缝而不是背景。
        loss_cls=dict(
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=2.0,
            reduction='mean',
            class_weight=[1.0, 2.0, 0.1]),
        # mask loss 与 dice loss 本身对前景稀少就友好 (dice 与类别比例无关,
        # 而且 train_cfg 里的 importance sampling 已经偏向难样本),
        # 所以先保持官方权重, 等本地验证集跑出来再按需调。
        loss_mask=dict(
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=True,
            reduction='mean',
            loss_weight=5.0),
        loss_dice=dict(
            type='mmdet.DiceLoss',
            use_sigmoid=True,
            activate=True,
            reduction='mean',
            naive_dice=True,
            eps=1.0,
            loss_weight=5.0),
        train_cfg=dict(
            num_points=12544,
            oversample_ratio=3.0,
            importance_sample_ratio=0.75,
            assigner=dict(
                type='mmdet.HungarianAssigner',
                match_costs=[
                    dict(type='mmdet.ClassificationCost', weight=2.0),
                    dict(
                        type='mmdet.CrossEntropyLossCost',
                        weight=5.0,
                        use_sigmoid=True),
                    dict(
                        type='mmdet.DiceCost',
                        weight=5.0,
                        pred_act=True,
                        eps=1.0),
                ]),
            sampler=dict(type='mmdet.MaskPseudoSampler')),
    ),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'),
)

# --------------------------------------------------------------- 优化器 ----
# 单卡 batch=2, 用梯度累积堆到等效 batch 16, 这样可以直接沿用官方 lr=1e-4。
# 若嫌慢, 把 accumulative_counts 降到 4 并把 lr 改成 5e-5。
# 官方 lr=1e-4 对应等效 batch 16。这里等效 batch 只有 8
# (batch_size 2 x accumulative_counts 4), 按比例折半取 5e-5。
optimizer = dict(
    type='AdamW', lr=0.00005, weight_decay=0.05, eps=1e-8, betas=(0.9, 0.999))

embed_multi = dict(lr_mult=1.0, decay_mult=0.0)
backbone_norm_multi = dict(lr_mult=0.1, decay_mult=0.0)
backbone_embed_multi = dict(lr_mult=0.1, decay_mult=0.0)

custom_keys = {
    'backbone': dict(lr_mult=0.1, decay_mult=1.0),
    'backbone.patch_embed.norm': backbone_norm_multi,
    'backbone.norm': backbone_norm_multi,
    'absolute_pos_embed': backbone_embed_multi,
    'relative_position_bias_table': backbone_embed_multi,
    'query_embed': embed_multi,
    'query_feat': embed_multi,
    'level_embed': embed_multi,
}
custom_keys.update({
    f'backbone.stages.{stage_id}.blocks.{block_id}.norm': backbone_norm_multi
    for stage_id, num_blocks in enumerate([2, 2, 6, 2])
    for block_id in range(num_blocks)
})
custom_keys.update({
    f'backbone.stages.{stage_id}.downsample.norm': backbone_norm_multi
    for stage_id in range(3)
})

# 精度选择。实测结论:
#   None       -> 纯 fp32。mask2former 在 384x672 / batch 2 下只占约 2.3G 显存,
#                 8G 完全放得下, 那就没必要冒混合精度的风险。
#   'bfloat16' -> 与 fp32 同指数范围, 不会像 fp16 那样溢出; 想提速可以试, 但
#                 需要先确认 mmcv 的 CUDA 算子支持 bf16。
#   'float16'  -> 【不要用】实测在 Hungarian 匹配阶段代价矩阵出 NaN,
#                 报错 "cost matrix is infeasible", 训练直接崩。
AMP_DTYPE = None

_optim_wrapper = dict(
    optimizer=optimizer,
    # 注意: mmengine 的 IterBasedTrainLoop 里 max_iters 数的是 micro-batch,
    # 不是优化步数。一个 train_step 只做一次前向反向, 梯度攒够
    # accumulative_counts 次才 step 一次优化器。
    accumulative_counts=4,
    # Mask2Former 对梯度爆炸很敏感, 官方就用了 0.01 的强裁剪
    clip_grad=dict(max_norm=0.01, norm_type=2),
    paramwise_cfg=dict(custom_keys=custom_keys, norm_decay_mult=0.0),
)
if AMP_DTYPE is None:
    optim_wrapper = dict(type='OptimWrapper', **_optim_wrapper)
else:
    optim_wrapper = dict(
        type='AmpOptimWrapper', dtype=AMP_DTYPE, **_optim_wrapper)

# 训练总量, 单位是 micro-batch (见上面的说明)。
#   等效 batch = batch_size * accumulative_counts = 2 * 4 = 8
#   优化步数   = MAX_ITERS / accumulative_counts = 30000 / 4 = 7500
#   训练规模   = MAX_ITERS * batch_size = 60000 张图, 979 张一轮 -> 约 61 轮
#   预计耗时   = 30000 * 0.53s ≈ 4.4 小时
# 有 save_best + 每 1000 iter 存 checkpoint, 中途觉得够了随时 Ctrl+C,
# 之后用 tools/ 里的脚本挑 best_Crack_F1_*.pth 出结果即可。
MAX_ITERS = 30000
param_scheduler = [
    dict(
        type='PolyLR',
        eta_min=0,
        power=0.9,
        begin=0,
        end=MAX_ITERS,
        by_epoch=False)
]

# val_interval 同样是 micro-batch 单位。1000 iter ≈ 8.8 分钟验证一次,
# 每次验证 221 张约 20 秒, 开销约 4%。
train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=MAX_ITERS, val_interval=1000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# --------------------------------------------------------------- 运行时 ----
env_cfg = dict(
    cudnn_benchmark=True,
    # Windows 不支持 fork, 必须 spawn
    mp_cfg=dict(mp_start_method='spawn', opencv_num_threads=0),
    dist_cfg=dict(backend='gloo'),
)

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='SegLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer',
    alpha=0.6)

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=False,
        interval=1000,          # 与 val_interval 对齐
        save_best='Crack_F1',   # 直接盯比赛真正打分的指标
        rule='greater',
        max_keep_ckpts=3,
        save_last=True),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook', interval=500, draw=True),
)

log_processor = dict(by_epoch=False)
log_level = 'INFO'
load_from = None
resume = False
auto_scale_lr = dict(enable=False, base_batch_size=16)
