# optimizer
optimizer = dict(
    type='AdamW',
    lr=0.0004,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    constructor='LayerDecayOptimizerConstructor',
    paramwise_cfg=dict(num_layers=12, layer_decay_rate=0.65))

optimizer_config = dict(
    type='DistOptimizerHook',
    update_interval=1,
    grad_clip=dict(max_norm=3, norm_type=2),
    coalesce=True,
    bucket_size_mb=-1,
    use_fp16=True)

# learning policy
lr_config = dict(
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-06,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)

# runtime settings
runner = dict(type='IterBasedRunner', max_iters=160000)
checkpoint_config = dict(by_epoch=False, interval=16000)
evaluation = dict(interval=16000, metric='mIoU', pre_eval=True)
evaluation = dict(interval=16000, metric='mIoU')
fp16 = None
work_dir = '/share/home/luoyang/MAE/Segmentation/output/'
gpu_ids = range(0, 1)