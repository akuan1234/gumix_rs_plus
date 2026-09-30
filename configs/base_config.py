# base configurations
model = dict(
    type='gumix_rs',
    clip_type='laion2b_s32b_b79k',
    model_type='ViT-H-14',
    dino_type='dinov3_sat',
    mask_generator='sam2',
    multi_scales=[1.0, 1.5],
    use_georsclip=True,  # GeoRSCLIP
    sam2_points_per_side=8,
    # Journal strategy: fixed prompt groups and canonical-to-alias blending.
    # Use --cfg-options model.prompt_type=imagenet to reproduce the conference prompt baseline.
    prompt_type='gumix_rs_plus',
    georsclip_checkpoint='checkpoints/RS5M_ViT-H-14.pt',
    dino_checkpoint='checkpoints/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth',
    sam2_checkpoint='checkpoints/sam2_hiera_large.pt',
)


test_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])

default_scope = 'mmseg'
env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='SegLocalVisualizer', vis_backends=vis_backends, alpha=1.0, name='visualizer')
log_processor = dict(by_epoch=False)
log_level = 'INFO'
load_from = None
resume = False

test_cfg = dict(type='TestLoop')

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=5, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=2000),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook', interval=5))
