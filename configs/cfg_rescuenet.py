_base_ = './base_config.py'

# Native IDs 0..10. Framework mIoU is explicitly the 11-class inclusive metric;
# background-excluded variants are separately named in independent recomputation.
model = dict(name_path='./configs/cls_rescuenet.txt',
             prob_thd=0.25, bg_idx=0)
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(448, 448), keep_ratio=True),
    dict(type='LoadAnnotations', reduce_zero_label=False, imdecode_backend='pillow'),
    dict(type='PackSegInputs'),
]
test_dataloader = dict(
    batch_size=1, num_workers=2, persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(type='RescueNetDataset', data_root='data/RescueNet',
        reduce_zero_label=False, ignore_index=255, test_mode=True,
        data_prefix=dict(img_path='test-org-img', seg_map_path='test-label-img'),
        pipeline=test_pipeline))
