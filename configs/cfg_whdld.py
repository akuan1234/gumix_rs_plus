_base_ = './base_config.py'

# Six classes without background. Internal 255 denotes rejection.
# With MMSeg reduce_zero_label export: class PNG IDs are 1..6, rejection is 0.
model = dict(name_path='./configs/cls_whdld.txt',
             prob_thd=0.30, bg_idx=255)
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(448, 448), keep_ratio=True),
    dict(type='LoadAnnotations', reduce_zero_label=True, imdecode_backend='pillow'),
    dict(type='PackSegInputs'),
]
test_dataloader = dict(
    batch_size=1, num_workers=2, persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(type='BaseSegDataset', data_root='data/WHDLD',
        img_suffix='.jpg', seg_map_suffix='.png', reduce_zero_label=True,
        ignore_index=255, test_mode=True,
        metainfo=dict(classes=('building', 'road', 'pavement', 'vegetation', 'bare soil', 'water'),
                      palette=[[255,0,0], [255,255,0], [192,192,0], [0,255,0], [128,128,128], [0,0,255]]),
        data_prefix=dict(img_path='Images', seg_map_path='Labels'), pipeline=test_pipeline))
