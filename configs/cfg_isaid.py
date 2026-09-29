_base_ = './base_config.py'

# Existing MMSeg-converter-style 896px validation crops; crop-pooled metrics,
# not merged original-image scoring and not the official instance AP task.
model = dict(name_path='./configs/cls_isaid.txt',
             prob_thd=0.4, bg_idx=0)
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(448, 448), keep_ratio=True),
    dict(type='LoadAnnotations', reduce_zero_label=False, imdecode_backend='pillow'),
    dict(type='PackSegInputs'),
]
test_dataloader = dict(
    batch_size=1, num_workers=2, persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(type='iSAIDDataset', data_root='data/iSAID',
        reduce_zero_label=False, ignore_index=255, test_mode=True,
        data_prefix=dict(img_path='img_dir/val', seg_map_path='ann_dir/val'),
        pipeline=test_pipeline))
