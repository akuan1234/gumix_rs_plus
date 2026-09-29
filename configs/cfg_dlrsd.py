_base_ = './base_config.py'

# model settings
model = dict(
    name_path='./configs/cls_dlrsd.txt',
    prob_thd=0.25,
    bg_idx=255,
)

# dataset settings
dataset_type = 'DLRSDDataset'
data_root = 'data/DLRSD'

test_pipeline = [
    dict(type='LoadImageFromFile', imdecode_backend='pillow'),
    dict(type='LoadAnnotations', imdecode_backend='pillow'),
    dict(type='PackSegInputs')
]

test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        reduce_zero_label=False, ignore_index=255, test_mode=True,
        data_prefix=dict(
            img_path='img_dir/val',
            seg_map_path='ann_dir/val'),
        pipeline=test_pipeline))
