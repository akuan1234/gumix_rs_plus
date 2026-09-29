_base_ = './base_config.py'

# model settings
model = dict(
    name_path='./configs/cls_floodnet.txt',
    prob_thd=0.3,
)

# dataset settings
# This protocol uses ten classes with native IDs 0..9, including background 0.
# Label 255 is ignored; labels are not reduced by one.
dataset_type = 'FloodNetDataset'
data_root = 'data/FlodNet'

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(448, 448), keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]

test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        reduce_zero_label=False,
        data_prefix=dict(
            img_path='val/val-org-img',
            seg_map_path='val/val-label-img'),
        pipeline=test_pipeline))
