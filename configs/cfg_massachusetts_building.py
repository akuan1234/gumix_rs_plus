_base_ = './base_config.py'

# model settings
model = dict(
    name_path='./configs/cls_massachusetts_building.txt',
    prob_thd=0.15,
)

custom_imports = dict(imports=['custom_mass_transforms'], allow_failed_imports=False)

# dataset settings
dataset_type = 'BaseSegDataset'
data_root = 'data/mass_building'

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='BinarizeMassachusettsBuilding'),
    dict(type='PackSegInputs')
]

test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        metainfo=dict(classes=('background', 'building'), palette=[[0, 0, 0], [255, 255, 255]]),
        data_root=data_root,
        img_suffix='.tiff',
        seg_map_suffix='.tif',
        data_prefix=dict(
            img_path='images',
            seg_map_path='label'),
        pipeline=test_pipeline))
