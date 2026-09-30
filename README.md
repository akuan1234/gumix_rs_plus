# GUMix-RS+

**Semantic Prototype Enrichment and Geometry-Uncertainty Routing for Training-Free Remote-Sensing Open-Vocabulary Segmentation**

GUMix-RS+ extends [GUMix-RS](https://github.com/akuan1234/gumix_rs) with grouped
remote-sensing prompts and canonical-to-alias prototype enrichment. GeoRSCLIP,
DINOv3-SAT and SAM2 are kept frozen.

## Installation

See [installation and pretrained weights](docs/INSTALL.md),
[dataset preparation and evaluation protocols](docs/DATASETS.md), and the
[evaluation guide](docs/EVALUATION.md).

## Evaluation

Run from the repository root:

```bash
CUDA_VISIBLE_DEVICES=0 python eval.py --config configs/cfg_udd5.py --max-samples 2 --work-dir work_dirs/udd5_smoke
```

This evaluates the first two UDD5 samples. Remove `--max-samples` for a full run
and use a separate output directory. Use `--data-root` and `--weights-dir` to
specify dataset and checkpoint locations, and `--out` to save label-ID PNGs.
Run one GPU per process with `batch_size=1`.

The default configuration uses
`prompt_type='gumix_rs_plus'`,
`sam2_points_per_side=8` and `multi_scales=[1.0, 1.5]`.
See [prompt construction](docs/METHOD.md) for details.

For Plus with a 32 by 32 SAM2 prompt grid:

```bash
CUDA_VISIBLE_DEVICES=0 python eval.py --config configs/cfg_udd5.py --cfg-options model.sam2_points_per_side=32 --work-dir work_dirs/udd5_plus32
```

The configs cover twelve multiclass and four binary evaluations, plus an extra
Massachusetts Building configuration. The multiclass tables use mIoU; the four
binary comparisons use foreground IoU. Dataset thresholds are shared across
the 8 by 8 and 32 by 32 grids. See the evaluation guide before reading results.

To evaluate the original ImageNet prompts with the same visual configuration:

```bash
CUDA_VISIBLE_DEVICES=0 python eval.py --config configs/cfg_openearthmap.py --cfg-options model.prompt_type=imagenet
```

## Acknowledgments and License

This project builds on GUMix-RS, OpenCLIP, GeoRSCLIP, DINOv3, SAM2 and
MMSegmentation. GUMix-RS+ project code uses the Apache-2.0 license. Bundled
third-party components retain their respective licenses; see
[third-party notices](THIRD_PARTY_NOTICES.md).
