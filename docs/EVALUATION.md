# Evaluation

Use the weights in [INSTALL.md](INSTALL.md) and the prepared image/label layouts
in [DATASETS.md](DATASETS.md). Run from the repository root. Dataset thresholds,
class order and label transforms are part of the protocol, including when
comparing prompt grids or the ImageNet prompt baseline.

## Smoke and full runs

Start with the first two samples in the configured UDD5 dataset order:

```bash
CUDA_VISIBLE_DEVICES=0 python eval.py --config configs/cfg_udd5.py --max-samples 2 --work-dir work_dirs/udd5_smoke --out work_dirs/udd5_smoke/predictions
```

Check that both samples finish, metrics are finite and two label-ID PNGs are
written. A smoke run checks execution; its scores are not full-dataset scores.
Use a fresh work directory and prediction directory for each run. The evaluator
does not clear old prediction files, so reusing a directory can mix outputs from
different runs.

For the complete 40-image UDD5 evaluation:

```bash
set -o pipefail
mkdir -p work_dirs/udd5_plus8
CUDA_VISIBLE_DEVICES=0 python eval.py --config configs/cfg_udd5.py --work-dir work_dirs/udd5_plus8 --out work_dirs/udd5_plus8/predictions 2>&1 | tee work_dirs/udd5_plus8/console.log
```

The recorded full UDD5 run with the frozen Plus configuration produced mIoU
42.08, mAcc 52.15 and aAcc 74.11 (percent). This reference requires the same
40 image/label pairs, weights and protocol. A sample count alone does not establish
dataset identity. If the score differs, compare the resolved config, file
identities and environment before starting other evaluations.

Select another dataset with `--config configs/cfg_<dataset>.py`, using the
literal config names in DATASETS.md. `--data-root` replaces the dataset root;
the image/label subdirectories in the config still apply. `--weights-dir`
expects the three filenames listed in INSTALL.md.

Each evaluation process uses one visible CUDA GPU and requires `batch_size=1`.
`CUDA_VISIBLE_DEVICES` selects the physical GPU; the process addresses it as
`cuda:0`. The entry point rejects a distributed `WORLD_SIZE` greater than one.

## Model and grid variants

Use a separate work directory for each variant. All four combinations use the
same dataset config, weights, thresholds and two visual scales.

| Variant | `--cfg-options` arguments |
| --- | --- |
| Plus, 8 by 8 | None; this is the default |
| Plus, 32 by 32 | `model.sam2_points_per_side=32` |
| ImageNet prompt baseline, 8 by 8 | `model.prompt_type=imagenet` |
| ImageNet prompt baseline, 32 by 32 | `model.prompt_type=imagenet model.sam2_points_per_side=32` |

For example:

```bash
CUDA_VISIBLE_DEVICES=0 python eval.py --config configs/cfg_udd5.py --cfg-options model.sam2_points_per_side=32 --work-dir work_dirs/udd5_plus32 --out work_dirs/udd5_plus32/predictions
CUDA_VISIBLE_DEVICES=0 python eval.py --config configs/cfg_udd5.py --cfg-options model.prompt_type=imagenet model.sam2_points_per_side=32 --work-dir work_dirs/udd5_imagenet32 --out work_dirs/udd5_imagenet32/predictions
```

The ImageNet setting restores the conference text prompts with the same visual
path and current dataset protocol. A historical score obtained with a different
threshold or sample set is a different comparison. The grid parameter counts
points along each side, so 8 and 32 correspond to 64 and 1024 prompt points.

## Fixed inference settings

| Setting | Value |
| --- | --- |
| CLIP encoder | GeoRSCLIP ViT-H/14 |
| DINO encoder | DINOv3-SAT ViT-L/16; last four intermediate layers |
| Mask generator | Original SAM2 Hiera-L |
| SAM2 predicted-IoU / stability thresholds | 0.4 / 0.4 |
| SAM2 multimask output | False |
| Visual scales | 1.0, 1.5 |
| Sliding crop / stride | 336 / 112 |
| Final logit scale | 40 |
| Text strategy | `stage2_blendalias06_tiny_compound05_pool10` |
| Batch size | 1 |

Image resizing and class-specific label conventions are in DATASETS.md. The
dataset `prob_thd` applies to the maximum class probability after scaling logits
by 40 and applying softmax, before the region majority-label step. It is separate
from the two SAM2 proposal-quality thresholds above. Low-confidence labels use
the dataset's `bg_idx` and may be changed by the subsequent majority step.

## Outputs and metrics

MMEngine writes a timestamped run directory under `--work-dir`, with the log,
resolved configuration and `results.json`. `--out` contains predicted label-ID
PNGs, not coloured overlays. Preserve the resolved config as well as any CLI
overrides. The PNG label offsets for reduced-zero datasets are in DATASETS.md;
do not compare them to internal IDs without undoing that offset.

The twelve multiclass evaluations use dataset-level mIoU from accumulated pixel
counts. They do not average image-wise IoUs. RescueNet includes all eleven
classes, including background; its background-excluded variants are different
metrics. WHDLD and DLRSD predictions with reject ID 255 count as false negatives
on valid ground truth. Only invalid ground-truth pixels are omitted.

For the four binary datasets, read the foreground row's **IoU** in the final
`per class results` table in the MMEngine log:

| Dataset | Foreground internal ID | Log row |
| --- | ---: | --- |
| WHU Sat.I | 1 | `building` |
| WHU Sat.II | 1 | `building` |
| CHN6-CUG | 1 | `road` |
| Massachusetts Road | 1 | `road` |

The log IoU is already a percentage, rounded to two decimals. Copy that value
for the binary table. `results.json` contains the framework's summary metrics;
its two-class `mIoU` is not foreground IoU. The entry point does not write a
separate `foreground_IoU` field.

For an independent check, use exported predictions and the same prepared GT.
Undo any PNG offset, exclude GT ignore values, and accumulate intersections,
predicted-class counts and GT-class counts over the complete dataset:

```text
IoU_c = intersection_c / (prediction_count_c + gt_count_c - intersection_c)
mIoU_percent = 100 * mean(IoU_c over classes with nonzero union)
foreground_IoU_percent = 100 * IoU_1
```

Retain valid-GT pixels whose predictions are rejected when counting GT support.
Do not discard them because the prediction is 255 (or exported 0 for WHDLD).
When averaging results across datasets, give each dataset equal weight; do not
pool different datasets' class counts into a single confusion matrix.

## Reproduction records and limits

For each run retain the command, resolved config, console log, metrics,
predictions, package revision/hash, weight hashes, input image/label identities,
`python -m pip freeze` and `nvidia-smi` output. Confirm the expected sample count
and a successful process exit. For timing measurements, stop GPU keepalive
allocations and other work on the measured GPU; record warm-up and measurement
conditions separately.

The supplied protocols define the evaluation units, sample counts and prepared
label space.
Original conversion scripts and exact input file lists are not bundled. In
particular, the original iSAID tiling command is unavailable; the full WHDLD and
DLRSD collections should not be described as independently certified test
splits. Follow DATASETS.md for the known evaluation units and counts. Recreating
data from a raw download requires checking that it matches those prepared inputs.

The implementation retains the region-ID handling used in the recorded runs.
The first sorted region ID is treated as unsegmented in parts of the visual
path and skipped in the final majority-label step. If ID 0 is absent, the
smallest positive region is treated this way instead. The frequency and metric
impact of this boundary case have not been measured. Changes to it require a
separate regression comparison against the recorded implementation.

Use one query per output class as in the supplied class files. Arbitrary
semicolon-separated queries for later classes are not supported by the current
postprocessing aggregation. Optional model backends in the source are not part
of the validated default GeoRSCLIP + DINOv3-SAT + SAM2 configuration.
