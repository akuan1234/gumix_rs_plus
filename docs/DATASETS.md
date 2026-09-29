# Dataset preparation

The journal evaluation uses 12 multiclass and four binary datasets. The release
also includes a Massachusetts Building configuration. See
[evaluation](EVALUATION.md) for commands, inference settings and metric extraction.

## Directory layout

Image and label paths below are relative to `data_root`. Pass `--data-root` to
use another location. Config stems refer to files under `configs/`.

| Config stem | Data root | Images | Labels |
| --- | --- | --- | --- |
| `cfg_openearthmap` | `data/OpenEarthMap/OpenEarthMap` | `img_dir/val` | `ann_dir/val` |
| `cfg_floodnet` | `data/FlodNet` | `val/val-org-img` | `val/val-label-img` |
| `cfg_vaihingen` | `data/vaihingen` | `img_dir/val` | `ann_dir/val` |
| `cfg_udd5` | `data/UDD5` | `val/src` | `val/gt_label` |
| `cfg_vdd` | `data/VDD` | `test/src` | `test/gt` |
| `cfg_loveda` | `data/LoveDA` | `img_dir/val` | `ann_dir/val` |
| `cfg_potsdam` | `data/potsdam` | `img_dir/val` | `ann_dir/val` |
| `cfg_uavid` | `data/UAVid_cvt` | `img_dir/test` | `ann_dir/test` |
| `cfg_isaid` | `data/iSAID` | `img_dir/val` | `ann_dir/val` |
| `cfg_whdld` | `data/WHDLD` | `Images` | `Labels` |
| `cfg_rescuenet` | `data/RescueNet` | `test-org-img` | `test-label-img` |
| `cfg_dlrsd` | `data/DLRSD` | `img_dir/val` | `ann_dir/val` |
| `cfg_whu_sat_I` | `data/whu-Satellite` | `image` | `label_cvt` |
| `cfg_whu_sat_II` | `data/whu-Satellite2` | `test/image` | `test/label_cvt` |
| `cfg_chn6-cug` | `data/CHN6-CUG` | `CHN6-CUG/image_cvt` | `CHN6-CUG/label_cvt` |
| `cfg_massachusetts_road` | `data/mass_roads` | `images` | `label_cvt` |
| `cfg_massachusetts_building` | `data/mass_building` | `images` | `label` |

`FlodNet` and the repeated `CHN6-CUG` directory are literal config paths.
Directories named `label_cvt` require preconverted integer class-ID masks.
Datasets and preprocessing scripts are not bundled. Preserve the configured
image and label suffixes; FloodNet and RescueNet labels use `_lab.png`.

## Class order and evaluation protocol

Each linked query file specifies the class order: line 1 maps to internal class
0, line 2 to class 1, and so on. Keep the file contents and order unchanged.
Dataset metadata may use a different display name for the same index; for
example, UDD5 class 4 is named `other` in metadata and queried as `background`.

`N` and `R` below define the label mapping. `bg_idx` is the internal label used
for a low-confidence prediction; it can be a class index or the rejection value
255. Thresholds are the fixed settings used for both SAM2 grids. Their historical
selection used evaluation feedback.

| Dataset | Samples | Classes and ordered queries | Mapping | `prob_thd` | `bg_idx` |
| --- | ---: | --- | :---: | ---: | ---: |
| OpenEarthMap | 500 | [9](../configs/cls_openearthmap.txt) | N | 0.10 | 0 |
| Vaihingen | 398 | [6](../configs/cls_vaihingen.txt) | R | 0.10 | 5 |
| Potsdam | 2016 | [6](../configs/cls_potsdam.txt) | R | 0.10 | 5 |
| FloodNet | 450 | [10](../configs/cls_floodnet.txt) | N | 0.30 | 0 |
| LoveDA | 1669 | [7](../configs/cls_loveda.txt) | R | 0.30 | 0 |
| UAVid | 1020 | [7](../configs/cls_uavid.txt) | N | 0.30 | 0 |
| UDD5 | 40 | [5](../configs/cls_udd5.txt) | N | 0.30 | 4 |
| VDD | 40 | [7](../configs/cls_vdd.txt) | N | 0.40 | 0 |
| iSAID | 11644 | [16](../configs/cls_isaid.txt) | N | 0.40 | 0 |
| WHDLD | 4940 | [6](../configs/cls_whdld.txt) | R | 0.30 | 255 |
| RescueNet | 450 | [11](../configs/cls_rescuenet.txt) | N | 0.25 | 0 |
| DLRSD | 2100 | [17](../configs/cls_dlrsd.txt) | N | 0.25 | 255 |
| WHU Satellite I | 204 | [2](../configs/cls_whu.txt) | N | 0.70 | 0 |
| WHU Satellite II | 3726 | [2](../configs/cls_whu.txt) | N | 0.70 | 0 |
| CHN6-CUG | 903 | [2](../configs/cls_chn6-cug.txt) | N | 0.80 | 0 |
| Massachusetts Road | 49 | [2](../configs/cls_roadval.txt) | N | 0.80 | 0 |

For a dataset with `K` classes, the mappings are:

| Mapping | Valid raw GT IDs | Ignored raw GT IDs | `reduce_zero_label` | Raw GT to internal ID | Internal class to saved PNG ID |
| --- | --- | --- | --- | --- | --- |
| N | `0..K-1` | `255` | `False` | unchanged | unchanged |
| R | `1..K` | `0`, `255` | `True` | subtract 1 | add 1 |

Both mappings use internal GT `ignore_index=255`. In N datasets, zero is a valid
class, including background where present. For R datasets, distinguish raw zero
from internal class zero. The PNG offset is applied by the MMSegmentation
exporter; compare saved predictions with raw GT using the corresponding mapping.

WHDLD and DLRSD have no background class. Rejected predictions use internal 255
and count as false negatives on valid GT. WHDLD PNG export adds one and casts to
`uint8`, so its saved class IDs are `1..6` and rejection is `0`. DLRSD retains
class IDs `0..16` and rejection `255`. Ignore pixels according to GT only;
discarding a valid GT pixel because its prediction is rejected changes the metric.

The multiclass metric averages IoU over every listed class, including background
where present. iSAID uses 16 classes and RescueNet uses 11. RescueNet variants
that exclude background or zero its confusion-matrix row and column are separate
metrics. The four binary datasets use class 0 for background and class 1 for
building or road; their primary result is class-1 foreground IoU. See
[evaluation](EVALUATION.md) for extracting it. Two-class mIoU averages the
background and foreground IoUs and must retain its own label.

All supplied query files contain one query per class. Commas remain inside that
query: the alias-enabled prompt builder retains the full query as the canonical
expression, adds fixed aliases and comma-separated parts, removes duplicates,
and keeps at most four expressions including the canonical one. A comma does
not create another output class. The parser also recognizes `; `, but the
downstream multi-query collapse assumes extra queries belong to the first
class. Arbitrary `; ` groups are unsupported; preserve the supplied one-query
format for these evaluations.

## Resolution and collection identity

DLRSD uses its native 256 by 256 images. The other 15 main configurations resize
images to fit within 448 by 448 pixels while preserving aspect ratio. Annotations
retain their original resolution, and final predictions are aligned to GT for
evaluation.

The sample counts above describe the evaluated collections. Directory names
such as `val` and `test` alone do not establish an official split. In particular:

- iSAID uses 11644 existing 896 by 896 validation crops from 458 source-image
  identities. Metrics pool pixels across crops, counting overlapping pixels in
  each crop; predictions are not merged into original images. The historical
  conversion command has not been recovered.
- WHDLD uses all 4940 available pairs; DLRSD uses all 2100 available pairs under
  the listed validation directories. Neither collection has a certified official
  held-out test split in the recovered evaluation material.
- RescueNet uses a 450-pair extraction from the listed test directories. Its
  label order was checked against the official loader; the source archive
  identity has not been independently certified.
- OpenEarthMap uses 500 samples and Potsdam uses 2016. Preserve these collection
  sizes and label conventions when comparing with the reported runs.

Sample counts alone cannot establish identical inputs. Check image identities,
label conversions and file contents when preparing another copy.

## Additional Massachusetts Building configuration

`cfg_massachusetts_building.py` is outside the 16-dataset protocol above. It uses
native-resolution `.tiff` images and `.tif` binary intensity masks, with
[background then building](../configs/cls_massachusetts_building.txt).
`BinarizeMassachusettsBuilding` maps zero to background and any nonzero value to
building; for multi-channel masks, any nonzero channel marks building. Masks
containing 255-valued ignore regions are incompatible with this transform.
