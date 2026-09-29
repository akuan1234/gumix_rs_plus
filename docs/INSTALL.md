# Installation

The evaluation pipeline targets Linux with an NVIDIA CUDA GPU and Python 3.10.

```bash
conda create -n gumix-rs-plus python=3.10 -y
conda activate gumix-rs-plus
python -m pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121
python -m pip install "numpy<2" openmim
mim install "mmcv==2.1.0"
python -m pip install -r requirements.txt
python -m pip check
```

MMCV must match the installed PyTorch and CUDA versions. See its
[installation guide](https://mmcv.readthedocs.io/en/latest/get_started/installation.html)
if a compatible wheel is not found. Use the bundled, modified `open_clip` and
backbone modules; do not replace them with stock installations.

## Pretrained weights

Place the following files under `checkpoints/`, or pass `--weights-dir` to `eval.py`:

| File | Source |
| --- | --- |
| `RS5M_ViT-H-14.pt` | [GeoRSCLIP](https://huggingface.co/Zilun/GeoRSCLIP), `ckpt/RS5M_ViT-H-14.pt` |
| `dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth` | [DINOv3](https://github.com/facebookresearch/dinov3), ViT-L/16 SAT-493M |
| `sam2_hiera_large.pt` | [SAM2](https://github.com/facebookresearch/sam2), original Hiera Large |

Weights are not included. Follow the providers' access and license requirements.
SAM2.1 checkpoints are not interchangeable with the original SAM2 configuration.

This release targets the default GeoRSCLIP + DINOv3-SAT + SAM2 pipeline.

Check the files against the hashes of the weights used in the reported runs:

| File | SHA256 |
| --- | --- |
| `RS5M_ViT-H-14.pt` | `62a3e79df886875976901b4a9b8d6a1c42b457879a5ce625c8eba9460cbc3a11` |
| `dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth` | `eadcf0ffc02418b6c22a885ea1a7aaeeef84fbf0f5bb4d0b7d1d36e68a964f48` |
| `sam2_hiera_large.pt` | `7442e4e9b732a508f80e141e7c2913437a3610ee0c77381a66658c3a445df87b` |

```bash
sha256sum checkpoints/RS5M_ViT-H-14.pt checkpoints/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth checkpoints/sam2_hiera_large.pt
```

The GeoRSCLIP loader prints missing and unexpected state-dict keys without
stopping automatically. The recorded checkpoint loads with all keys matched.
Investigate a different loading report or hash before comparing results.

## Environment checks

The recorded A40 evaluation stack used PyTorch 2.1.2+cu121, torchvision
0.16.2+cu121, MMCV 2.1.0, MMEngine 0.10.7, MMSegmentation 1.2.2 and NumPy 1.26.4.
`requirements.txt` includes version ranges for other dependencies; it is not a
full environment lock. A fresh installation must pass the checks below and the
UDD5 smoke run in the [evaluation guide](EVALUATION.md) before a full evaluation.

```bash
python -m pip check
python -c "import torch, torchvision, mmcv, mmengine, mmseg; print(torch.__version__, torchvision.__version__, mmcv.__version__, mmengine.__version__, mmseg.__version__); assert torch.cuda.is_available()"
python -c "import custom_datasets, custom_mass_transforms, gumix_rs_segmentor; from gumix_rs_segmentor import _SAM2_IMPORT_ERROR; assert _SAM2_IMPORT_ERROR is None, repr(_SAM2_IMPORT_ERROR)"
```

Run these commands from the repository root. Import success checks the local
modules and installed packages; the smoke run also checks weights, data loading
and GPU inference. Save `python -m pip freeze` and `nvidia-smi` output alongside
each experiment. Optional mask backends and other backbone variants in the
source are outside the default pipeline's evaluation protocol.
