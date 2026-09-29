"""Evaluate GUMix-RS+ with MMSegmentation."""
import argparse
import math
import os
from pathlib import Path

from mmengine.config import Config, DictAction
from mmengine.fileio import dump


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--work-dir')
    parser.add_argument('--data-root')
    parser.add_argument('--weights-dir')
    parser.add_argument('--out', help='Directory for predicted segmentation maps.')
    parser.add_argument('--max-samples', type=int, help='Evaluate the first N samples.')
    parser.add_argument('--cfg-options', nargs='+', action=DictAction)
    return parser.parse_args()


def main():
    args = parse_args()
    if int(os.environ.get('WORLD_SIZE', '1')) != 1:
        raise RuntimeError('This evaluation entry point supports one GPU per run.')
    cfg = Config.fromfile(args.config, import_custom_modules=False)
    if args.cfg_options:
        cfg.merge_from_dict(args.cfg_options)
    cfg.launcher = 'none'
    cfg.work_dir = args.work_dir or cfg.get('work_dir', str(Path('work_dirs') / Path(args.config).stem))
    cfg.model.device = 'cuda:0'
    if args.data_root:
        cfg.test_dataloader.dataset.data_root = args.data_root
    if args.weights_dir:
        directory = Path(args.weights_dir)
        cfg.model.georsclip_checkpoint = str(directory / 'RS5M_ViT-H-14.pt')
        cfg.model.dino_checkpoint = str(directory / 'dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth')
        cfg.model.sam2_checkpoint = str(directory / 'sam2_hiera_large.pt')
    if args.out:
        cfg.test_evaluator.output_dir = args.out
    if args.max_samples is not None:
        if args.max_samples < 1:
            raise ValueError('--max-samples must be positive.')
        cfg.test_dataloader.dataset.indices = args.max_samples
    if cfg.test_dataloader.get('batch_size', 1) != 1:
        raise ValueError('Evaluation requires batch_size=1.')
    for key in ('georsclip_checkpoint', 'dino_checkpoint', 'sam2_checkpoint', 'name_path'):
        path = cfg.model.get(key)
        if path and not Path(path).is_file():
            raise FileNotFoundError(f'{key}: {path}')

    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('Evaluation requires an NVIDIA CUDA GPU.')
    torch.cuda.set_device(0)
    import custom_datasets  # noqa: F401
    import custom_mass_transforms  # noqa: F401
    import gumix_rs_segmentor  # noqa: F401
    from mmengine.runner import Runner
    from mmengine.utils import import_modules_from_strings

    if cfg.get('custom_imports'):
        import_modules_from_strings(**cfg.custom_imports)
    runner = Runner.from_cfg(cfg)
    results = runner.test()
    metrics = {key: float(value) if math.isfinite(float(value)) else None
               for key, value in results.items()}
    dump(metrics, str(Path(runner.log_dir) / 'results.json'))


if __name__ == '__main__':
    main()
