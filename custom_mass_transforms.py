import numpy as np
from mmcv.transforms import BaseTransform

from mmseg.registry import TRANSFORMS


@TRANSFORMS.register_module()
class BinarizeMassachusettsBuilding(BaseTransform):
    """Convert Massachusetts building masks to binary labels.

    The raw masks are stored as background/building intensity maps. MMSeg
    metrics expect class ids, so all non-zero pixels are mapped to class 1.
    """

    def transform(self, results):
        gt_seg_map = results.get('gt_seg_map', None)
        if gt_seg_map is not None:
            if gt_seg_map.ndim == 3:
                gt_seg_map = np.any(gt_seg_map > 0, axis=-1)
            else:
                gt_seg_map = gt_seg_map > 0
            results['gt_seg_map'] = gt_seg_map.astype(np.uint8)
        return results
