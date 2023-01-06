import itertools
from polimorfo.datasets import CocoDataset
from pathlib import Path
from polimorfo.utils import maskutils
import numpy as np
from typing import *
import pandas as pd


def compute_iou_from_polygons(gt_polygons: List, pd_polygons: List, h: int, w: int):
    """Computes IoU for two intances defined by two set of polygons.

    Args:
        gt_polygons (List): polygons of a candidate ground truth instance.
        pd_polygons (List): polygons of a predicted instance.
        h (int): height of the image.
        w (int): width of the image.

    Returns:
        float: IoU of the two instances.
    """
    gt_mask = maskutils.polygons_to_mask(gt_polygons, h, w)
    pd_mask = maskutils.polygons_to_mask(pd_polygons, h, w)

    intersection = (gt_mask * pd_mask).sum()
    union = np.sign(gt_mask + pd_mask).sum()
    if union == 0:
        return 0.0
    return intersection / union


def create_iou_evaluation_table(gt_ds: CocoDataset, pd_ds: CocoDataset) -> pd.DataFrame:
    for img_id, gt_img_meta in gt_ds.imgs.items():
        pd_img_meta = pd_ds.imgs[img_id]
        assert gt_img_meta["file_name"] == pd_img_meta["file_name"]
        h, w = gt_img_meta["height"], gt_img_meta["width"]

        gt_anns, pd_anns = gt_ds.get_annotations(img_id), pd_ds.get_annotations(img_id)

        all_class_ids = {l["category_id"] for l in gt_anns}.union(
            {l["category_id"] for l in pd_anns}
        )

        data = []

        for class_id in all_class_ids:
            gt_anns_class_id = list(
                filter(lambda x: x["category_id"] == class_id, gt_anns)
            )
            pd_anns_class_id = list(
                filter(lambda x: x["category_id"] == class_id, pd_anns)
            )
            for gt_meta, pd_meta in itertools.product(
                gt_anns_class_id, pd_anns_class_id
            ):
                data.append(
                    (
                        img_id,
                        gt_img_meta["file_name"],
                        gt_meta["category_id"],
                        pd_meta["category_id"],
                        compute_iou_from_polygons(
                            gt_meta["segmentation"], pd_meta["segmentation"], h, w
                        ),
                    )
                )
    return pd.DataFrame(
        data,
        columns=["image_id", "file_name", "gt_category_id", "pd_category_id", "iou"],
    )
