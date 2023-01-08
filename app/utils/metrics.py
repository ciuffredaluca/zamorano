import itertools
import multiprocessing
from polimorfo.datasets import CocoDataset
from pathlib import Path
from polimorfo.utils import maskutils
import numpy as np
from typing import *
import pandas as pd
import psutil


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


def _compute_iou_per_image(x):

    img_id, gt_ds, pd_ds = x

    gt_img_meta = gt_ds.imgs[img_id]
    pd_img_meta = pd_ds.imgs[img_id]

    assert gt_img_meta["file_name"] == pd_img_meta["file_name"]
    h, w = gt_img_meta["height"], gt_img_meta["width"]

    gt_anns, pd_anns = gt_ds.get_annotations(img_id), pd_ds.get_annotations(img_id)

    all_class_ids = {l["category_id"] for l in gt_anns}.union(
        {l["category_id"] for l in pd_anns}
    )

    data = []

    for class_id in all_class_ids:
        gt_anns_class_id = list(filter(lambda x: x["category_id"] == class_id, gt_anns))
        pd_anns_class_id = list(filter(lambda x: x["category_id"] == class_id, pd_anns))
        for gt_meta, pd_meta in itertools.product(gt_anns_class_id, pd_anns_class_id):
            data.append(
                (
                    img_id,
                    gt_img_meta["file_name"],
                    gt_meta["category_id"],
                    pd_meta["category_id"],
                    pd_meta.get("score", 1.0),
                    compute_iou_from_polygons(
                        gt_meta["segmentation"], pd_meta["segmentation"], h, w
                    ),
                )
            )

    return data


def create_iou_evaluation_table(
    gt_ds: CocoDataset, pd_ds: CocoDataset, num_cores=1
) -> pd.DataFrame:

    items = [(idx, gt_ds, pd_ds) for idx in range(len(gt_ds))]

    with multiprocessing.Pool(num_cores) as pool:
        res = pool.map(_compute_iou_per_image, items)

    def __flatten(l):
        return [item for sublist in l for item in sublist]

    data = __flatten(res)

    return pd.DataFrame(
        data,
        columns=["image_id", "file_name", "gt_category_id", "pd_category_id", "score", "iou"],
    )
