import pandas as pd
from polimorfo.datasets import CocoDataset
from typing import *
from dataclasses import dataclass
import itertools


def create_index_table(ds: CocoDataset) -> pd.DataFrame:
    def _extract_anns_data(entry):
        ann_id, ann_meta = entry
        return (
            ann_meta["image_id"],
            ann_id,
            ann_meta["category_id"],
            ann_meta.get("score", 1.0),
        )

    def _extract_imgs_data(entry):
        img_id, img_meta = entry
        return (
            img_id,
            img_meta["file_name"],
            img_meta["height"],
            img_meta["width"],
            img_meta.get("loss", 0.0),
        )

    anns_data = list(map(_extract_anns_data, ds.anns.items()))
    imgs_data = list(map(_extract_imgs_data, ds.imgs.items()))

    anns_df = pd.DataFrame(
        anns_data, columns=["image_id", "ann_id", "category_id", "score"]
    )
    imgs_df = pd.DataFrame(
        imgs_data, columns=["image_id", "file_name", "height", "width", "loss"]
    )

    return imgs_df.merge(anns_df, on="image_id", how="inner")


def check_dataset_consistency(
    ds: CocoDataset,
    keys={
        "images": {
            "license",
            "file_name",
            "coco_url",
            "height",
            "width",
            "date_captured",
            "data_captured",  # typo in creating coco dataset
            "flickr_url",
            "id",
        },
        "anns": {
            "segmentation",
            "area",
            "iscrowd",
            "image_id",
            "bbox",
            "category_id",
            "id",
            "polygons",
            "attributes",
            "score",
        },
    },
):
    for _, img_meta in ds.imgs.items():
        if not set(img_meta.keys()) <= keys["images"]:
            raise ValueError(
                f"Images keys in dataset are not consistent with default values {keys['images']}"
            )

    for ann_id, ann_meta in ds.anns.items():
        if not set(ann_meta.keys()) <= keys["anns"]:
            raise ValueError(
                f"Annotation keys in dataset are not consistent with default values {keys['anns']}. Got: {set(ann_meta.keys())}"
            )
        for polygons_field in ["polygons", "segmentation"]:
            if polygons_field in ann_meta and not isinstance(
                ann_meta[polygons_field], list
            ):
                raise ValueError(
                    f"Segmentation field provided at annotation id {ann_id} is not a list of polygons."
                )


# TODO: check if same numebr of images, ...
def check_gt_pd_compatibility(gt_ds: CocoDataset, pd_ds: CocoDataset) -> bool:
    return True
