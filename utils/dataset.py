import pandas as pd
from polimorfo.datasets import CocoDataset


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
