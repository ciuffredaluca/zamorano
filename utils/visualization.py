from matplotlib.figure import Figure
from polimorfo.utils import maskutils
from PIL import Image
import numpy as np
from typing import *


def convert_figure_to_image(fig: Figure) -> Image.Image:
    _ = fig.tight_layout(pad=0)
    _ = fig.canvas.draw()
    image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image_from_plot = image_from_plot.reshape(
        fig.canvas.get_width_height()[::-1] + (3,)
    )
    return Image.fromarray(image_from_plot)


def format_visualization_input(
    img: Image.Image,
    coco_annotations: List[Dict],
) -> Tuple[List]:

    boxes = []
    labels = []
    scores = []
    masks = []
    for ann in coco_annotations:
        boxes.append(ann["bbox"])
        labels.append(ann["category_id"])
        if "segmentation" in ann and ann["segmentation"] is not None:
            mask = maskutils.polygons_to_mask(
                ann["segmentation"], img.height, img.width
            )
            masks.append(mask)
        if "score" in ann:
            scores.append(float(ann["score"]))

    if not len(scores):
        scores = [1] * len(coco_annotations)

    if len(masks):
        masks = np.array(masks)
    else:
        masks = None
    return boxes, labels, scores, masks
