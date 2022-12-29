import streamlit as st
from polimorfo.datasets import CocoDataset
from polimorfo.utils import visualizeutils
from utils.visualization import convert_figure_to_image, format_visualization_input
from matplotlib.figure import Figure
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

ds: CocoDataset = st.session_state.ds
index_table = st.session_state.index_table

st.markdown("## View selection")

selection = st.selectbox("Selection method", ("Slider", "Single image", "Crete view"))

if selection == "Slider":
    num_images = len(st.session_state.ds)
    image_id = st.slider("Select an image:", 0, num_images - 1)
elif selection == "Single image":
    image_id = st.number_input("Image id:", 0)
else:
    cols = st.columns(2)
    view_features = cols[0].selectbox(
        "Filter criteria", ("Loss", "None")  # TODO: extend for other custom features
    )
    if view_features == "Loss":
        loss_range = cols[1].slider("Loss range", 0.0, 1.0, (0.0, 1.0))
        ranged_indexes = index_table[
            (index_table.loss >= min(loss_range))
            & (index_table.loss <= max(loss_range))
        ]
        image_idxs = ranged_indexes.image_id.unique()
        image_slider_idx = st.slider(
            "Filtered image", 0, len(image_idxs) - 1, label_visibility="hidden"
        )
        image_id = image_idxs[image_slider_idx]

        loss_df = pd.DataFrame(
            [(image_id, ds.imgs[image_id]["file_name"], ds.imgs[image_id]["loss"])],
            columns=["Image id", "Image name", "Loss"],
        )
        st.table(loss_df)
    else:
        image_id = 0


# load anns
anns = ds.get_annotations(img_idx=image_id)
# filter
with st.expander("Confidence thresholds"):
    thresholds = {}
    cols = st.columns(2)
    for idx, (cat_id, cat) in enumerate(ds.cats.items()):
        thresholds[cat_id] = cols[idx % 2].slider(cat["name"], 0.0, 1.0, 0.01)

filtered_anns = []
for ann in anns:
    if ann.get("score", 1.0) >= thresholds[ann["category_id"]]:
        filtered_anns.append(ann)

img = ds.load_image(image_id)
boxes, labels, scores, masks = format_visualization_input(img, filtered_anns)

# visualize
fig = Figure()
canvas = FigureCanvas(fig)
ax = fig.gca()

with st.expander("Visualization options"):
    show_boxes = st.checkbox("Show boxes", False)
    show_masks = st.checkbox("Show masks", True)
    color_border_only = st.checkbox("Color border only", False)
    line_width = st.number_input("Line width", 2)
    font_size = st.number_input("Font size", value=10, min_value=6)

visualizeutils.draw_instances(
    img,
    boxes,
    labels,
    scores,
    masks,
    st.session_state.idx_class_dict,
    ds.imgs[image_id]["file_name"],
    ax=ax,
    figsize=(18, 6),
    colors=st.session_state.colors,
    show_boxes=show_boxes,
    show_masks=show_masks,
    min_score=0.0,
    min_area=0.0,
    box_type=visualizeutils.BoxType.xywh,
    color_border_only=color_border_only,
    line_width=line_width,
    font_size=font_size,
)

img = convert_figure_to_image(fig)
st.image(img)
