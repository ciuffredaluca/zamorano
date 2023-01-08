import random
import streamlit as st
from utils import load_config
from pathlib import Path
from polimorfo.datasets import CocoDataset
from polimorfo.utils import visualizeutils
import psutil
import json

from utils.dataset import (
    create_index_table,
    check_gt_pd_compatibility,
    check_dataset_consistency,
)
from utils.metrics import create_iou_evaluation_table


def _load_dataset(dataset_path, images_path):
    ds = CocoDataset(dataset_path, images_path)
    check_dataset_consistency(ds)
    ds.update_images_path(lambda x: Path(x).name)
    ds.reindex()
    return ds


@st.experimental_memo
def convert_df(df):
    return df.to_csv(index=False).encode("utf-8")


# load configuration file
conf = load_config()
datasets_folder = Path(conf.datasets_folder)
datasets_list = [p.name for p in datasets_folder.iterdir() if p.is_dir()]

st.markdown("# Dataset selection")

dataset = st.selectbox(
    "Select dataset:",
    datasets_list,
)
dataset_path = datasets_folder / dataset

if not (dataset_path / "images").exists():
    st.warning("⚠️ **Achtung!** The selected dataset does not have an images folder.")

json_files = list(dataset_path.glob("*.json"))

with st.sidebar:
    predictions_only = st.checkbox("Predictions only", False)

if predictions_only:
    if len(json_files):
        annotations = st.selectbox(
            "Select annotations:",
            json_files,
        )
else:
    if len(json_files):
        annotations = st.selectbox(
            "Select prediction annotations:",
            json_files,
        )
        gt_annotations = st.selectbox(
            "Select ground truth annotations:",
            json_files,
        )

with st.sidebar:
    min_conf_load = st.checkbox("Load min confidences", False)

if min_conf_load:
    min_conf_file = st.selectbox(
        "Select min_conf_dict:",
        json_files,
    )
    with open(dataset_path / min_conf_file) as fp:
        min_conf_dict = json.load(fp)
else:
    min_conf_dict = None
st.session_state.default_min_conf_dict = min_conf_dict
st.session_state.min_conf_dict = min_conf_dict

if st.button("Load dataset"):

    ds = _load_dataset(dataset_path / annotations, dataset_path / "images")

    if not predictions_only:
        gt_ds = _load_dataset(dataset_path / gt_annotations, dataset_path / "images")
        # assert for same images, classes...
        assert check_gt_pd_compatibility(gt_ds, ds)
    else:
        gt_ds = None

    # cache dataset info
    st.session_state.ds = ds
    st.session_state.gt_ds = gt_ds
    st.session_state.idx_class_dict = {idx: cat["name"] for idx, cat in ds.cats.items()}
    st.session_state.colors = visualizeutils.generate_colormap(
        len(st.session_state.idx_class_dict) + 1
    )
    st.session_state.index_table = create_index_table(ds)
    st.session_state.iou_table = None

if not predictions_only:
    num_cpus = psutil.cpu_count(logical=False)
    num_cores_dft = num_cpus - 2
    num_cores = st.sidebar.number_input(
        "Number of cores:", value=num_cores_dft, min_value=1, max_value=num_cpus
    )
    if st.button("Pre-compute metrics"):
        if st.session_state.gt_ds is not None:
            st.session_state.iou_table = create_iou_evaluation_table(
                st.session_state.gt_ds, st.session_state.ds, num_cores
            )
        else:
            st.session_state.gt_ds = None
            st.warning("⚠️ **Achtung!** Ground truth dataset is not loaded.")

if st.session_state.get("iou_table") is not None:
    st.dataframe(st.session_state.iou_table)
    csv = convert_df(st.session_state.iou_table)
    st.download_button(
        "Download IoU table", csv, "iou_table.csv", "text/csv", key="download-csv"
    )
