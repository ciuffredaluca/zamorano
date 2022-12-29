import random
import streamlit as st
from utils import load_config
from pathlib import Path
from polimorfo.datasets import CocoDataset
from polimorfo.utils import visualizeutils

from utils.dataset import create_index_table

conf = load_config()
datasets_folder = Path(conf.datasets_folder)

datasets_list = [p.name for p in datasets_folder.iterdir() if p.is_dir()]

dataset = st.selectbox(
    "Select dataset:",
    datasets_list,
)
dataset_path = datasets_folder / dataset

if not (dataset_path / "images").exists():
    st.warning("⚠️ **Achtung!** The selected dataset does not have an images folder.")

json_files = list(dataset_path.glob("*.json"))
if len(json_files):
    annotations = st.selectbox(
        "Select annotations:",
        json_files,
    )

if st.button("Load dataset"):

    ds = CocoDataset(dataset_path / annotations, dataset_path / "images")
    ds.reindex()

    # TODO: temp for testing
    for ann in ds.anns.values():
        ann["score"] = random.random() * 0.35 + 0.65

    for img in ds.imgs.values():
        img["loss"] = random.random() * 0.65 + 0.35

    # cache dataset info
    st.session_state.ds = ds
    st.session_state.idx_class_dict = {idx: cat["name"] for idx, cat in ds.cats.items()}
    st.session_state.colors = visualizeutils.generate_colormap(
        len(st.session_state.idx_class_dict) + 1
    )
    st.session_state.index_table = create_index_table(ds)
