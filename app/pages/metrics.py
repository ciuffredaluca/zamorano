import streamlit as st

if st.session_state.iou_table is not None:
    print(st.session_state.iou_table.head())
