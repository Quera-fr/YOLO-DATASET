import streamlit as st
from ultralytics import YOLO

st.set_page_config(layout="centered",
                   page_title="Fracture Detection",
                   page_icon="🦿")

st.title('Fracture Detection with YOLOv11')
st.image('https://storage.googleapis.com/kaggle-datasets-images/4296838/7391382/c6adfbf1646aa13b86de244fd486e2fc/dataset-cover.jpg?t=2024-01-12-18-15-46')

model = YOLO('runs/best.pt')

uploaded_img = st.file_uploader("Choose your file", type=["jpg", "jpeg", "png"])

if uploaded_img is not None:
    with open('tmp/' + uploaded_img.name, "wb") as f:
        f.write(uploaded_img.getbuffer())
    
    pred = model('tmp/' + uploaded_img.name)
    st.write("Prediction:")
    st.image(pred[0].plot())
    st.write(pred[0].boxes.data)