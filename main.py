#Libs
import cv2
from PIL import Image
import streamlit as st
from helper import detect_plate
from helper import plate_read



#Title
st.title("License Plate Reading")

#Header
st.header("Upload an image")
#files
file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
#Models
plate_detection_model = ("models/plate_detection.pt")
plate_read_model=("models/plate_read.pt")
#İmg
st.header("Original Image")
if file is not None:
    img = Image.open(file).convert("RGB")
    st.image(img)


#Plate
st.header("Result")
if file is not None:
    plate,detected_imgs=detect_plate(img,plate_detection_model)
    for i in range(len(detected_imgs)):
        st.image(detected_imgs[i],width=400)

    #Text
    st.header("Plate Text")
    plate_texts=plate_read(detected_imgs,plate_read_model)
    for plates_text in plate_texts:
        st.write(plates_text)