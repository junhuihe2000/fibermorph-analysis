import streamlit as st
from PIL import Image
import numpy as np
import skimage

import matplotlib.pyplot as plt

from utils import segment_hair_fragments, remove_junction, calculate_curvature
from utils import crop_section, calculate_section

# ----------------------------------------
# Define (or replace) this with your own processing
def process_image(image: Image.Image):
    """
    Stub function. Replace this with your image-processing logic.
    For now, it just returns the original image.
    """
    return image
# ----------------------------------------

def main():
    st.title("Hair Cross-Sectional Calculator")
    st.write("Upload an image of hair cross-section, and the app will segment the hair sections and calculate their morphology.")

    uploaded_file = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg", "tif", "tiff"])
    if not uploaded_file:
        st.info("Awaiting image upload…")
        return

    # Load and display the original
    image = Image.open(uploaded_file)
    if image.mode != "L":
        image = image.convert("L")
    st.subheader("Original Image")
    st.image(image, use_container_width=True)

    # Segment the hair sections
    st.subheader("Segmented Hair Sections")
    crop = crop_section(np.array(image))
    section, seg = calculate_section(crop)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Cropped Hair Section")
        st.image(Image.fromarray(crop), use_container_width=True)
    with col2:
        st.subheader("Segmented Hair Section")
        st.image(Image.fromarray(seg), use_container_width=True)

    # calculate section morphology
    st.subheader("Section Analysis")
    st.write(f"Area: {section['area']:.2f} pixels")
    st.write(f"Eccentricity: {section['eccentricity']:.2f}")
    st.write(f"Major Axis Length: {section['major_axis_length']:.2f} pixels")
    st.write(f"Minor Axis Length: {section['minor_axis_length']:.2f} pixels")
    

if __name__ == "__main__":
    main()