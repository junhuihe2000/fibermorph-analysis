import streamlit as st
from PIL import Image
import numpy as np
import skimage

import matplotlib.pyplot as plt

from utils import segment_hair_fragments, remove_junction, calculate_curvature

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
    st.title("Hair Curvature Calculator")
    st.write("Upload an image of hair fragments, and the app will segment the hair fibers and calculate their curvature.")

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

    # Length range for hair fragments
    st.sidebar.header("Parameters in curvature calculation")
    resolution = st.sidebar.number_input("Resolution of the image (number of pixels per mm)", 1.0, 1000.0, 132.0, 1.0)
    min_size = st.sidebar.slider("Minimum pixel size of hair fragments", 10, 200, 50)

    # Segment the hair fragments
    skeleton, resolution = segment_hair_fragments(np.array(image), resolution=resolution, min_size=min_size)
    dilated = skimage.morphology.dilation(skeleton, np.ones((3, 3)))
    dilated_image = Image.fromarray(skimage.util.invert(dilated))
    st.subheader("Segmented Hair Fragments")
    st.image(dilated_image, use_container_width=True)

    # Remove junctions
    pruned = remove_junction(skeleton)

    # calculate curvature
    st.subheader("Curvature Analysis")
    curvatures, curv_summary = calculate_curvature(pruned, resolution)
    fig, ax = plt.subplots()
    ax.hist(curvatures, bins=64, color="steelblue", alpha=0.7)
    ax.axvline(curv_summary["mean_curvature"], color="red", linestyle="dashed", linewidth=1)
    ax.axvline(curv_summary["median_curvature"], color="green", linestyle="dotted", linewidth=1)
    stats_text = f'Mean Curv = {curv_summary["mean_curvature"]:.2f}\nMedian Curv = {curv_summary["median_curvature"]:.2f}\nLength = {curv_summary["total_length"]:.2f}'
    ax.text(0.95, 0.95, stats_text,
             horizontalalignment='right',
             verticalalignment='top',
             transform=plt.gca().transAxes,
             bbox=dict(facecolor='white', alpha=0.5))
    ax.set_title("Curvature Distribution")
    ax.set_xlabel("Curvature")
    ax.set_ylabel("Frequency")
    st.pyplot(fig)

    

if __name__ == "__main__":
    main()