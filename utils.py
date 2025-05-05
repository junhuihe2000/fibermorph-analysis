
import numpy as np
import skimage
import scipy.ndimage
from PIL import Image

import cv2

from sklearn.manifold import Isomap
from scipy.interpolate import make_splrep


def remove_junction(ske_img, min_size: int = 30):
    """
    Remove junctions from the skeleton hair image.
    """
    
    conv_kernel = np.ones((3, 3), dtype=np.uint8)
    conv_image = scipy.ndimage.convolve(ske_img.astype(np.uint8), conv_kernel, mode="constant", cval=0.0)
    conv_image = np.where(ske_img, conv_image, 0)

    # 0 = background, 2 = edge, 3 = middle, 4,5 = junction
    edge_mark = ~np.isin(conv_image, [0, 3])
    conv_kernel2 = np.ones((5, 5), dtype=np.uint8)
    edge_mark = scipy.ndimage.convolve(edge_mark.astype(np.uint8), conv_kernel2, mode="constant", cval=0.0)
    edge_mark = np.where(edge_mark, 1, 0)

    prune_img = np.where(edge_mark, False, ske_img)
    prune_img = skimage.morphology.remove_small_objects(prune_img, min_size=min_size, connectivity=2)
    
    return prune_img

def segment_hair_fragments(image: np.ndarray, min_size: int = 50, width: int = 1800) -> np.ndarray:
    """
    Segments hair fibers from the input image using adaptive thresholding and morphological operations.
    
    Args:
        image (np.ndarray): Input grayscale image.
        min_size (int): Minimum size of the object to be segmented.
        width (int): Width of the image for resizing.

    Returns:
        np.ndarray: Segmented binary image with isolated hair fibers.
    """

    # Resize the image to a fixed width while maintaining aspect ratio
    height = int(image.shape[0] * (width / image.shape[1]))
    image = cv2.resize(image, (width, height))
    
    # apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    # apply adaptive thresholding
    adaptive_thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 9
    )

    # Dilate to connect fragmented edges
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_CLOSE, kernel, iterations=3)

    # skeletonize the image
    skeleton = skimage.morphology.skeletonize(closed)
    skeleton = skimage.morphology.remove_small_objects(skeleton, min_size=min_size, connectivity=2)
    
    return skeleton

# calculate the curvature of a single hair fragment
def cal_curv_single_hair(single_hair, resolution):
    """Calculate the mean curvature of a single hair fragment.

    Parameters
    ----------
    single_hair : Iterable
        A list of RegionProperties (most importantly, coordinates) from scikit-image regionprops function.
    resolution : int
        Number of pixels per mm.

    Returns
    -------
    float
        The mean curvature of the hair fragment.

    """
    
    # get the coordinates of the hair fragment
    coords = single_hair.coords

    # fit a principal curve to the coordinates

    # use Isomap to fit a curve
    # Isomap is a non-linear dimensionality reduction technique
    isomap = Isomap(n_neighbors=5, n_components=1)
    hair_isomap = isomap.fit_transform(coords).flatten()
    # sort the coordinates
    indices = np.argsort(hair_isomap)
    sorted_hair_isomap = hair_isomap[indices]
    sorted_coords = coords[indices, :]
    # fit a spline to the coordinates
    spl_x = make_splrep(sorted_hair_isomap, sorted_coords[:, 0], s=len(sorted_hair_isomap))
    spl_y = make_splrep(sorted_hair_isomap, sorted_coords[:, 1], s=len(sorted_hair_isomap))

    # calculate the first and second derivatives of the spline
    dx = spl_x.derivative(1)(sorted_hair_isomap)
    dy = spl_y.derivative(1)(sorted_hair_isomap)
    ddx = spl_x.derivative(2)(sorted_hair_isomap)
    ddy = spl_y.derivative(2)(sorted_hair_isomap)
    # calculate the curvature
    curvature = np.abs(dx * ddy - dy * ddx) / (dx ** 2 + dy ** 2) ** (3 / 2) * resolution

    return curvature

def calculate_curvature(prune: np.ndarray, resolution: float = 132.0):
    """
    Calculate the curvature of the hair fibers in the skeleton image.
    
    Args:
        prune (np.ndarray): Skeletonized binary image of hair fibers.

    Returns:
        np.ndarray: Curvature image.
    """
    
    # label the image
    label_img = skimage.measure.label(prune)
    # get the region properties
    morph = skimage.measure.regionprops(label_img)

    # calculate the mean curvature of each hair fragment
    curvatures = [cal_curv_single_hair(single_hair, resolution) for single_hair in morph]

    # calculate the weighted mean curvature of the entire image
    if len(morph) == 0:
        # if there are no hair fragments, return NaN
        length = 0
        mean_curvature = np.nan
        median_curvature = np.nan
    else:
        curvatures = np.concatenate(curvatures)
        length = np.sum([single_hair.area for single_hair in morph]) / resolution
        mean_curvature = np.mean(curvatures)
        median_curvature = np.median(curvatures)

    return curvatures, {"mean_curvature": mean_curvature, "median_curvature": median_curvature, "total_length": length}

def weighted_median(values, weights):
    values = np.array(values)
    weights = np.array(weights)

    # Sort by values
    sorted_indices = np.argsort(values)
    sorted_values = values[sorted_indices]
    sorted_weights = weights[sorted_indices]

    # Cumulative weights
    cumulative_weight = np.cumsum(sorted_weights)
    cutoff = 0.5 * np.sum(sorted_weights)

    # Find the first value where cumulative weight exceeds 50%
    median_idx = np.where(cumulative_weight >= cutoff)[0][0]
    return sorted_values[median_idx]


def crop_section(image: np.ndarray, min_size: int = 500, max_size: int = 20000, pad: int = 100, width: int = 1300) -> np.ndarray:
    """
    Crop the section of the image containing hair fragments.
    
    Args:
        image (np.ndarray): Input image.
        min_size (int): Minimum size of the object to be cropped.
        max_size (int): Maximum size of the object to be cropped.
        pad (int): Padding around the cropped section.
        width (int): Width of the image for resizing.

    Returns:
        np.ndarray: Cropped section of the image.
    """
    
    # Resize the image to a fixed width while maintaining aspect ratio
    height = int(image.shape[0] * (width / image.shape[1]))
    image = cv2.resize(image, (width, height))
    
    im_center = np.array(image.shape) // 2

    _, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    bin_img = skimage.segmentation.clear_border(thresh)
    # label the image
    label_im, _ = skimage.measure.label(bin_img, connectivity=2, return_num=True)
    # region properties
    props = skimage.measure.regionprops(label_image=label_im, intensity_image=image)
    props = [region for region in props if region.area > min_size and region.area < max_size]
    # calculate the distances between the centroids of the regions and the image center
    distances = [np.linalg.norm(region.centroid - im_center) for region in props]
    section = props[np.argmin(distances)]

    # crop the image to the selected region
    minr, minc, maxr, maxc = section.bbox
    crop_img = image[minr-pad:maxr+pad, minc-pad:maxc+pad]
    
    return crop_img

def calculate_section(crop_img: np.ndarray, min_size: int = 500, max_size: int = 20000):
    """
    Calculate the section of the hair fragments in the cropped image.
    
    Args:
        crop_img (np.ndarray): Cropped image of hair fragments.
        min_size (int): Minimum size of the object to be segmented.
        max_size (int): Maximum size of the object to be segmented.

    Returns:
        np.ndarray: Curvature image.
    """
    
    crop_im_center = np.array(crop_img.shape) // 2
    _, thresh = cv2.threshold(crop_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    seg_im = skimage.segmentation.morphological_chan_vese(crop_img, 40, init_level_set=thresh, smoothing=4)

    crop_label_im, _ = skimage.measure.label(seg_im, connectivity=2, return_num=True)

    crop_props = skimage.measure.regionprops(label_image=crop_label_im, intensity_image=crop_img)
    crop_props = [region for region in crop_props if region.area > min_size and region.area < max_size]
    # calculate the distances between the centroids of the regions and the image center
    distances = [np.linalg.norm(region.centroid - crop_im_center) for region in crop_props]
    section = crop_props[np.argmin(distances)]

    return section, seg_im > 0