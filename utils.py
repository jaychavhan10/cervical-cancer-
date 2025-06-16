import cv2
import numpy as np
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.measure import label, regionprops
from skimage.morphology import remove_small_objects

def extract_features(image_path):
    # Load and preprocess image
    image = imread(image_path)
    gray = rgb2gray(image)
    binary = gray > 0.5  # Adjust threshold as needed

    # Label image and extract regions
    labeled = label(binary)
    labeled = remove_small_objects(labeled, min_size=100)
    props = regionprops(labeled)

    if len(props) == 0:
        raise ValueError("No valid object detected")

    nucleus = props[0]  # first detected region

    # Nucleus features
    nucleus_area = nucleus.area
    nucleus_perimeter = nucleus.perimeter
    nucleus_major = nucleus.major_axis_length
    nucleus_minor = nucleus.minor_axis_length
    nucleus_elong = nucleus_major / nucleus_minor if nucleus_minor != 0 else 0
    nucleus_round = (4 * np.pi * nucleus_area) / (nucleus_perimeter ** 2) if nucleus_perimeter != 0 else 0

    cytoplasm = props[0]  # Use the largest region
    cytoplasm_area = cytoplasm.area
    cytoplasm_perimeter = cytoplasm.perimeter
    cytoplasm_major = cytoplasm.major_axis_length
    cytoplasm_minor = cytoplasm.minor_axis_length
    cytoplasm_elong = cytoplasm_major / cytoplasm_minor if cytoplasm_minor != 0 else 0
    cytoplasm_round = (4 * np.pi * cytoplasm_area) / (cytoplasm_perimeter ** 2) if cytoplasm_perimeter != 0 else 0

    # K/C Ratio
    kc_ratio = nucleus_area / cytoplasm_area if cytoplasm_area != 0 else 0

    return {
        "Nucleus/Cytoplasm Area Ratio": kc_ratio,
        "Nucleus Area": nucleus_area,
        "Nucleus Perimeter": nucleus_perimeter,
        "Nucleus Major Axis Length": nucleus_major,
        "Nucleus Minor Axis Length": nucleus_minor,
        "Nucleus Elongation": nucleus_elong,
        "Nucleus Roundness": nucleus_round,
        "Cytoplasm Area": cytoplasm_area,
        "Cytoplasm Perimeter": cytoplasm_perimeter,
        "Cytoplasm Major Axis Length": cytoplasm_major,
        "Cytoplasm Minor Axis Length": cytoplasm_minor,
        "Cytoplasm Elongation": cytoplasm_elong,
        "Cytoplasm Roundness": cytoplasm_round,
    }
