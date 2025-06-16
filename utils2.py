import numpy as np
import cv2
from PIL import Image
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.measure import label, regionprops
from skimage.morphology import remove_small_objects, closing, square

def extract_features(image_path):
    # === Load Image ===
    image = Image.open(image_path).convert("RGB")
    img_rgb_array = np.array(image)

    # === Color-Based Segmentation ===
    nucleus_color = np.array([0, 0, 255])      # Blue nucleus
    cytoplasm_color = np.array([0, 0, 128])    # Dark blue cytoplasm

    nucleus_mask = np.all(np.abs(img_rgb_array - nucleus_color) < 50, axis=-1)
    cytoplasm_mask = np.all(np.abs(img_rgb_array - cytoplasm_color) < 50, axis=-1)

    nucleus_mask = closing(nucleus_mask, square(3))
    cytoplasm_mask = closing(cytoplasm_mask, square(3))

    # Label regions
    nucleus_label = label(nucleus_mask)
    cytoplasm_label = label(cytoplasm_mask)

    nucleus_props = regionprops(nucleus_label)
    cytoplasm_props = regionprops(cytoplasm_label)

    if len(nucleus_props) == 0 or len(cytoplasm_props) == 0:
        raise ValueError("Nucleus or cytoplasm not detected properly.")

    nucleus = nucleus_props[0]
    cytoplasm = cytoplasm_props[0]

    # === Grayscale-Based Binary (for optional use) ===
    image = imread(image_path)
    gray = rgb2gray(image)
    binary = gray > 0.5
    labeled = label(binary)
    labeled = remove_small_objects(labeled, min_size=100)
    props = regionprops(labeled)

    

    # === Feature Extraction ===
    nucleus_area = nucleus.area
    nucleus_perimeter = nucleus.perimeter
    nucleus_major = nucleus.major_axis_length
    nucleus_minor = nucleus.minor_axis_length
    nucleus_elong = nucleus_major / nucleus_minor if nucleus_minor != 0 else 0
    nucleus_round = (4 * np.pi * nucleus_area) / (nucleus_perimeter ** 2) if nucleus_perimeter != 0 else 0

    cytoplasm_area = cytoplasm.area
    cytoplasm_perimeter = cytoplasm.perimeter
    cytoplasm_major = cytoplasm.major_axis_length
    cytoplasm_minor = cytoplasm.minor_axis_length
    cytoplasm_elong = cytoplasm_major / cytoplasm_minor if cytoplasm_minor != 0 else 0
    cytoplasm_round = (4 * np.pi * cytoplasm_area) / (cytoplasm_perimeter ** 2) if cytoplasm_perimeter != 0 else 0

    kc_ratio = nucleus_area / cytoplasm_area if cytoplasm_area != 0 else 0

    features = {
        "Nucleus Area": nucleus_area,                # Kerne_A
        "Cytoplasm Area": cytoplasm_area,            # Cyto_A
        "Nucleus/Cytoplasm Area Ratio": kc_ratio,    # K/C
        "Nucleus Minor Axis Length": nucleus_minor,  # KerneShort
        "Cytoplasm Minor Axis Length": cytoplasm_minor, # CytoShort
        "Nucleus Major Axis Length": nucleus_major,  # KerneLong
        "Cytoplasm Major Axis Length": cytoplasm_major, # CytoLong
        "Nucleus Elongation": nucleus_elong,         # KerneElong
        "Nucleus Roundness": nucleus_round,          # KerneRund
        "Cytoplasm Roundness": cytoplasm_round,      # CytoRund
        "Cytoplasm Elongation": cytoplasm_elong,     # CytoElong
        "Nucleus Perimeter": nucleus_perimeter,      # KernePeri
        "Cytoplasm Perimeter": cytoplasm_perimeter   # CytoPeri
    }

    return features, nucleus_mask.astype(np.uint8), cytoplasm_mask.astype(np.uint8)
