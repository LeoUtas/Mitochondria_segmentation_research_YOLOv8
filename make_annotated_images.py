import sys, os

# ________________ HANDLE THE PATH THING ________________ #
# get the absolute path of the script's directory
script_path = os.path.dirname(os.path.abspath(__file__))
# get the parent directory of the script's directory
parent_path = os.path.dirname(script_path)
sys.path.append(parent_path)
from utils_data import *
from utils_model import *
from logger import *


# ________________ MAKE ANNOTATED IMAGES FOR TRAIN ________________ #
path_to_images = os.path.join(script_path, "input", "data", "train", "images")
path_to_annotations = os.path.join(script_path, "input", "data", "train", "labels")
path_to_save_annotated_images = os.path.join(
    script_path, "input", "data", "train", "annotated_images"
)

make_annotated_images(
    path_to_images, path_to_annotations, path_to_save_annotated_images
)


# ________________ MAKE ANNOTATED IMAGES FOR TEST ________________ #
path_to_images = os.path.join(script_path, "input", "data", "test", "images")
path_to_annotations = os.path.join(script_path, "input", "data", "test", "labels")
path_to_save_annotated_images = os.path.join(
    script_path, "input", "data", "test", "annotated_images"
)

make_annotated_images(
    path_to_images, path_to_annotations, path_to_save_annotated_images
)
