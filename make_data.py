import sys, os

# ________________ HANDLE THE PATH THING ________________ #
# get the absolute path of the script's directory
script_path = os.path.dirname(os.path.abspath(__file__))
# get the parent directory of the script's directory
parent_path = os.path.dirname(script_path)
sys.path.append(parent_path)
from utils_data import *
from utils_model import *


base_path_to_input = os.path.join(script_path, "resource", "data_org", "data")
base_path_to_output = os.path.join(
    script_path,
    "input",
    "data",
)


# ________________ MAKE DATA READY ________________ #
def make_data():
    # Make train dataset
    convert_to_yolo_data(
        path_to_input_images=os.path.join(base_path_to_input, "train", "train_images"),
        path_to_input_JSON=os.path.join(
            base_path_to_input, "train", "train_images", "train.json"
        ),
        path_to_output_images=os.path.join(base_path_to_output, "train", "images"),
        path_to_output_labels=os.path.join(base_path_to_output, "train", "labels"),
    )

    # Make test dataset
    convert_to_yolo_data(
        path_to_input_images=os.path.join(base_path_to_input, "test", "test_images"),
        path_to_input_JSON=os.path.join(
            base_path_to_input, "test", "test_images", "test.json"
        ),
        path_to_output_images=os.path.join(base_path_to_output, "test", "images"),
        path_to_output_labels=os.path.join(base_path_to_output, "test", "labels"),
    )

    # Make the YAML configuration file
    make_yaml(
        path_to_input_JSON=os.path.join(
            base_path_to_input, "train", "train_images", "train.json"
        ),
        path_to_output_yaml=os.path.join(base_path_to_output, "data.yaml"),
        path_to_train=os.path.join(base_path_to_output, "train", "images"),
        path_to_test=os.path.join(base_path_to_output, "test", "images"),
        path_to_val=None,
    )


if __name__ == "__main__":
    make_data()
