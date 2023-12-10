import sys, os
from time import time

# ________________ HANDLE THE PATH THING ________________ #
# get the absolute path of the script's directory
script_path = os.path.dirname(os.path.abspath(__file__))
# get the parent directory of the script's directory
parent_path = os.path.dirname(script_path)
sys.path.append(parent_path)

from utils_data import *
from utils_model import *

start_time = time()


# ________________ MAKE MODEL CONFIGURATION ________________ #
# ****** --------- ****** #
test_name = "test2"
note = ""
# ****** --------- ****** #


# model_yaml = "yolov8n-seg.yaml"
# model_pt = "yolov8n-seg.pt"

model_yaml = "yolov8x-seg.yaml"
model_pt = "yolov8x-seg.pt"

data = os.path.join(parent_path, "input", "data", "data.yaml")
project = os.path.join(parent_path, "models", test_name)
path_to_val_model = os.path.join(
    parent_path, "models", test_name, "train", "weights", "best.pt"
)
# custom configuration
epochs = 100
patience = 0
batch = 16
image_size = 1024
device = 0
workers = 8
pretrained = True
optimizer = "auto"
verbose = True
lr0 = 0.01
weight_decay = 0.0005


model = YOLOSegmentation(
    test_name=test_name,
    model_yaml=model_yaml,
    model_pt=model_pt,
    data=data,
    project=project,
    note=note,
    # custom configuration
    epochs=epochs,
    patience=patience,
    batch=batch,
    image_size=image_size,
    device=device,
    workers=workers,
    pretrained=pretrained,
    optimizer=optimizer,
    verbose=verbose,
    lr0=lr0,
    weight_decay=weight_decay,
)

# _ TRAIN _ #
model.train()

# # ________________ VALIDATE ________________ #
# model.validate(path_to_val_model=path_to_val_model)

execution_time = round((time() - start_time) / 60, 2)
print(f"Execution time: {execution_time} mins")
