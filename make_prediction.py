import sys, os
from utils_model import *

# ________________ HANDLE THE PATH THING ________________ #
# get the absolute path of the script's directory
script_path = os.path.dirname(os.path.abspath(__file__))
# get the parent directory of the script's directory
parent_path = os.path.dirname(script_path)
sys.path.append(parent_path)


# ________________ MAKE MODEL CONFIGURATION ________________ #
# ****** --------- ****** #
test_name = "test2"
note = ""
# ****** --------- ****** #


# ________________ MAKE SEGMENTATION IMAGES & CSV => SAVE ________________ #
path_to_chosen_model = os.path.join(
    script_path, "models", test_name, "train", "weights", "best.pt"
)
path_to_images = os.path.join(script_path, "input", "data", "test", "images")
path_to_save_annotated_images = os.path.join(
    script_path, "output", "viz", test_name, "segm_images"
)
path_to_save_CSV = os.path.join(script_path, "output", "data", test_name, "CSVs")
if not os.path.exists(path_to_save_CSV):
    os.makedirs(path_to_save_CSV)

make_predictions_images(
    path_to_chosen_model,
    path_to_images,
    path_to_save_annotated_images,
    path_to_save_CSV,
    conf=0.2,
)


# ________________ COMBINE CSVs => SAVE ________________ #
path_to_CSVs = path_to_save_CSV
path_to_save_CSV = os.path.join(
    script_path, "output", "data", test_name, "combined_CSV"
)
scalar = 1024 / 640

combine_CSVs(path_to_CSVs, path_to_save_CSV, scalar)


# ________________ MAKE COMPARISON IMAGES => SAVE AS HTML ________________ #
path_to_ground_truth = os.path.join(
    script_path, "resource", "data_org", "data", "test", "test_masks", "mitochondria"
)
path_to_segm_image = path_to_save_annotated_images
path_to_save_html = os.path.join(
    script_path, "output", "viz", test_name, "compare_images"
)
if not os.path.exists(path_to_save_html):
    os.makedirs(path_to_save_html)

make_comparison_images(
    path_to_ground_truth, path_to_segm_image, path_to_save_html, resize=0.625
)


# ________________ MAKE COMPARE BAR PLOTS ________________ #
path_to_segmentation_csv = os.path.join(path_to_save_CSV, "segm_df.csv")
path_to_ground_truth_csv = os.path.join(
    script_path, "input", "data", "test", "combined_CSV", "annotations_df.csv"
)
path_to_save_bar_plots = os.path.join(
    script_path, "output", "viz", test_name, "compare_plots"
)
if not os.path.exists(path_to_save_bar_plots):
    os.makedirs(path_to_save_bar_plots)

unique_image_ids = get_unique_image_ids(path_to_ground_truth_csv)

for image_id in unique_image_ids:
    plotter = PlotMitochondria(
        path_to_segmentation_csv, path_to_ground_truth_csv, image_id
    )
    plotter.plot_and_save(image_id, path_to_save_bar_plots, figsize=(2.5, 1.2))
