import os, sys, leafmap
import pandas as pd
import seaborn as sns
from exception import CustomException
from ultralytics import YOLO
from matplotlib import pyplot as plt
from skimage.measure import regionprops
from PIL import Image


# ________________ MAKE MODEL CONFIGURATION ________________ #
class YOLOSegmentation:
    def __init__(
        self, test_name, model_yaml, model_pt, data, project, note="", **kwargs
    ):
        # Default configuration values
        self.config = {
            "test_name": test_name,
            "model_yaml": model_yaml,
            "model_pt": model_pt,
            "data": data,
            "project": project,
            "epochs": 100,
            "patience": 50,
            "batch": 16,
            "imgsz": 640,  # image size
            "device": 0,
            "workers": 8,
            "pretrained": True,
            "optimizer": "auto",
            "verbose": True,
            "lr0": 0.01,
            "weight_decay": 0.0005,
            # "dropout": 0.0,
            "note": note,
        }

        # Apply any overrides provided upon initialization
        self.config.update(kwargs)

        # Create the project directory if it doesn't exist
        os.makedirs(self.config["project"], exist_ok=True)

    def update_config(self, **kwargs):
        # Update the configuration with new values
        self.config.update(kwargs)

    def train(self):
        try:
            # Instantiate the model using the provided YAML and PT files
            model = YOLO(self.config["model_yaml"])
            model = YOLO(self.config["model_pt"])

            # Start the training process with the provided configuration
            results = model.train(
                data=self.config["data"],
                project=self.config["project"],
                epochs=self.config["epochs"],
                patience=self.config["patience"],
                batch=self.config["batch"],
                imgsz=self.config["imgsz"],
                device=self.config["device"],
                workers=self.config["workers"],
                pretrained=self.config["pretrained"],
                optimizer=self.config["optimizer"],
                verbose=self.config["verbose"],
                lr0=self.config["lr0"],
                weight_decay=self.config["weight_decay"],
                # dropout=self.config["dropout"],
            )

            return results

        except Exception as e:
            raise CustomException(e, sys)

    def validate(self, path_to_val_model):
        try:
            model = YOLO(path_to_val_model)

            metrics = model.val(
                data=self.config["data"],
                project=self.config["project"],
            )

            return metrics

        except Exception as e:
            raise CustomException(e, sys)


# ________________ MAKE PREDICTION 1 IMAGE ________________ #
def make_prediction_1image(
    path_to_chosen_model,
    image_file_name,
    path_to_image,
    path_to_save_annotated_image,
    path_to_save_CSV,
    conf=0.25,
):
    try:
        model = YOLO(path_to_chosen_model)

        full_path_to_image = os.path.join(path_to_image, image_file_name)

        prediction = model.predict(full_path_to_image, conf=conf)

        prediction_array = prediction[0].plot()

        # Create a figure and axis to display the image
        fig, ax = plt.subplots(1)
        ax.imshow(prediction_array)
        ax.axis("off")  # Hide the axes

        full_path_to_save_annotated_image = os.path.join(
            path_to_save_annotated_image, image_file_name
        )

        plt.savefig(
            full_path_to_save_annotated_image, bbox_inches="tight", pad_inches=0
        )
        plt.close(fig)

        prediction_0 = prediction[0]

        extracted_masks = prediction_0.masks.data
        # Extract the boxes, which likely contain class IDs
        detected_boxes = prediction_0.boxes.data
        # Extract class IDs from the detected boxes
        class_labels = detected_boxes[:, -1].int().tolist()
        masks_by_class = {name: [] for name in prediction_0.names.values()}

        # Iterate through the masks and class labels
        for mask, class_id in zip(extracted_masks, class_labels):
            class_name = prediction_0.names[class_id]  # Map class ID to class name
            masks_by_class[class_name].append(mask.cpu().numpy())

        # Initialize a list to store the properties
        props_list = []

        # Iterate through all classes
        for class_name, masks in masks_by_class.items():
            # Iterate through the masks for this class
            for mask in masks:
                # Convert the mask to an integer type if it's not already
                mask = mask.astype(int)

                # Apply regionprops to the mask
                props = regionprops(mask)

                # Extract the properties you want (e.g., area, perimeter) and add them to the list
                for prop in props:
                    area = prop.area
                    perimeter = prop.perimeter
                    # Add other properties as needed

                    # Append the properties and class name to the list
                    props_list.append(
                        {
                            "Class Name": class_name,
                            "Area": area,
                            "Perimeter": perimeter,
                        }
                    )

            # Convert the list of dictionaries to a DataFrame
            props_df = pd.DataFrame(props_list)
            # Save the DataFrame to a CSV file
            image_file_name_withoutEXT, _ = os.path.splitext(image_file_name)
            full_path_to_save_CSV = os.path.join(
                path_to_save_CSV, f"{image_file_name_withoutEXT}.csv"
            )
            props_df.to_csv(full_path_to_save_CSV, index=False)

    except Exception as e:
        raise CustomException(e, sys)


# ________________ MAKE PREDICTION IMAGES ________________ #
def make_predictions_images(
    path_to_chosen_model,
    path_to_images,
    path_to_save_annotated_images,
    path_to_save_CSV,
    conf=0.25,
):
    try:
        # Create the output directory if it doesn't exist
        if not os.path.exists(path_to_save_annotated_images):
            os.makedirs(path_to_save_annotated_images)

        if not os.path.exists(path_to_save_CSV):
            os.makedirs(path_to_save_CSV)

        model = YOLO(path_to_chosen_model)

        # Process each image file in the input directory
        for image_file_name in os.listdir(path_to_images):
            # Check for image file extensions
            if not image_file_name.lower().endswith((".png", ".jpg", ".jpeg")):
                continue

            # Full path to the image file
            path_to_image = os.path.join(path_to_images, image_file_name)

            prediction = model.predict(path_to_image, conf=conf)
            prediction_array = prediction[0].plot()

            # Create a figure and axis to display the image
            fig, ax = plt.subplots(1)
            ax.imshow(prediction_array)
            ax.axis("off")  # Hide the axes

            # Save the annotated image to the output directory
            save_path = os.path.join(path_to_save_annotated_images, image_file_name)
            plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
            plt.close(fig)

            prediction_0 = prediction[0]

            extracted_masks = prediction_0.masks.data
            # Extract the boxes, which likely contain class IDs
            detected_boxes = prediction_0.boxes.data
            # Extract class IDs from the detected boxes
            class_labels = detected_boxes[:, -1].int().tolist()
            masks_by_class = {name: [] for name in prediction_0.names.values()}

            # Iterate through the masks and class labels
            for mask, class_id in zip(extracted_masks, class_labels):
                class_name = prediction_0.names[class_id]  # Map class ID to class name
                masks_by_class[class_name].append(mask.cpu().numpy())

            # Initialize a list to store the properties
            props_list = []

            # Iterate through all classes
            for class_name, masks in masks_by_class.items():
                # Iterate through the masks for this class
                for mask in masks:
                    # Convert the mask to an integer type if it's not already
                    mask = mask.astype(int)

                    # Apply regionprops to the mask
                    props = regionprops(mask)

                    # Extract the properties you want (e.g., area, perimeter) and add them to the list
                    for prop in props:
                        area = prop.area
                        perimeter = prop.perimeter
                        # Add other properties as needed

                        # Append the properties and class name to the list
                        props_list.append(
                            {
                                "Class Name": class_name,
                                "Area": area,
                                "Perimeter": perimeter,
                            }
                        )

            # Convert the list of dictionaries to a DataFrame
            props_df = pd.DataFrame(props_list)
            # Save the DataFrame to a CSV file
            image_file_name_withoutEXT, _ = os.path.splitext(image_file_name)
            full_path_to_save_CSV = os.path.join(
                path_to_save_CSV, f"{image_file_name_withoutEXT}.csv"
            )
            props_df.to_csv(full_path_to_save_CSV, index=False)

    except Exception as e:
        # CustomException should be defined elsewhere in your code
        raise CustomException(e, sys)


# ________________ MAKE COMPARISON IMAGES AND SAVE TO .HTML ________________ #
# resize images before making comparison image
def resize_image(input_path, output_path, resize=0.5):
    try:
        with Image.open(input_path) as img:
            width, height = img.size
            new_size = (int(width * resize), int(height * resize))
            img = img.resize(new_size, Image.ANTIALIAS)
            img.save(output_path)

    except Exception as e:
        raise CustomException(e, sys)


# ________________ MAKE COMPARISON IMAGES AND SAVE TO .HTML ________________ #
def make_comparison_images(
    path_to_ground_truth, path_to_segm_image, path_to_save_html, resize=1.0
):
    try:
        if not os.path.exists(path_to_save_html):
            os.makedirs(path_to_save_html)

        # Loop over the images in the input folder
        for image_name in os.listdir(path_to_ground_truth):
            if not image_name.lower().endswith((".png", ".jpg", ".jpeg")):
                continue

            full_path_to_ground_truth = os.path.join(path_to_ground_truth, image_name)
            full_path_to_segm_image = os.path.join(path_to_segm_image, image_name)
            image_name_0 = os.path.splitext(image_name)[0]
            full_path_to_save_html = os.path.join(path_to_save_html, image_name_0)

            if resize != 1.0:
                # Resize images
                resized_ground_truth = f"{full_path_to_ground_truth}_resized.png"
                resized_segm_image = f"{full_path_to_segm_image}_resized.png"
                resize_image(full_path_to_ground_truth, resized_ground_truth, resize)
                resize_image(full_path_to_segm_image, resized_segm_image, resize=1.0)
            else:
                resized_ground_truth = full_path_to_ground_truth
                resized_segm_image = full_path_to_segm_image

            leafmap.image_comparison(
                resized_ground_truth,
                resized_segm_image,
                label1="True labels",
                label2="Image Segmentation",
                starting_position=50,
                out_html=f"{full_path_to_save_html}.html",
            )

            # Optionally, remove resized images to save disk space
            if resize != 1.0:
                os.remove(resized_ground_truth)
                os.remove(resized_segm_image)

    except Exception as e:
        raise CustomException(e, sys) from e


def combine_CSVs(path_to_CSVs, path_to_save_CSV, scalar):
    """
    Combines all CSV files in a given folder into a single CSV file with additional columns for file name and image ID.

    Params:
    - path_to_CSVs: Path to the folder containing the CSV files.
    - path_to_save_CSV: Path to the folder to save combined CSV file.

    """
    try:
        if not os.path.exists(path_to_save_CSV):
            os.makedirs(path_to_save_CSV)

        # Initialize a list to store DataFrames
        dataframes = []

        # Get all CSV file paths in the directory
        csv_files = [f for f in os.listdir(path_to_CSVs) if f.endswith(".csv")]
        csv_files.sort()  # Sort the files to maintain order

        # Go through each CSV file and append it to the DataFrame list
        for i, file_name in enumerate(csv_files, 1):
            file_path = os.path.join(path_to_CSVs, file_name)
            df = pd.read_csv(file_path)
            df["Area"] *= scalar
            df["File Name"] = file_name.replace(
                ".csv", ".png"
            )  # Assuming the images have .png extension
            df["Image ID"] = i
            dataframes.append(df)

        # Concatenate all DataFrames into a single DataFrame
        combined_df = pd.concat(dataframes, ignore_index=True)

        # Save the combined DataFrame to a CSV file
        combined_csv_path = os.path.join(path_to_save_CSV, "segm_df.csv")
        combined_df.to_csv(combined_csv_path, index=False)

    except Exception as e:
        raise CustomException(e, sys)


# ________________ MAKE COMPARISON BAR PLOTS ________________ #
class PlotMitochondria:
    def __init__(self, path_to_segmentation_csv, path_to_ground_truth_csv, image_id=1):
        self.path_to_segmentation_csv = path_to_segmentation_csv
        self.path_to_ground_truth_csv = path_to_ground_truth_csv
        self.image_id = image_id
        self.segmentation_data, self.ground_truth_data = self.read_and_prepare_data()

    def read_and_prepare_data(self):
        try:
            segmentation_df = pd.read_csv(self.path_to_segmentation_csv)
            ground_truth_df = pd.read_csv(self.path_to_ground_truth_csv)

            segmentation_df.rename(
                columns={"area": "Area", "image ID": "Image ID"}, inplace=True
            )
            ground_truth_df.rename(
                columns={"area": "Area", "image ID": "Image ID"}, inplace=True
            )

            segmentation_data = segmentation_df[
                segmentation_df["Image ID"] == self.image_id
            ]
            ground_truth_data = ground_truth_df[
                ground_truth_df["Image ID"] == self.image_id
            ]

            return segmentation_data, ground_truth_data

        except Exception as e:
            raise CustomException(e, sys) from e

    def calculate_metrics(self):
        try:
            mean_area_segmentation = self.segmentation_data["Area"].mean()
            mean_area_ground_truth = self.ground_truth_data["Area"].mean()
            count_segmentation = self.segmentation_data.shape[0]
            count_ground_truth = self.ground_truth_data.shape[0]

            return (
                mean_area_segmentation,
                mean_area_ground_truth,
                count_segmentation,
                count_ground_truth,
            )

        except Exception as e:
            raise CustomException(e, sys) from e

    def create_plot_data(self, prediction_label, true_label):
        try:
            (
                mean_area_segmentation,
                mean_area_ground_truth,
                count_segmentation,
                count_ground_truth,
            ) = self.calculate_metrics()

            mean_area_data = pd.DataFrame(
                {
                    "Dataset": [prediction_label, true_label],
                    "Mean Area": [mean_area_segmentation, mean_area_ground_truth],
                }
            )

            mitochondria_counts = pd.DataFrame(
                {
                    "Dataset": [prediction_label, true_label],
                    "Count": [count_segmentation, count_ground_truth],
                }
            )

            return mean_area_data, mitochondria_counts

        except Exception as e:
            raise CustomException(e, sys) from e

    def add_text_annotations(self, ax, dataset_names, font_size):
        try:
            for i, p in enumerate(ax.patches):
                width = p.get_width()
                ax.text(
                    p.get_x() + width / 2,
                    p.get_y() + p.get_height() / 2,
                    dataset_names[i % len(dataset_names)],
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=font_size,
                )

        except Exception as e:
            raise CustomException(e, sys) from e

    def make_bar_plot(
        self, data, x, y, palette, xlabel, title, font_size=9, figsize=(5, 2)
    ):
        try:
            f, ax = plt.subplots(1, 1, figsize=figsize)
            sns.barplot(x=x, y=y, data=data, palette=palette, ax=ax, orient="h")
            ax.set(xlabel=xlabel, ylabel="", title=title)
            ax.yaxis.set_visible(False)
            sns.despine(left=True, bottom=True, ax=ax)
            self.add_text_annotations(ax, data[y].tolist(), font_size)
            plt.tight_layout()
            return f

        except Exception as e:
            raise CustomException(e, sys) from e

    def plot_and_save(
        self,
        image_id,
        path_to_save_bar_plots,
        prediction_label="Prediction",
        true_label="True Label",
        font_size=9,
        figsize=(5, 2),
    ):
        try:
            mean_area_data, mitochondria_counts = self.create_plot_data(
                prediction_label, true_label
            )
            bar_colors = {prediction_label: "#8B4513", true_label: "#2F4F4F"}

            formatted_image_id = str(image_id).zfill(
                3
            )  # Ensure image_id is at least 3 digits

            fig = self.make_bar_plot(
                mean_area_data,
                "Mean Area",
                "Dataset",
                bar_colors,
                "",
                "",  # plot title goes here
                font_size,
                figsize,
            )
            fig.savefig(
                os.path.join(
                    path_to_save_bar_plots, f"barplot_area_{formatted_image_id}.png"
                )
            )
            plt.close(fig)

            fig = self.make_bar_plot(
                mitochondria_counts,
                "Count",
                "Dataset",
                bar_colors,
                "",
                "",  # plot title goes here
                font_size,
                figsize,
            )
            fig.savefig(
                os.path.join(
                    path_to_save_bar_plots, f"barplot_count_{formatted_image_id}.png"
                )
            )
            plt.close(fig)

        except Exception as e:
            raise CustomException(e, sys)


# ________________ GET UNIQUE IMAGE IDS ________________ #
def get_unique_image_ids(csv_path):
    try:
        df = pd.read_csv(csv_path)
        unique_ids = df["Image ID"].unique()
        all_unique_ids = set(unique_ids) | set(unique_ids)
        return sorted(all_unique_ids)

    except Exception as e:
        raise CustomException(e, sys)
