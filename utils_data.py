import json, sys, os, shutil, yaml, cv2
from exception import CustomException
import matplotlib.pyplot as plt
import matplotlib.patches as patches


# ________________ CONVERT TO YOLO DATA FORMAT ________________ #
def convert_to_yolo_data(
    path_to_input_images,
    path_to_input_JSON,
    path_to_output_images,
    path_to_output_labels,
):
    try:
        # Open JSON file containing image annotations
        file = open(path_to_input_JSON)
        data = json.load(file)
        file.close()

        # Create directories for output images and labels
        os.makedirs(path_to_output_images, exist_ok=True)
        os.makedirs(path_to_output_labels, exist_ok=True)

        # List to store filenames
        file_names = []
        for file_name in os.listdir(path_to_input_images):
            if file_name.endswith(".png"):
                source = os.path.join(path_to_input_images, file_name)
                destination = os.path.join(path_to_output_images, file_name)
                shutil.copy(source, destination)
                file_names.append(file_name)

        # Function to get image annotations
        def get_image_annotation(image_id):
            return [
                annotation
                for annotation in data["annotations"]
                if annotation["image_id"] == image_id
            ]

        # Function to get image data
        def get_image(file_name):
            return next(
                (image for image in data["images"] if image["file_name"] == file_name),
                None,
            )

        # Iterate through filenames and process each image
        for file_name in file_names:
            image = get_image(file_name)
            image_id = image["id"]
            image_w = image["width"]
            image_h = image["height"]
            image_annotation = get_image_annotation(image_id)

            # Write normalized polygon data to a text file
            if image_annotation:
                with open(
                    os.path.join(
                        path_to_output_labels, f"{os.path.splitext(file_name)[0]}.txt"
                    ),
                    "a",
                ) as file_object:
                    for annotation in image_annotation:
                        current_category = annotation["category_id"] - 1
                        polygon = annotation["segmentation"][0]
                        normalized_polygon = [
                            format(
                                coord / image_w if i % 2 == 0 else coord / image_h,
                                ".6f",
                            )
                            for i, coord in enumerate(polygon)
                        ]
                        file_object.write(
                            f"{current_category} " + " ".join(normalized_polygon) + "\n"
                        )
        return len(file_names)

    except Exception as e:
        raise CustomException(e, sys)


# ________________ MAKE YAML FILE ________________ #
def make_yaml(
    path_to_input_JSON,
    path_to_output_yaml,
    path_to_train,
    path_to_test,
    path_to_val=None,
):
    try:
        with open(path_to_input_JSON) as file:
            data = json.load(file)

        # Extract the category names
        names = [category["name"] for category in data["categories"]]

        # Number of classes
        number_classes = len(names)

        # Create a dictionary with the required content
        yaml_data = {
            "names": names,
            "nc": number_classes,
            "train": path_to_train,
            "test": path_to_test,
            "val": path_to_val if path_to_val else "",
        }

        # Write the dictionary to a YAML file
        with open(path_to_output_yaml, "w") as file:
            yaml.dump(yaml_data, file, default_flow_style=False)

    except Exception as e:
        raise CustomException(e, sys)


# ________________ MAKE ANNOTATED IMAGES ________________ #
def make_annotated_images(
    path_to_images, path_to_annotations, path_to_save_annotated_images, colors=None
):
    try:
        if not os.path.exists(path_to_save_annotated_images):
            os.makedirs(path_to_save_annotated_images)

        # Get list of image files
        image_files = [
            f
            for f in os.listdir(path_to_images)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]

        # Loop over the image files
        for image_name in image_files:
            # Construct the basename to match annotation file
            basename = os.path.splitext(image_name)[0]
            annotation_file = basename + ".txt"  # Assuming annotation files are .txt

            # Paths to image and its corresponding annotation
            path_to_image = os.path.join(path_to_images, image_name)
            path_to_annotation = os.path.join(path_to_annotations, annotation_file)

            if not os.path.exists(path_to_annotation):
                print(f"No annotation file found for {image_name}")
                continue

            # Load image using OpenCV and convert it from BGR to RGB color space
            image = cv2.imread(path_to_image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            img_h, img_w, _ = image.shape

            # Create a figure and axis to display the image
            fig, ax = plt.subplots(1)
            ax.imshow(image)
            ax.axis("off")  # Hide the axes

            if colors is None:
                colors = plt.cm.get_cmap("tab10")

            # Open the annotation file and process each line
            with open(path_to_annotation, "r") as file:
                for line in file:
                    parts = line.strip().split()
                    category_id = int(parts[0])
                    color = colors(category_id % 10)
                    polygon = [float(coord) for coord in parts[1:]]
                    polygon = [
                        coord * img_w if i % 2 == 0 else coord * img_h
                        for i, coord in enumerate(polygon)
                    ]
                    polygon = [
                        (polygon[i], polygon[i + 1]) for i in range(0, len(polygon), 2)
                    ]
                    patch = patches.Polygon(
                        polygon, closed=True, edgecolor=color, fill=False
                    )
                    ax.add_patch(patch)

            path_to_save_annotated_image = os.path.join(
                path_to_save_annotated_images, image_name
            )
            plt.savefig(path_to_save_annotated_image, bbox_inches="tight", pad_inches=0)
            plt.close(fig)

        print(
            f"{len(image_files)}  annotated images were saved to {path_to_save_annotated_images}"
        )

    except Exception as e:
        raise CustomException(e, sys)
