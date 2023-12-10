<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#introduction">Introduction</a>
    </li>
    <li><a href="#technical-tools">Technical Tools</a></li>
    <li><a href="#data-source">Data source</a></li>
    <li><a href="#data-processing">Data processing</a></li>
    <li><a href="#the-design">The design</a></li>
    <li><a href="#results">Results</a></li>
    <li><a href="#how-to-use-the-source-code">How to use the source code</a></li>
    <li><a href="#the-bottom-line">The Bottom Line</a></li>
    <li><a href="#reference">Reference</a></li>
  </ol>
</details>

### Introduction

This repository contains the source code for an instance segmentation project, focusing on segmenting mitochondria in microscopic images. The architecture of You Only Look Once framework (i.e., <a href="https://github.com/facebookresearch/detectron2/blob/main/README.md"> YOLOv8 </a>) has been utilized as the main tool for this task.

### Technical tools

The orginal paper of You Only Look Once YOLO architecture <a href="https://arxiv.org/pdf/1506.02640.pdf">(Redmon, J. et al., 2016)</a>.

The application documentation of <a href="https://docs.ultralytics.com/"> YOLOv8 </a> by Ultralytics.

-   Pytorch
-   YOLOv8
-   Opencv
-   numpy
-   pandas
-   scikit-image
-   leafmap
-   matplotlib
-   seaborn
-   Docker

### Data source

This project utilizes an electron microscopy image dataset from the CA1 area of the hippocampus, annotated for mitochondria by highly trained researchers. For more information and access to the dataset, visit the <a href="https://www.epfl.ch/labs/cvlab/data/data-em/"> Electron Microscopy Dataset</a>.

### Data processing

This project makes use of a dataset that includes PNG images along with their corresponding COCOJSON annotations. For the purpose of transforming these COCOJSON annotations into the TXT format, as required by YOLO, I utilized and refined some code originally introduced by <a href="https://www.youtube.com/watch?v=NYeJvxe5nYw"> Bhattiprolu, S., 2023</a>.

The usage of DataHandler and AnnotationHandler

```python
# Make image data
convert_to_yolo_data(path_to_input_images, path_to_input_JSON, path_to_output_images, path_to_output_labels)

# Make the YAML configuration file
make_yaml(path_to_input_JSON, path_to_output_yaml, path_to_train, path_to_test, path_to_val)
```

### The design

<img src="media/code_structure.png" alt="" width="680">

The above diagram illustrates the fundamental architecture of this project. Below, you will find an example outline detailing the code setups necessary for experimenting with various configurations of YOLOv8 for this specific task.

Model configuration for experiment was done in the file YOLO8_0.py with the outline of code structure as follows:

```python
# ________________ MAKE MODEL CONFIGURATION ________________ #
# ****** --------- ****** #
test_name = "test2"
note = ""
# ****** --------- ****** #
model_yaml = "yolov8x-seg.yaml"
model_pt = "yolov8x-seg.pt"

# ________________ MAKE TRAIN DATA READY FOR TRAINING ________________ #
...

# custom configuration
...

model = YOLOSegmentation(
    ...
)

# _ TRAIN _ #
model.train()

```

Ultralytics provides a comprehensive model evaluation for each time of model training. In this project, I simply arranged for the model results to be stored in a specifically named folder, corresponding to the name of the test, within the models directory.

### Results

This project involved numerous experiments, but only the most significant results are highlighted here.

<br>

<h6 align="center">
Model performance metrics
</h6>

<p align="center">
    <img src="models/test0/train/results.png" alt="" width="820">
</p>

<br>

<h6 align="center">
Visualization of prediction on an unseen image (name: test_001.png)
</h6>

<p align="center">
    <img src="output/viz/test0/segm_images/test_001.png" alt="" width="680">
</p>

<br>

<h6 align="center">
Visualization of the ground truth for the predicted image (name: test_001.png)
</h6>

<p align="center">
    <img src="media/test_001 copy.png" alt="" width="680">
</p>

<br>

In a comparative experiment with this dataset, I evaluated both Detectron2 and YOLOv8. The results indicated that, for this particular task, Detectron2 demonstrated superior performance over YOLOv8 (see <a href="https://mito-app-651cbfda9bde.herokuapp.com/"> comparison images)</a>. However, in some cases, YOLOv8 is better for the task of object detection.

<br>

### How to use the source code

##### Using the source code for development

-   Fork/clone this repository (https://github.com/LeoUtas/Mitochondria_segmentation_research_YOLOv8.git).
-   First thing first, before proceeding, ensure that you are in the root directory of the project.
-   Get the docker container ready:

    -   Run docker build (it might take a while for installing all the required dependencies to your local docker image).

    ```cmd
    docker build -t <name of the docker image> .
    ```

    -   Run a docker container in an interactive mode (once the docker image is built, you can run a docker container).

    ```cmd
    docker run -it -v "$(PWD):/app" <name of the docker image> /bin/bash
    ```

    -   Now, it should be running inside the interactive mode of the docker container to explore the code functionalities.

    -   The YOLO8_0.py file is responsible for training the model. To comply with GitHub's file size restrictions, the trained results are not synchronized between the remote and local versions of this project.
    -   It is recommended to train the model using GPU.

    ```python
    # ****** --------- ****** #
    test_name = "name of the test"
    note = ""
    # ****** --------- ****** #
    ```

    -   When you're done, you can simply type "exit" to escape the development environment

    ```
    exit
    ```

-   Also, stop running the container when you're done:

```cmd
docker stop <name of the container>
```

### The bottom line

I'm excited to share this repository! Please feel free to explore its functionalities. Thank you for this far. Have a wonderful day ahead!

Best,
Hoang Ng

### Reference

Bhattiprolu, S. (2023, Sep 20). 332 - All about image annotations​ [Video]. YouTube. https://www.youtube.com/watch?v=NYeJvxe5nYw

Redmon, J. et al., 2016. You Only Look Once: Unified, Real-Time Object Detection. In 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR). pp. 779–788.
