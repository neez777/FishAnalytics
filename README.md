# Fish Analytics Notebook

This project uses a Jupyter Notebook (`FloSam_FishFinder_Backup.ipynb`) to create and train a dataset for a site-specific fish detection model. The process involves several steps, from extracting frames from a video to training a YOLO model and performing inference. A second notebook, `Stats.ipynb`, is used for statistical analysis of the results.

## Workflow

The main workflow is orchestrated from the `FloSam_FishFinder_Backup.ipynb` notebook and involves the following steps:

1.  **Frame Extraction**: FFMPEG is used to extract frames from a source video.
2.  **Fish Detection**: Florence-2 is used to locate fish in the extracted frames and create bounding boxes.
3.  **Segmentation**: The bounding boxes from Florence-2 are used to assist SAM-2 in creating segmentation masks.
4.  **COCO Conversion**: The segmentation masks are converted to COCO format polygons and exported as a JSON file.
5.  **CVAT Annotation (Optional)**: The images and JSON file can be imported into CVAT to add species labels to the annotations.
6.  **YOLO Conversion**: The annotations are converted to YOLO bounding box format.
7.  **Data Cleaning**: Small and overlapping annotations are identified and assigned to a discard class.
8.  **Image Cropping**: The images are cropped around the remaining annotations to create a dataset of individual fish images.
9.  **Dataset Splitting**: The cropped images are split into training, validation, and testing sets.
10. **YOLO Training**: A YOLO model is trained on the created dataset.
11. **Inference**: The trained model is used to run inference on new images.
12. **Data Extraction**: The results are analyzed to extract MaxN and create species accumulation data.

## Supporting Scripts

The following Python scripts are used to support the main workflow in the notebook:

*   **`coco-to-yolo-conversion.py`**: Converts COCO annotations with segmentation masks to YOLO bounding box format.
*   **`cocomask-to-yolo-conversion.py`**: Converts COCO annotations with polygon segmentations to YOLO bounding box format using the Shapely library.
*   **`cvat-xml-to-yolo-converter.py`**: Converts annotations from CVAT's XML format to YOLO bounding box format.
*   **`cvat-reimport-organiser.py`**: Organizes image and annotation files into a structure for re-importing into CVAT.

## Statistical Analysis

The `Stats.ipynb` notebook is used to perform statistical analysis on the results of the different annotation methods. It includes 1-way and 2-way ANOVA tests, as well as Tukey's HSD test to compare the performance of different methods.
