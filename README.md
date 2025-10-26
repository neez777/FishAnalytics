# Computer Vision Annotation Toolkit

This project is a collection of Python scripts for converting and organizing computer vision annotations between different formats, including COCO, YOLO, and CVAT. These tools are designed to streamline the workflow of preparing and managing datasets for object detection and segmentation tasks.

## Scripts

### COCO to YOLO Conversion

There are two scripts available for converting COCO annotations to YOLO format.

#### 1. `coco-to-yolo-conversion.py`

*   **Purpose**: Converts COCO annotations with segmentation masks (both RLE and polygon formats) to YOLO bounding box format.
*   **Dependencies**: `pycocotools`, `numpy`
*   **Usage**:
    ```bash
    python coco-to-yolo-conversion.py <input_dir> <output_dir>
    ```
    *   `input_dir`: Directory containing the COCO JSON file (e.g., `instances_default.json`).
    *   `output_dir`: Directory where the YOLO format annotation files will be saved.

#### 2. `cocomask-to-yolo-conversion.py`

*   **Purpose**: Converts COCO annotations with polygon segmentations to YOLO bounding box format using the Shapely library for efficient polygon processing. This script preserves the original COCO category IDs.
*   **Dependencies**: `numpy`, `shapely`
*   **Usage**:
    ```bash
    python cocomask-to-yolo-conversion.py <input_dir> <output_dir>
    ```
    *   `input_dir`: Directory containing the COCO JSON file (e.g., `instances_default.json`).
    *   `output_dir`: Directory where the YOLO format annotation files will be saved.

### CVAT to YOLO Conversion

#### `cvat-xml-to-yolo-converter.py`

*   **Purpose**: Converts annotations from CVAT's XML format to YOLO bounding box format.
*   **Dependencies**: None
*   **Usage**:
    ```bash
    python cvat-xml-to-yolo-converter.py <input_dir> <output_dir>
    ```
    *   `input_dir`: Directory containing the CVAT XML file (e.g., `annotations.xml`).
    *   `output_dir`: Directory where the YOLO format annotation files will be saved.

### CVAT Re-import Organizer

#### `cvat-reimport-organiser.py`

*   **Purpose**: Organizes image and annotation files into a specific structure required for re-importing into CVAT. It creates a zip file that can be easily uploaded to CVAT.
*   **Dependencies**: None
*   **Usage**:
    The script is set up to be run directly, but you will need to modify the `input_directory` and `output_zip_file` variables within the script to match your project's paths.
