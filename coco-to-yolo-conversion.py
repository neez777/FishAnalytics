import json
import os
from pycocotools import mask as maskUtils
import numpy as np
import argparse

"""
COCO Mask to YOLO Format Converter
---------------------------------
Purpose: This script converts annotation files from COCO format (with segmentation masks)
to YOLO format. It handles both RLE (Run-Length Encoding) and polygon segmentation formats
from COCO and converts them to bounding box coordinates in YOLO format.

The script processes segmentation masks to find the encompassing bounding boxes and
converts them to the normalized coordinates required by YOLO.

Required Dependencies:
- pycocotools
- numpy
"""

def coco_mask_to_yolo(segmentation, image_width, image_height):
    """
    Convert COCO segmentation (RLE or polygon) to YOLO bounding box format.
    
    Args:
        segmentation (dict): COCO segmentation in RLE or polygon format
        image_width (int): Width of the image
        image_height (int): Height of the image
        
    Returns:
        list: Normalized coordinates [x_center, y_center, width, height] in YOLO format
              or None if the mask is empty
    """
    # Convert segmentation to binary mask based on format
    if isinstance(segmentation['counts'], list):
        # Handle polygon format by converting to RLE
        rle = maskUtils.frPyObjects([segmentation], image_height, image_width)
    else:
        # Handle RLE format directly
        rle = segmentation
        
    # Decode RLE to binary mask
    binary_mask = maskUtils.decode(rle)
    
    # Find the bounding box coordinates from the binary mask
    horizontal_indices = np.where(np.any(binary_mask, axis=0))[0]
    vertical_indices = np.where(np.any(binary_mask, axis=1))[0]
    
    if horizontal_indices.shape[0]:
        # Extract bounding box coordinates
        x, x_max = horizontal_indices[[0, -1]]
        y, y_max = vertical_indices[[0, -1]]
        
        # Convert to YOLO format (normalized coordinates)
        x_center = (x + x_max) / 2 / image_width
        y_center = (y + y_max) / 2 / image_height
        width = (x_max - x) / image_width
        height = (y_max - y) / image_height
        
        return [x_center, y_center, width, height]
    else:
        # Return None for empty masks
        return None

def convert_coco_to_yolo(coco_file, output_dir):
    """
    Convert COCO format annotations to YOLO format.
    
    Args:
        coco_file (str): Path to the COCO JSON annotation file
        output_dir (str): Directory where YOLO format annotations will be saved
    
    Creates one .txt file per image with YOLO format annotations:
    <class_id> <x_center> <y_center> <width> <height>
    """
    # Load and parse COCO JSON file
    with open(coco_file, 'r') as f:
        coco_data = json.load(f)
    
    # Create output directory if needed
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each image in the dataset
    for image in coco_data['images']:
        image_id = image['id']
        image_width = image['width']
        image_height = image['height']
        
        # Filter annotations for current image
        image_annotations = [
            ann for ann in coco_data['annotations'] 
            if ann['image_id'] == image_id
        ]
        
        # Convert each annotation to YOLO format
        yolo_annotations = []
        for ann in image_annotations:
            class_id = ann['category_id']
            segmentation = ann['segmentation']
            
            # Convert segmentation to YOLO bbox
            yolo_bbox = coco_mask_to_yolo(segmentation, image_width, image_height)
            
            if yolo_bbox:
                # Format annotation string: class_id x_center y_center width height
                yolo_annotations.append(f"{class_id} {' '.join(map(str, yolo_bbox))}")
        
        # Write annotations to file if any exist
        if yolo_annotations:
            # Use image filename (without extension) for annotation file
            output_file = os.path.join(
                output_dir, 
                f"{image['file_name'].rsplit('.', 1)[0]}.txt"
            )
            with open(output_file, 'w') as f:
                f.write('\n'.join(yolo_annotations))
    
    print(f"Conversion complete. YOLO format annotations saved in {output_dir}")

if __name__ == "__main__":
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description="Convert COCO annotations to YOLO format")
    parser.add_argument("input_dir", help="Directory containing the COCO JSON file")
    parser.add_argument("output_dir", help="Directory to save YOLO format annotations")
    
    # Parse command line arguments
    args = parser.parse_args()
    
    # Construct path to COCO file and perform conversion
    coco_file_path = os.path.join(args.input_dir, "instances_default.json")
    convert_coco_to_yolo(coco_file_path, args.output_dir)
