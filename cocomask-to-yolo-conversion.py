import json
import os
import numpy as np
from shapely.geometry import Polygon
import argparse

"""
COCO Polygon to YOLO Format Converter
-----------------------------------
Purpose: This script converts COCO format annotations with polygon segmentations
to YOLO format bounding box annotations. It uses Shapely to handle polygon
operations and extract bounding boxes efficiently.

The script maintains original category IDs from the COCO dataset instead of
remapping them, which is useful when working with specific predefined category IDs.

Required Dependencies:
- numpy
- shapely
"""

def coco_mask_to_yolo(segmentation, image_width, image_height):
    """
    Convert COCO polygon segmentation to YOLO bounding box format.
    
    Args:
        segmentation (list): List of polygon coordinates in COCO format [x1,y1,x2,y2,...]
        image_width (int): Width of the image
        image_height (int): Height of the image
        
    Returns:
        tuple: Normalized coordinates (x_center, y_center, width, height) in YOLO format
    """
    # Reshape flat list of coordinates into pairs of (x,y) points
    points = np.array(segmentation[0]).reshape(-1, 2)
    
    # Create a Shapely polygon for efficient bounds calculation
    polygon = Polygon(points)
    
    # Get the min/max bounds of the polygon (x_min, y_min, x_max, y_max)
    x, y, max_x, max_y = polygon.bounds
    
    # Convert to YOLO format (normalized center coordinates and dimensions)
    x_center = (x + max_x) / 2 / image_width
    y_center = (y + max_y) / 2 / image_height
    width = (max_x - x) / image_width
    height = (max_y - y) / image_height
    
    return x_center, y_center, width, height

def convert_coco_to_yolo(coco_file_path, output_directory):
    """
    Convert COCO format annotations to YOLO format.
    
    Args:
        coco_file_path (str): Path to the COCO JSON annotation file
        output_directory (str): Directory where YOLO format annotations will be saved
    
    Creates one .txt file per image with YOLO format annotations:
    <category_id> <x_center> <y_center> <width> <height>
    Maintains original COCO category IDs in the output.
    """
    # Load COCO format annotations
    with open(coco_file_path, 'r') as f:
        coco_data = json.load(f)

    # Create output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)

    # Create lookup dictionary for image information using image_id as key
    image_map = {image['id']: image for image in coco_data['images']}
    
    # Create category mapping that preserves original COCO category IDs
    category_map = {category['id']: category['id'] for category in coco_data['categories']}

    # Process each annotation
    for annotation in coco_data['annotations']:
        # Get image information
        image_id = annotation['image_id']
        image_info = image_map[image_id]
        image_width = image_info['width']
        image_height = image_info['height']
        
        # Get category ID (maintaining original ID)
        category_id = annotation['category_id']
        yolo_category = category_map[category_id]
        
        # Convert polygon segmentation to YOLO bbox
        segmentation = annotation['segmentation']
        yolo_bbox = coco_mask_to_yolo(segmentation, image_width, image_height)
        
        # Format YOLO annotation line
        yolo_line = f"{yolo_category} {' '.join([f'{coord:.6f}' for coord in yolo_bbox])}\n"
        
        # Write to output file (append mode to handle multiple annotations per image)
        output_file = os.path.join(output_directory, f"{image_info['file_name'].split('.')[0]}.txt")
        with open(output_file, 'a') as f:
            f.write(yolo_line)
        
        print(f"Processed annotation for category {yolo_category}")
    
    print(f"Conversion complete. YOLO format files saved in {output_directory}")

if __name__ == "__main__":
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description="Convert COCO mask annotations to YOLO format")
    parser.add_argument("input_dir", help="Directory containing the COCO JSON file")
    parser.add_argument("output_dir", help="Directory to save YOLO format annotations")
    
    # Parse command line arguments
    args = parser.parse_args()
    
    # Construct path to COCO file and perform conversion
    coco_file_path = os.path.join(args.input_dir, "instances_default.json")
    convert_coco_to_yolo(coco_file_path, args.output_dir)
