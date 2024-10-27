import json
import os
import numpy as np
from shapely.geometry import Polygon
import argparse

def coco_mask_to_yolo(segmentation, image_width, image_height):
    # Convert the segmentation to a numpy array
    points = np.array(segmentation[0]).reshape(-1, 2)
    
    # Create a Polygon from the points
    polygon = Polygon(points)
    
    # Get the bounding box
    x, y, max_x, max_y = polygon.bounds
    
    # Convert to YOLO format
    x_center = (x + max_x) / 2 / image_width
    y_center = (y + max_y) / 2 / image_height
    width = (max_x - x) / image_width
    height = (max_y - y) / image_height
    
    return x_center, y_center, width, height

def convert_coco_to_yolo(coco_file_path, output_directory):
    with open(coco_file_path, 'r') as f:
        coco_data = json.load(f)

    # Create output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)

    # Create a mapping of image_id to file_name
    image_map = {image['id']: image for image in coco_data['images']}

    # Use original category IDs instead of remapping them
    category_map = {category['id']: category['id'] for category in coco_data['categories']}

    for annotation in coco_data['annotations']:
        image_id = annotation['image_id']
        image_info = image_map[image_id]
        image_width = image_info['width']
        image_height = image_info['height']
        
        category_id = annotation['category_id']
        yolo_category = category_map[category_id]  # Now uses original category ID
        
        segmentation = annotation['segmentation']
        yolo_bbox = coco_mask_to_yolo(segmentation, image_width, image_height)
        
        # Create YOLO format line
        yolo_line = f"{yolo_category} {' '.join([f'{coord:.6f}' for coord in yolo_bbox])}\n"
        
        # Write to file
        output_file = os.path.join(output_directory, f"{image_info['file_name'].split('.')[0]}.txt")
        with open(output_file, 'a') as f:
            f.write(yolo_line)

        print(f"Processed annotation for category {yolo_category}")  # Added debug print

    print(f"Conversion complete. YOLO format files saved in {output_directory}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert COCO mask annotations to YOLO format")
    parser.add_argument("input_dir", help="Directory containing the COCO JSON file")
    parser.add_argument("output_dir", help="Directory to save YOLO format annotations")
    args = parser.parse_args()

    coco_file_path = os.path.join(args.input_dir, "instances_default.json")
    convert_coco_to_yolo(coco_file_path, args.output_dir)