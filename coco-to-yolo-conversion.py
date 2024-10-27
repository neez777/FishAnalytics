import json
import os
from pycocotools import mask as maskUtils
import numpy as np
import argparse

def coco_mask_to_yolo(segmentation, image_width, image_height):
    # Convert COCO RLE or polygon to binary mask
    if isinstance(segmentation['counts'], list):
        # Polygon format
        rle = maskUtils.frPyObjects([segmentation], image_height, image_width)
    else:
        # RLE format
        rle = segmentation
    binary_mask = maskUtils.decode(rle)

    # Find bounding box
    horizontal_indicies = np.where(np.any(binary_mask, axis=0))[0]
    vertical_indicies = np.where(np.any(binary_mask, axis=1))[0]
    if horizontal_indicies.shape[0]:
        x, x_max = horizontal_indicies[[0, -1]]
        y, y_max = vertical_indicies[[0, -1]]
        
        # Compute YOLO format
        x_center = (x + x_max) / 2 / image_width
        y_center = (y + y_max) / 2 / image_height
        width = (x_max - x) / image_width
        height = (y_max - y) / image_height
        
        return [x_center, y_center, width, height]
    else:
        return None

def convert_coco_to_yolo(coco_file, output_dir):
    # Load COCO JSON file
    with open(coco_file, 'r') as f:
        coco_data = json.load(f)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Process each image
    for image in coco_data['images']:
        image_id = image['id']
        image_width = image['width']
        image_height = image['height']
        
        # Find annotations for this image
        image_annotations = [ann for ann in coco_data['annotations'] if ann['image_id'] == image_id]
        
        # Convert annotations to YOLO format
        yolo_annotations = []
        for ann in image_annotations:
            class_id = ann['category_id']
            segmentation = ann['segmentation']
            yolo_bbox = coco_mask_to_yolo(segmentation, image_width, image_height)
            if yolo_bbox:
                yolo_annotations.append(f"{class_id} {' '.join(map(str, yolo_bbox))}")
        
        # Write YOLO annotations to file
        if yolo_annotations:
            output_file = os.path.join(output_dir, f"{image['file_name'].rsplit('.', 1)[0]}.txt")
            with open(output_file, 'w') as f:
                f.write('\n'.join(yolo_annotations))

    print(f"Conversion complete. YOLO format annotations saved in {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert COCO annotations to YOLO format")
    parser.add_argument("input_dir", help="Directory containing the COCO JSON file")
    parser.add_argument("output_dir", help="Directory to save YOLO format annotations")
    args = parser.parse_args()

    coco_file_path = os.path.join(args.input_dir, "instances_default.json")
    convert_coco_to_yolo(coco_file_path, args.output_dir)