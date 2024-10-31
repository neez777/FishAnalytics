import xml.etree.ElementTree as ET
import os
from collections import defaultdict
import argparse

"""
CVAT XML to YOLO Format Converter
--------------------------------
Purpose: This script converts annotation files from CVAT's XML format to YOLO format.
It processes polygon annotations from CVAT and converts them into bounding box
annotations in YOLO format (class_id, x_center, y_center, width, height).

The script handles:
- Parsing CVAT XML export files
- Converting polygon coordinates to bounding box coordinates
- Normalizing coordinates to YOLO format (relative to image dimensions)
- Generating individual annotation files for each frame
"""

def cvat_xml_to_yolo(xml_file, output_dir):
    """
    Converts CVAT XML annotations to YOLO format annotation files.
    
    Args:
        xml_file (str): Path to the input CVAT XML file
        output_dir (str): Directory where YOLO format annotations will be saved
    
    The function creates one .txt file per frame, with each line in YOLO format:
    <class_id> <x_center> <y_center> <width> <height>
    where all coordinates are normalized between 0 and 1
    """
    # Parse the XML file using ElementTree
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # Extract total number of frames from XML
    size = root.find('.//task/size').text
    total_frames = int(size)

    # Get image dimensions for coordinate normalization
    width = int(root.find('.//image').attrib['width'])
    height = int(root.find('.//image').attrib['height'])

    # Create a mapping of label names to numeric IDs
    labels = {}
    for i, label in enumerate(root.findall('.//label/name')):
        labels[label.text] = i

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Use defaultdict to collect all annotations for each frame
    frame_annotations = defaultdict(list)

    # Process each image (frame) in the XML
    for image in root.findall('.//image'):
        frame = int(image.attrib['id'])
        image_name = image.attrib['name']
        
        # Process all polygon annotations in the current frame
        for polygon in image.findall('.//polygon'):
            label = polygon.attrib['label']
            # Split points string into individual coordinates
            points = polygon.attrib['points'].split(';')
            
            # Extract and convert x,y coordinates from points
            x_coords = [float(p.split(',')[0]) for p in points]
            y_coords = [float(p.split(',')[1]) for p in points]
            
            # Calculate bounding box center and dimensions
            # Normalize all values by dividing by image dimensions
            x_center = sum(x_coords) / len(x_coords) / width
            y_center = sum(y_coords) / len(y_coords) / height
            bbox_width = (max(x_coords) - min(x_coords)) / width
            bbox_height = (max(y_coords) - min(y_coords)) / height
            
            # Format and store YOLO annotation
            # Format: <class_id> <x_center> <y_center> <width> <height>
            frame_annotations[frame].append(
                f"{labels[label]} {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}"
            )

    # Write out annotation files for each frame
    for frame in range(total_frames):
        output_file = os.path.join(output_dir, f"frame_{frame:04d}.txt")
        with open(output_file, 'w') as f:
            if frame in frame_annotations:
                f.write('\n'.join(frame_annotations[frame]))

    print(f"Conversion complete. YOLO format annotations saved in {output_dir}")

if __name__ == "__main__":
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description="Convert CVAT XML annotations to YOLO format")
    parser.add_argument("input_dir", help="Directory containing the CVAT XML file")
    parser.add_argument("output_dir", help="Directory to save YOLO format annotations")
    
    # Parse command line arguments
    args = parser.parse_args()
    
    # Construct path to XML file and perform conversion
    xml_file_path = os.path.join(args.input_dir, "annotations.xml")
    cvat_xml_to_yolo(xml_file_path, args.output_dir)
