import xml.etree.ElementTree as ET
import os
from collections import defaultdict
import argparse

def cvat_xml_to_yolo(xml_file, output_dir):
    # Parse the XML file
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # Get image size
    size = root.find('.//task/size').text
    total_frames = int(size)

    # Get image dimensions
    width = int(root.find('.//image').attrib['width'])
    height = int(root.find('.//image').attrib['height'])

    # Create a dictionary to store labels and their corresponding IDs
    labels = {}
    for i, label in enumerate(root.findall('.//label/name')):
        labels[label.text] = i

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Dictionary to store annotations for each frame
    frame_annotations = defaultdict(list)

    # Iterate through all images (frames)
    for image in root.findall('.//image'):
        frame = int(image.attrib['id'])
        image_name = image.attrib['name']
        
        # Iterate through all polygon annotations for this image
        for polygon in image.findall('.//polygon'):
            label = polygon.attrib['label']
            points = polygon.attrib['points'].split(';')
            
            # Convert points to YOLO format
            x_coords = [float(p.split(',')[0]) for p in points]
            y_coords = [float(p.split(',')[1]) for p in points]
            
            x_center = sum(x_coords) / len(x_coords) / width
            y_center = sum(y_coords) / len(y_coords) / height
            bbox_width = (max(x_coords) - min(x_coords)) / width
            bbox_height = (max(y_coords) - min(y_coords)) / height
            
            # Store YOLO format annotation
            frame_annotations[frame].append(f"{labels[label]} {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}")

    # Write annotations to files
    for frame in range(total_frames):
        output_file = os.path.join(output_dir, f"frame_{frame:04d}.txt")
        with open(output_file, 'w') as f:
            if frame in frame_annotations:
                f.write('\n'.join(frame_annotations[frame]))

    print(f"Conversion complete. YOLO format annotations saved in {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert CVAT XML annotations to YOLO format")
    parser.add_argument("input_dir", help="Directory containing the CVAT XML file")
    parser.add_argument("output_dir", help="Directory to save YOLO format annotations")
    args = parser.parse_args()

    xml_file_path = os.path.join(args.input_dir, "annotations.xml")
    cvat_xml_to_yolo(xml_file_path, args.output_dir)
