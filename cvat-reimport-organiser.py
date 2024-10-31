import os
import shutil
import zipfile

"""
CVAT Dataset Organization Script
-------------------------------
Purpose: This script organizes image and annotation files into a specific structure
required for import into CVAT (Computer Vision Annotation Tool). It processes a directory
containing image classes, copies relevant files, maintains annotations, and packages
everything into a zip file ready for CVAT import.

The script expects an input directory with the following structure:
    input_dir/
        obj.data
        obj.names
        class_id_folders/
            images.jpg
            annotations.txt
"""

def organize_files_for_cvat(input_dir, output_zip):
    """
    Organizes files from input directory into CVAT-compatible format and creates a zip archive.
    
    Args:
        input_dir (str): Path to input directory containing class folders and config files
        output_zip (str): Path where the output zip file should be created
    """
    # Create a temporary directory for organizing files before zipping
    temp_dir = 'temp_cvat_import'
    os.makedirs(temp_dir, exist_ok=True)

    # Copy configuration files needed for CVAT
    shutil.copy(os.path.join(input_dir, 'obj.data'), temp_dir)
    shutil.copy(os.path.join(input_dir, 'obj.names'), temp_dir)

    # Create directory for storing images and annotations
    obj_subset_dir = os.path.join(temp_dir, 'obj_subset_data')
    os.makedirs(obj_subset_dir, exist_ok=True)

    # List to store paths for train.txt
    train_txt_content = []

    # Process each class directory
    for class_id in os.listdir(input_dir):
        class_dir = os.path.join(input_dir, class_id)
        # Only process numbered directories above 1
        if os.path.isdir(class_dir) and class_id.isdigit() and int(class_id) > 1:
            # Get all image files in the current class directory
            image_files = [f for f in os.listdir(class_dir) 
                         if f.endswith(('.jpg', '.jpeg', '.png'))]
            
            for image_filename in image_files:
                # Process each image and its corresponding annotation
                src_image_path = os.path.join(class_dir, image_filename)
                dst_image_path = os.path.join(obj_subset_dir, image_filename)
                
                # Copy image file to destination
                shutil.copy(src_image_path, dst_image_path)
                train_txt_content.append(f'obj_subset_data/{image_filename}')

                # Handle annotation file
                anno_filename = os.path.splitext(image_filename)[0] + '.txt'
                src_anno_path = os.path.join(class_dir, anno_filename)
                dst_anno_path = os.path.join(obj_subset_dir, anno_filename)

                if os.path.exists(src_anno_path):
                    # Copy existing annotation file if it exists
                    shutil.copy(src_anno_path, dst_anno_path)
                else:
                    # Create empty annotation file if none exists
                    print(f"Creating empty annotation file for {image_filename}")
                    with open(dst_anno_path, 'w') as f:
                        pass

    # Create train.txt with list of all images
    with open(os.path.join(temp_dir, 'train.txt'), 'w') as f:
        f.write('\n'.join(train_txt_content))

    # Create zip archive of the organized files
    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(temp_dir):
            for file in files:
                file_path = os.path.join(root, file)
                # Preserve relative paths in zip file
                arcname = os.path.relpath(file_path, temp_dir)
                zipf.write(file_path, arcname)

    # Clean up temporary directory
    shutil.rmtree(temp_dir)

    # Print summary of operations
    print(f"Zip file created successfully: {output_zip}")
    print(f"Created {len(train_txt_content)} image entries in train.txt")

# Example usage with specific paths
input_directory = "C:/Users/neez/OneDrive/Uni/VLS301/Completed_Annotations/Waych1fp20s/cropped_v4"
output_zip_file = os.path.join("C:/Users/neez/OneDrive/Uni/VLS301/Completed_Annotations/Waych1fp20s/", "archive.zip")
organize_files_for_cvat(input_directory, output_zip_file)
