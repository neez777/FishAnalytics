import os
import shutil
import zipfile

def organize_files_for_cvat(input_dir, output_zip):
    # Create a temporary directory for organizing files
    temp_dir = 'temp_cvat_import'
    os.makedirs(temp_dir, exist_ok=True)

    # Copy obj.data and obj.names
    shutil.copy(os.path.join(input_dir, 'obj.data'), temp_dir)
    shutil.copy(os.path.join(input_dir, 'obj.names'), temp_dir)

    # Create obj_subset_data directory (without angle brackets)
    obj_subset_dir = os.path.join(temp_dir, 'obj_subset_data')
    os.makedirs(obj_subset_dir, exist_ok=True)

    # Gather image and annotation files
    train_txt_content = []
    for class_id in os.listdir(input_dir):
        class_dir = os.path.join(input_dir, class_id)
        if os.path.isdir(class_dir) and class_id.isdigit() and int(class_id) > 1:
            # First, find all image files
            image_files = [f for f in os.listdir(class_dir) 
                         if f.endswith(('.jpg', '.jpeg', '.png'))]
            
            for image_filename in image_files:
                # Copy image file
                src_image_path = os.path.join(class_dir, image_filename)
                dst_image_path = os.path.join(obj_subset_dir, image_filename)
                shutil.copy(src_image_path, dst_image_path)
                train_txt_content.append(f'obj_subset_data/{image_filename}')

                # Check for corresponding annotation file
                anno_filename = os.path.splitext(image_filename)[0] + '.txt'
                src_anno_path = os.path.join(class_dir, anno_filename)
                dst_anno_path = os.path.join(obj_subset_dir, anno_filename)

                if os.path.exists(src_anno_path):
                    # Copy existing annotation file
                    shutil.copy(src_anno_path, dst_anno_path)
                else:
                    # Create empty annotation file
                    print(f"Creating empty annotation file for {image_filename}")
                    with open(dst_anno_path, 'w') as f:
                        pass  # Creates an empty file

    # Create train.txt
    with open(os.path.join(temp_dir, 'train.txt'), 'w') as f:
        f.write('\n'.join(train_txt_content))

    # Create zip file
    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(temp_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, temp_dir)
                zipf.write(file_path, arcname)

    # Clean up temporary directory
    shutil.rmtree(temp_dir)

    print(f"Zip file created successfully: {output_zip}")
    print(f"Created {len(train_txt_content)} image entries in train.txt")

# Usage with environment variables
input_directory = "C:/Users/neez/OneDrive/Uni/VLS301/Completed_Annotations/Waych1fp20s/cropped_v4"
output_zip_file = os.path.join("C:/Users/neez/OneDrive/Uni/VLS301/Completed_Annotations/Waych1fp20s/", "archive.zip")

organize_files_for_cvat(input_directory, output_zip_file)