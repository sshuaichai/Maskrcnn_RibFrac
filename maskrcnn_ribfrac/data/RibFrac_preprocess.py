import os
import nibabel as nib
import numpy as np
import argparse

def process_nii_files(base_dir, output_dir):
    sub_dirs = ['imagesTr', 'imagesTs', 'labelsTr', 'labelsTs']

    for sub_dir in sub_dirs:
        current_dir = os.path.join(base_dir, sub_dir)


        output_sub_dir = os.path.join(output_dir, sub_dir)
        os.makedirs(output_sub_dir, exist_ok=True)

        for file_name in os.listdir(current_dir):
            if file_name.endswith('.nii.gz'):
                file_path = os.path.join(current_dir, file_name)

                image = nib.load(file_path)

                image_data = np.asanyarray(image.dataobj)

                rotated_image_data = np.rot90(image_data, k=-1, axes=(0, 1))

                flipped_image_data = np.flip(rotated_image_data, axis=1)

                new_image = nib.Nifti1Image(flipped_image_data.astype(image_data.dtype), affine=image.affine)
                output_file_path = os.path.join(output_sub_dir, file_name)
                nib.save(new_image, output_file_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process NIfTI files for RibFrac dataset")
    parser.add_argument('--data-path', required=True, help="Path to the dataset root")
    parser.add_argument('--output-dir', required=True, help="Output directory for processed data")

    args = parser.parse_args()

    process_nii_files(args.data_path, args.output_dir)
