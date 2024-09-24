import os
import nibabel as nib
import numpy as np
import argparse

def process_nii_files(base_dir, output_dir):
    # 定义需要处理的子目录列表
    sub_dirs = ['imagesTr', 'imagesTs', 'labelsTr', 'labelsTs']

    for sub_dir in sub_dirs:
        current_dir = os.path.join(base_dir, sub_dir)

        # 确保输出目录存在
        output_sub_dir = os.path.join(output_dir, sub_dir)
        os.makedirs(output_sub_dir, exist_ok=True)

        # 遍历当前子目录下的所有nii.gz文件
        for file_name in os.listdir(current_dir):
            if file_name.endswith('.nii.gz'):
                file_path = os.path.join(current_dir, file_name)

                # 加载NIfTI文件
                image = nib.load(file_path)

                # 直接使用image.dataobj来保持原始数据类型，保持数据的原始类型不变。
                image_data = np.asanyarray(image.dataobj)

                # 横断面向右旋转90度（逆时针旋转）
                rotated_image_data = np.rot90(image_data, k=-1, axes=(0, 1))

                # 左右翻转
                flipped_image_data = np.flip(rotated_image_data, axis=1)

                # 创建新的NIfTI图像并保存，确保使用与原始图像相同的dtype
                new_image = nib.Nifti1Image(flipped_image_data.astype(image_data.dtype), affine=image.affine)
                output_file_path = os.path.join(output_sub_dir, file_name)
                nib.save(new_image, output_file_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process NIfTI files for RibFrac dataset")
    parser.add_argument('--data-path', required=True, help="Path to the dataset root")
    parser.add_argument('--output-dir', required=True, help="Output directory for processed data")

    args = parser.parse_args()

    process_nii_files(args.data_path, args.output_dir)
