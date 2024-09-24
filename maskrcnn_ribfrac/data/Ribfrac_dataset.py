import os
import numpy as np
import nibabel as nib
from PIL import Image
import pandas as pd
from skimage import measure
import SimpleITK as sitk
import json
import argparse

def read_label_info(csv_file_path):
    df = pd.read_csv(csv_file_path)
    label_info = {(row['public_id'], row['label_id']): row['label_code'] for index, row in df.iterrows()}
    return label_info

def windowing(image, window_center, window_width):
    min_window = window_center - window_width // 2
    max_window = window_center + window_width // 2
    windowed_image = np.clip(image, min_window, max_window)
    windowed_image = 255 * (windowed_image - min_window) / (max_window - min_window)
    return windowed_image.astype(np.uint8)

def create_directories(output_dir):
    directories = ["train", "val", "annotations"]
    for directory in directories:
        os.makedirs(os.path.join(output_dir, directory), exist_ok=True)

def extract_boxes_and_labels(segmentation_object_image, labels_info, patient_id, include_negative_one=True, connectivity=2):
    boxes = []
    labels = []
    segmentations = []
    unique_labels = np.unique(segmentation_object_image)[1:]

    for label_id in unique_labels:
        label_code = labels_info.get((patient_id, label_id), -1)
        if label_code == -1 and not include_negative_one:
            continue
        mask = segmentation_object_image == label_id
        labeled_mask = measure.label(mask, connectivity=connectivity)
        regions = measure.regionprops(labeled_mask)

        for region in regions:
            if region.area >= 100:
                minr, minc, maxr, maxc = region.bbox
                boxes.append([minc, minr, maxc, maxr])
                labels.append(1)
                contours = measure.find_contours(mask, 0.5)
                for contour in contours:
                    contour = np.flip(contour, axis=1)
                    segmentation = contour.ravel().tolist()
                    if len(segmentation) >= 6:
                        segmentations.append(segmentation)

    return boxes, labels, segmentations

def create_coco_annotation(filename, boxes, labels, segmentations, annotations, image_id, annotation_id):
    for box, label_code, segmentation in zip(boxes, labels, segmentations):
        x_min, y_min, x_max, y_max = box
        width = x_max - x_min
        height = y_max - y_min

        annotation = {
            "id": annotation_id,
            "image_id": image_id,
            "category_id": label_code,
            "bbox": [x_min, y_min, width, height],
            "area": width * height,
            "iscrowd": 0,
            "segmentation": [segmentation]
        }
        annotations.append(annotation)
        annotation_id += 1

    return annotation_id

def process_nii_to_coco(dataset_root, output_dir, csv_file_path, image_subdir, label_subdir, image_filenames, coco_annotations, image_id, annotation_id, subset):
    labels_info = read_label_info(csv_file_path)
    patient_ids = pd.read_csv(csv_file_path)['public_id'].unique()
    for patient_id in patient_ids:
        resampled_image_path = os.path.join(dataset_root, image_subdir, f"{patient_id}-image_resampled.nii.gz")
        resampled_label_path = os.path.join(dataset_root, label_subdir, f"{patient_id}-label_resampled.nii.gz")

        image = nib.load(resampled_image_path).get_fdata()
        label = nib.load(resampled_label_path).get_fdata()

        for z in range(image.shape[2]):
            unique_labels = np.unique(label[:, :, z])
            if not np.any(unique_labels[unique_labels != 0]):
                continue

            slice_filename = f"{patient_id}_slice_{z}.jpg"
            jpeg_path = os.path.join(output_dir, subset, slice_filename)

            relevant_labels = [labels_info.get((patient_id, int(label_id)), -1) for label_id in unique_labels]
            if not any(code in [1, 2, 3, 4, -1] for code in relevant_labels):
                continue

            slice_image = windowing(image[:, :, z], 400, 1800)
            pil_image = Image.fromarray(slice_image)
            pil_image.save(jpeg_path)

            image_filenames[subset].append({
                "id": image_id,
                "file_name": slice_filename,
                "width": pil_image.width,
                "height": pil_image.height
            })

            segmentation_image = np.zeros_like(label[:, :, z], dtype=np.uint8)
            valid_instance_count = 0
            for label_id in unique_labels:
                if label_id == 0:
                    continue
                label_code = labels_info.get((patient_id, label_id), -1)
                mask = label[:, :, z] == label_id
                segmentation_image[mask] = label_id
                valid_instance_count += 1

            boxes, labels, segmentations = extract_boxes_and_labels(segmentation_image, labels_info, patient_id, include_negative_one=True)

            if len(boxes) != valid_instance_count:
                print(f"Warning: num_boxes:{len(boxes)} and num_instances:{valid_instance_count} do not correspond. "
                      f"Removing {slice_filename} and its segmentation.")
                os.remove(jpeg_path)
                # 移除相应的标注信息
                image_filenames[subset] = [img for img in image_filenames[subset] if img["file_name"] != slice_filename]
                coco_annotations[subset] = [ann for ann in coco_annotations[subset] if ann["image_id"] != image_id]
                continue

            if boxes:
                annotation_id = create_coco_annotation(slice_filename, boxes, labels, segmentations, coco_annotations[subset], image_id, annotation_id)
                image_id += 1
            else:
                os.remove(jpeg_path)
                image_filenames[subset] = [img for img in image_filenames[subset] if img["file_name"] != slice_filename]
                print(f"Removed {slice_filename} due to no detected boxes.")

    return image_id, annotation_id

def resample_nii_file(input_path, output_path, new_spacing=[1.0, 1.0, 1.0]):
    reader = sitk.ImageFileReader()
    reader.SetFileName(input_path)
    image = reader.Execute()

    original_size = image.GetSize()
    original_spacing = image.GetSpacing()

    new_size = [int(round(osz * osp / ns)) for osz, osp, ns in zip(original_size, original_spacing, new_spacing)]

    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(new_spacing)
    resample.SetSize(new_size)
    resample.SetOutputOrigin(image.GetOrigin())
    resample.SetOutputDirection(image.GetDirection())
    resample.SetInterpolator(sitk.sitkNearestNeighbor)

    resampled_image = resample.Execute(image)
    sitk.WriteImage(resampled_image, output_path)

def process_dataset_for_both_sets(dataset_root, output_dir, resample_spacing=[1.0, 1.0, 1.0]):
    create_directories(output_dir)

    image_filenames = {"train": [], "val": []}
    coco_annotations = {"train": [], "val": []}
    categories = [
        {"id": 1, "name": "rib_fracture"}
    ]

    coco_output = {
        "categories": categories
    }

    csv_file_paths = {
        'train': os.path.join(dataset_root, 'ribfrac-train-info.csv'),
        'val': os.path.join(dataset_root, 'ribfrac-val-info.csv')
    }

    image_id = 0
    annotation_id = 0

    for subset, csv_file_path in csv_file_paths.items():
        df = pd.read_csv(csv_file_path)
        image_subdir = "imagesTr" if subset == 'train' else "imagesTs"
        label_subdir = "labelsTr" if subset == 'train' else "labelsTs"

        for index, row in df.iterrows():
            patient_id = row['public_id']
            image_path = os.path.join(dataset_root, image_subdir, f"{patient_id}-image.nii.gz")
            label_path = os.path.join(dataset_root, label_subdir, f"{patient_id}-label.nii.gz")
            resampled_image_path = image_path.replace(".nii.gz", "_resampled.nii.gz")
            resampled_label_path = label_path.replace(".nii.gz", "_resampled.nii.gz")

            if not os.path.exists(resampled_image_path) or not os.path.exists(resampled_label_path):
                resample_nii_file(image_path, resampled_image_path, resample_spacing)
                resample_nii_file(label_path, resampled_label_path, resample_spacing)

        image_id, annotation_id = process_nii_to_coco(dataset_root, output_dir, csv_file_path, image_subdir, label_subdir, image_filenames, coco_annotations, image_id, annotation_id, subset)

    # 同步检查和清理无效标注
    for subset in ["train", "val"]:
        valid_image_ids = set(img["id"] for img in image_filenames[subset])
        coco_annotations[subset] = [ann for ann in coco_annotations[subset] if ann["image_id"] in valid_image_ids]

    for subset in ["train", "val"]:
        with open(os.path.join(output_dir, "annotations", f"instances_{subset}.json"), 'w') as f:
            json.dump({"images": image_filenames[subset], "annotations": coco_annotations[subset], "categories": coco_output["categories"]}, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process Ribfrac dataset to COCO format")
    parser.add_argument('--data-path', required=True, help="Path to the dataset root")
    parser.add_argument('--output-dir', required=True, help="Output directory for processed data")
    parser.add_argument('--spacing', nargs=3, type=float, default=[0.74749817, 0.74749817, 1.13184524], help="Resampling spacing for NIfTI files")

    args = parser.parse_args()

    process_dataset_for_both_sets(args.data_path, args.output_dir, args.spacing)
