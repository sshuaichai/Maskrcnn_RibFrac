import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import os
import time
import json
import numpy as np
from PIL import Image
import torch
import pydicom
from maskrcnn_ribfrac.network_files import MaskRCNN

import warnings
warnings.filterwarnings("ignore", category=UserWarning,
                        message="torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument.")

def ensure_dir_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def dicom_to_image(dicom_file, default_window_center=40, default_window_width=350):
    dicom = pydicom.dcmread(dicom_file)
    pixel_array = dicom.pixel_array

    window_center = getattr(dicom, 'WindowCenter', default_window_center)
    window_width = getattr(dicom, 'WindowWidth', default_window_width)
    window_center = window_center[0] if isinstance(window_center, pydicom.multival.MultiValue) else window_center
    window_width = window_width[0] if isinstance(window_width, pydicom.multival.MultiValue) else window_width
    intercept = getattr(dicom, 'RescaleIntercept', 0)
    slope = getattr(dicom, 'RescaleSlope', 1)

    pixel_array = pixel_array * slope + intercept
    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2
    pixel_array = np.clip(pixel_array, img_min, img_max)
    pixel_array = ((pixel_array - img_min) / (img_max - img_min) * 255.0).astype(np.uint8)

    img = Image.fromarray(pixel_array).convert('RGB')
    return img, dicom, img_min, img_max

def create_model(model_id="maskrcn50", num_classes=2, box_thresh=0.3):
    if model_id == "maskrcn50":
        from maskrcnn_ribfrac.backbone import resnet50_fpn_backbone
        backbone = resnet50_fpn_backbone()
    elif model_id == "maskrcn101":
        from maskrcnn_ribfrac.backbone import resnet101_fpn_backbone
        backbone = resnet101_fpn_backbone()
    elif model_id == "maskrcn152":
        from maskrcnn_ribfrac.backbone import resnet152_fpn_backbone
        backbone = resnet152_fpn_backbone()
    else:
        raise ValueError(f"Unsupported model_id: {model_id}")

    model = MaskRCNN(backbone,
                     num_classes=num_classes,
                     rpn_score_thresh=box_thresh,
                     box_score_thresh=box_thresh)
    return model

def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


def save_dicom(img, dicom, output_path):
    pixel_array = np.array(img)
    dicom.SamplesPerPixel = 3
    dicom.PhotometricInterpretation = "RGB"
    dicom.PlanarConfiguration = 0
    dicom.BitsAllocated = 8
    dicom.BitsStored = 8
    dicom.HighBit = 7
    dicom.Rows, dicom.Columns, _ = pixel_array.shape
    dicom.PixelData = pixel_array.tobytes()
    dicom.save_as(output_path)


def save_jpg(img, output_path):
    img.save(output_path)


def merge_boxes_and_masks(predict_boxes, predict_masks, predict_scores):
    merged_boxes = []
    merged_masks = []
    merged_scores = []

    used = np.zeros(len(predict_boxes), dtype=bool)

    for i in range(len(predict_boxes)):
        if used[i]:
            continue

        current_box = predict_boxes[i]
        current_mask = predict_masks[i]
        current_score = predict_scores[i]
        used[i] = True

        # 合并所有与 current_box 重叠的框
        while True:
            merged = False
            for j in range(i + 1, len(predict_boxes)):
                if used[j]:
                    continue
                next_box = predict_boxes[j]
                if box_overlap(current_box, next_box):
                    current_box = merge_two_boxes(current_box, next_box)
                    current_mask = np.maximum(current_mask, predict_masks[j])  # 合并mask
                    current_score = (current_score + predict_scores[j]) / 2  # 平均得分
                    used[j] = True
                    merged = True

            if not merged:
                break

        merged_boxes.append(current_box)
        merged_masks.append(current_mask)
        merged_scores.append(current_score)

    return np.array(merged_boxes), np.array(merged_masks), np.array(merged_scores)


def box_overlap(box1, box2):
    # 计算两个框的重叠区域
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    if x2 <= x1 or y2 <= y1:
        return False  # 没有重叠

    return True


def merge_two_boxes(box1, box2):
    # 将两个框合并为一个更大的框
    x1 = min(box1[0], box2[0])
    y1 = min(box1[1], box2[1])
    x2 = max(box1[2], box2[2])
    y2 = max(box1[3], box2[3])

    return [x1, y1, x2, y2]


def load_image(file_path):
    """加载 DICOM 或 JPG/PNG 图像"""
    ext = os.path.splitext(file_path)[-1].lower()

    if ext in ['.dcm', '.ima']:  # DICOM 格式
        return dicom_to_image(file_path)
    elif ext in ['.jpg', '.jpeg', '.png']:  # JPG 或 PNG 格式
        img = Image.open(file_path).convert('RGB')
        return img, None, None, None  # 返回 None 表示非 DICOM 图像
    else:
        raise ValueError("Unsupported file format. Please use DICOM or JPG/PNG.")
