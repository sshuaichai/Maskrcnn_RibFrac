import torch
import torch.utils.data
from pycocotools import mask as coco_mask
from pycocotools.coco import COCO
import numpy as np


def coco_remove_images_without_annotations(dataset, ids):
    """
    删除coco数据集中没有目标，或者目标面积非常小的数据
    refer to:
    https://github.com/pytorch/vision/blob/master/references/detection/coco_utils.py
    :param dataset:
    :param cat_list:
    :return:
    """
    def _has_only_empty_bbox(anno):
        return all(any(o <= 1 for o in obj["bbox"][2:]) for obj in anno)

    def _has_valid_annotation(anno):
        # if it's empty, there is no annotation
        if len(anno) == 0:
            return False
        # if all boxes have close to zero area, there is no annotation
        if _has_only_empty_bbox(anno):
            return False

        return True

    valid_ids = []
    for ds_idx, img_id in enumerate(ids):
        ann_ids = dataset.getAnnIds(imgIds=img_id, iscrowd=None)
        anno = dataset.loadAnns(ann_ids)

        if _has_valid_annotation(anno):
            valid_ids.append(img_id)

    return valid_ids


def convert_coco_poly_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        # 如果mask为空，则说明没有目标，直接返回数值为0的mask
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


def convert_to_coco_api(self):
    coco_ds = COCO()
    ann_id = 1
    dataset = {"images": [], "categories": [], "annotations": []}
    categories = set()
    for img_idx in range(len(self)):
        targets, h, w = self.get_annotations(img_idx)
        img_id = targets["image_id"].item()
        img_dict = {"id": img_id, "height": h, "width": w}
        dataset["images"].append(img_dict)

        bboxes = targets["boxes"].clone()
        bboxes[:, 2:] -= bboxes[:, :2]  # Convert (x_min, y_min, x_max, y_max) to (x_min, y_min, w, h)
        bboxes = bboxes.tolist()
        labels = targets["labels"].tolist()
        areas = targets["area"].tolist()
        iscrowd = targets["iscrowd"].tolist()

        if "masks" in targets:
            masks = targets["masks"]
            if masks.dim() == 4:
                if masks.size(1) != 1:
                    # 假设我们只关心第一个通道，这需要根据您的具体情况进行调整
                    masks = masks[:, 0, :, :].unsqueeze(1)  # 选择第一个通道并保持维度为[N, 1, H, W]
                masks = masks.squeeze(1)  # 从[N, 1, H, W]转换为[N, H, W]
            elif masks.dim() != 3:
                raise ValueError(
                    f"Masks tensor has an unexpected shape: {masks.shape}. Expected 3 dimensions after squeeze.")


        num_objs = len(bboxes)
        for i in range(num_objs):
            ann = {
                "image_id": img_id,
                "bbox": bboxes[i],
                "category_id": labels[i],
                "area": areas[i],
                "iscrowd": iscrowd[i],
                "id": ann_id
            }
            categories.add(labels[i])

            if "masks" in targets:
                # 确保掩码是Fortran连续的
                fortran_mask = np.asfortranarray(masks[i].cpu().numpy())
                ann["segmentation"] = coco_mask.encode(fortran_mask)
            dataset["annotations"].append(ann)
            ann_id += 1

    dataset["categories"] = [{"id": i} for i in sorted(categories)]
    coco_ds.dataset = dataset
    coco_ds.createIndex()
    return coco_ds
