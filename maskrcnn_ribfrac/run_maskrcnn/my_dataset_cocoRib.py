import os
import json

import torch
from PIL import Image
import torch.utils.data as data
from pycocotools.coco import COCO
from maskrcnn_ribfrac.utils import coco_remove_images_without_annotations, convert_coco_poly_mask


class CocoDetection(data.Dataset):
    """`MS Coco Detection <https://cocodataset.org/>`_ Dataset.

    Args:
        root (string): Root directory where images are downloaded to.
        dataset (string): train or val.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
    """

    def __init__(self, root, dataset="train", transforms=None):
        super(CocoDetection, self).__init__()
        assert dataset in ["train", "val"], 'dataset must be in ["train", "val"]'
        anno_file = f"instances_{dataset}.json"
        assert os.path.exists(root), "file '{}' does not exist.".format(root)
        self.img_root = os.path.join(root, dataset)
        assert os.path.exists(self.img_root), "path '{}' does not exist.".format(self.img_root)
        self.anno_path = os.path.join(root, "annotations", anno_file)
        assert os.path.exists(self.anno_path), "file '{}' does not exist.".format(self.anno_path)

        self.mode = dataset
        self.transforms = transforms
        self.coco = COCO(self.anno_path)

        data_classes = dict([(v["id"], v["name"]) for k, v in self.coco.cats.items()])
        max_index = max(data_classes.keys())
        coco_classes = {}
        for k in range(1, max_index + 1):
            if k in data_classes:
                coco_classes[k] = data_classes[k]
            else:
                coco_classes[k] = "N/A"

        if dataset == "train":
            json_str = json.dumps(coco_classes, indent=4)
            with open("cocorib_indices.json", "w") as f:
                f.write(json_str)

        self.coco_classes = coco_classes

        ids = list(sorted(self.coco.imgs.keys()))
        if dataset == "train":
            valid_ids = coco_remove_images_without_annotations(self.coco, ids)
            self.ids = valid_ids
        else:
            self.ids = ids

    def parse_targets(self,
                      img_id: int,
                      coco_targets: list,
                      w: int = None,
                      h: int = None):
        assert w > 0
        assert h > 0

        anno = [obj for obj in coco_targets if obj['iscrowd'] == 0]

        boxes = [obj["bbox"] for obj in anno]

        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        # [xmin, ymin, w, h] -> [xmin, ymin, xmax, ymax]
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] for obj in anno])

        segmentations = [obj["segmentation"] for obj in anno]
        if any(segmentations):
            masks = convert_coco_poly_mask(segmentations, h, w)
        else:
            masks = torch.zeros((0, h, w), dtype=torch.uint8)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        if masks.size(0) > 0:
            masks = masks[keep]
        area = area[keep]
        iscrowd = iscrowd[keep]

        if masks.size(0) == 0:
            print(f"Error: Image ID {img_id} has no valid masks.")
            print(f"Shapes of boxes: {boxes.shape}")
            print(f"Shapes of masks: {masks.shape}")
            raise ValueError(f"Image ID {img_id} has no valid masks.")

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        target["masks"] = masks
        target["image_id"] = torch.tensor([img_id])

        # for conversion to coco api
        target["area"] = area
        target["iscrowd"] = iscrowd

        return target

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: Tuple (image, target). target is the object returned by ``coco.loadAnns``.
        """
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        coco_target = coco.loadAnns(ann_ids)

        path = coco.loadImgs(img_id)[0]['file_name']
        img = Image.open(os.path.join(self.img_root, path)).convert('RGB')

        w, h = img.size
        target = self.parse_targets(img_id, coco_target, w, h)
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.ids)

    def get_height_and_width(self, index):
        coco = self.coco
        img_id = self.ids[index]

        img_info = coco.loadImgs(img_id)[0]
        w = img_info["width"]
        h = img_info["height"]
        return h, w

    @staticmethod
    def collate_fn(batch):
        return tuple(zip(*batch))


# import os
# import json
# import random
# import torch
# import matplotlib.pyplot as plt
# import torchvision.transforms as ts
# from PIL import Image
# import transforms
# from torchvision.transforms import InterpolationMode
# import numpy as np
#
# random.seed(1)
# np.random.seed(1)
# torch.manual_seed(1)
#
# # Load category index
# category_index = {}
# try:
#     json_file = open('..\cocorib_indices.json', 'r')
#     category_index = json.load(json_file)
# except Exception as e:
#     print(e)
#     exit(-1)
#
# data_transform = {
#         "train": transforms.Compose([
#             transforms.ToTensor(),
#             transforms.ToDtype(torch.float32, scale=True),
#             transforms.RandomHorizontalFlip(prob=0.5),
#             transforms.RandomVerticalFlip(prob=0.5),
#             transforms.RandomRotateCounterClockwise90(prob=0.5),
#             transforms.RandomRotatealittle(angle_range=(-90, 90), prob=0.5),
#             transforms.RandomIoUCrop(),
#             transforms.RandomPhotometricDistort(
#                 contrast=(0.5, 1.5),
#                 saturation=(0.5, 1.5),
#                 hue=(-0.05, 0.05),
#                 brightness=(0.875, 1.125),
#                 p=0.5
#             ),
#             transforms.ScaleJitter(
#                 target_size=(512, 512),
#                 scale_range=(0.7, 1.3),
#                 interpolation=InterpolationMode.BILINEAR,
#                 antialias=True
#             ),
#             transforms.RandomZoomOut(
#                 fill=[0, 0, 0],
#                 side_range=(1, 1.5),
#                 p=0.5
#             ),
#             transforms.RandomShortestSize(
#                 min_size=[512],
#                 max_size=1024,
#                 interpolation=InterpolationMode.BILINEAR,
#                 antialias=True
#             ),
#         ]),
#         "val": transforms.Compose([
#             transforms.ToTensor(),
#             transforms.ToDtype(torch.float32, scale=True)
#         ])
# }
#
# def visualize_samples(data_set, category_index, sample_count=30, cols=5):
#     rows = (sample_count + cols - 1) // cols
#     fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
#
#     for idx, index in enumerate(random.sample(range(0, len(data_set)), k=sample_count)):
#         img, target = data_set[index]
#         img = ts.ToPILImage()(img)
#
#         from draw_box_utils2 import draw_objs
#         plot_img = draw_objs(
#             img,
#             target["boxes"].numpy(),
#             target["labels"].numpy(),
#             np.ones(target["labels"].shape[0]),
#             target["masks"].numpy(),
#             category_index=category_index,
#             box_thresh=0.5,
#             mask_thresh=0.5,
#             line_thickness=1,
#             font_size=10
#         )
#
#         ax = axes[idx // cols, idx % cols]
#         ax.imshow(plot_img)
#         ax.axis('off')
#
#     for i in range(sample_count, rows * cols):
#         fig.delaxes(axes.flatten()[i])
#
#     plt.tight_layout()
#     plt.show()
#
# train_data_set = CocoDetection(root=r"data_path", dataset="train",
#                                transforms=data_transform["train"])
# print("Dataset size:", len(train_data_set))
#
# visualize_samples(train_data_set, category_index)
