import random
from typing import Dict, List, Optional, Tuple, Union

import torch
import torchvision
from torch import nn, Tensor
from torchvision import ops
from torchvision.transforms import functional as F, InterpolationMode, transforms as T
import math

class Compose:
    """组合多个transform函数"""
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

class ToTensor(nn.Module):
    """将PIL图像转换为Tensor。"""
    def forward(
        self, image: Tensor, target: Optional[Dict[str, Tensor]] = None
    ) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        image = F.pil_to_tensor(image)  # 将PIL图像转换为Tensor
        return image, target  # 返回转换后的图像和未修改的目标

class ToDtype(nn.Module):
    """将图像Tensor转换为指定的数据类型。"""
    def __init__(self, dtype: torch.dtype, scale: bool = False) -> None:
        super().__init__()
        self.dtype = dtype  # 目标数据类型
        self.scale = scale  # 是否在转换时进行归一化处理

    def forward(
        self, image: Tensor, target: Optional[Dict[str, Tensor]] = None
    ) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        if not self.scale:
            return image.to(dtype=self.dtype), target  # 转换数据类型，不进行归一化
        image = F.convert_image_dtype(image, self.dtype)  # 转换数据类型并进行归一化
        return image, target


class RandomHorizontalFlip(object):
    """随机水平翻转图像以及bboxes"""
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            height, width = image.shape[-2:]
            image = image.flip(-1)  # 水平翻转图片
            bbox = target["boxes"]
            # bbox: xmin, ymin, xmax, ymax
            bbox[:, [0, 2]] = width - bbox[:, [2, 0]]  # 翻转对应bbox坐标信息
            target["boxes"] = bbox
            if "masks" in target:
                target["masks"] = target["masks"].flip(-1)
        return image, target


class RandomRotateCounterClockwise90(object):
    """随机逆时针旋转图像 90 度并相应调整目标检测框和掩码。"""
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image: Tensor, target: Optional[Dict[str, Tensor]] = None) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        if random.random() < self.prob:
            height, width = image.shape[-2:]

            # 旋转图像
            image = image.permute(0, 2, 1).flip(-1)

            if 'boxes' in target:
                # 对每个检测框（xmin, ymin, xmax, ymax），重新计算坐标
                bbox = target['boxes']
                ymin = bbox[:, 0]
                ymax = bbox[:, 2]
                xmin = height - bbox[:, 3]
                xmax = height - bbox[:, 1]

                # 更新检测框坐标
                target['boxes'] = torch.stack([xmin, ymin, xmax, ymax], dim=1)

            if 'masks' in target:
                # 旋转掩码
                target['masks'] = target['masks'].permute(0, 2, 1).flip(-1)

        return image, target

class RandomRotateClockwise90(object):
    """随机顺时针旋转图像 90 度，并相应调整目标检测框和掩码。"""
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image: Tensor, target: Optional[Dict[str, Tensor]] = None) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        if random.random() < self.prob:
            height, width = image.shape[-2:]

            # 顺时针旋转图像
            image = image.permute(0, 2, 1).flip(-2)

            if 'boxes' in target:
                # 对每个检测框（xmin, ymin, xmax, ymax），重新计算坐标
                bbox = target['boxes']
                xmin = bbox[:, 1]
                xmax = bbox[:, 3]
                ymin = width - bbox[:, 2]
                ymax = width - bbox[:, 0]

                # 更新检测框坐标
                target['boxes'] = torch.stack([xmin, ymin, xmax, ymax], dim=1)

            if 'masks' in target:
                # 顺时针旋转掩码
                target['masks'] = target['masks'].permute(0, 2, 1).flip(-2)

        return image, target


class RandomVerticalFlip(object):
    """随机垂直翻转图像及其标注"""
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            height = image.shape[1]  # 获取翻转前的图像高度，注意，这应在翻转前完成
            image = F.vflip(image)  # 垂直翻转图片
            if 'boxes' in target:
                # 更新 bbox: ymin, ymax
                ymin = height - target['boxes'][:, 3]
                ymax = height - target['boxes'][:, 1]
                target['boxes'][:, 1] = ymin
                target['boxes'][:, 3] = ymax
            if 'masks' in target:
                target['masks'] = target['masks'].flip(-2)  # 对应垂直翻转掩码
        return image, target


class RandomIoUCrop(nn.Module):
    def __init__(
        self,
        min_scale: float = 0.3,
        max_scale: float = 1.0,
        min_aspect_ratio: float = 0.5,
        max_aspect_ratio: float = 2.0,
        sampler_options: Optional[List[float]] = None,
        trials: int = 40,
    ):
        super().__init__()
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.min_aspect_ratio = min_aspect_ratio
        self.max_aspect_ratio = max_aspect_ratio
        if sampler_options is None:
            sampler_options = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
        self.options = sampler_options
        self.trials = trials

    def forward(
        self, image: Tensor, target: Optional[Dict[str, Tensor]] = None
    ) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        if target is None:
            raise ValueError("The targets can't be None for this transform.")

        if isinstance(image, torch.Tensor):
            if image.ndimension() not in {2, 3}:
                raise ValueError(f"image should be 2/3 dimensional. Got {image.ndimension()} dimensions.")
            elif image.ndimension() == 2:
                image = image.unsqueeze(0)

        _, orig_h, orig_w = F.get_dimensions(image)

        while True:
            idx = int(torch.randint(low=0, high=len(self.options), size=(1,)))
            min_jaccard_overlap = self.options[idx]
            if min_jaccard_overlap >= 1.0:
                return image, target

            for _ in range(self.trials):
                r = self.min_scale + (self.max_scale - self.min_scale) * torch.rand(2)
                new_w = int(orig_w * r[0])
                new_h = int(orig_h * r[1])
                aspect_ratio = new_w / new_h
                if not (self.min_aspect_ratio <= aspect_ratio <= self.max_aspect_ratio):
                    continue

                r = torch.rand(2)
                left = int((orig_w - new_w) * r[0])
                top = int((orig_h - new_h) * r[1])
                right = left + new_w
                bottom = top + new_h
                if left == right or top == bottom:
                    continue

                cx = 0.5 * (target["boxes"][:, 0] + target["boxes"][:, 2])
                cy = 0.5 * (target["boxes"][:, 1] + target["boxes"][:, 3])
                is_within_crop_area = (left < cx) & (cx < right) & (top < cy) & (cy < bottom)
                if not is_within_crop_area.any():
                    continue

                boxes = target["boxes"][is_within_crop_area]
                ious = torchvision.ops.boxes.box_iou(
                    boxes, torch.tensor([[left, top, right, bottom]], dtype=boxes.dtype, device=boxes.device)
                )
                if ious.max() < min_jaccard_overlap:
                    continue

                target["boxes"] = boxes
                target["labels"] = target["labels"][is_within_crop_area]
                # 关键改动
                # 裁剪后的masks处理：
                if "masks" in target:
                    target["masks"] = target["masks"][is_within_crop_area]
                    target["masks"] = target["masks"][:, top:bottom, left:right]

                target["boxes"][:, 0::2] -= left
                target["boxes"][:, 1::2] -= top
                target["boxes"][:, 0::2].clamp_(min=0, max=new_w)
                target["boxes"][:, 1::2].clamp_(min=0, max=new_h)
                image = F.crop(image, top, left, new_h, new_w)

                return image, target



class RandomPhotometricDistort(nn.Module):
    """随机调整图像的光度属性，如对比度、饱和度、色调和亮度。"""
    def __init__(
        self,
        contrast: Tuple[float, float] = (0.5, 1.5),
        saturation: Tuple[float, float] = (0.5, 1.5),
        hue: Tuple[float, float] = (-0.05, 0.05),
        brightness: Tuple[float, float] = (0.875, 1.125),
        p: float = 0.5,
    ):
        super().__init__()
        self._brightness = T.ColorJitter(brightness=brightness)  # 亮度调整
        self._contrast = T.ColorJitter(contrast=contrast)  # 对比度调整
        self._hue = T.ColorJitter(hue=hue)  # 色调调整
        self._saturation = T.ColorJitter(saturation=saturation)  # 饱和度调整
        self.p = p  # 应用此变换的概率

    def forward(
        self, image: Tensor, target: Optional[Dict[str, Tensor]] = None
    ) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        if isinstance(image, torch.Tensor):
            if image.ndimension() not in {2, 3}:
                raise ValueError(f"图像应为2/3维。当前维度为 {image.ndimension()}。")
            elif image.ndimension() == 2:
                image = image.unsqueeze(0)  # 升维处理，适应后续操作

        r = torch.rand(7)  # 随机决定哪些属性将被调整

        if r[0] < self.p:
            image = self._brightness(image)  # 调整亮度

        contrast_before = r[1] < 0.5
        if contrast_before and r[2] < self.p:
            image = self._contrast(image)  # 调整对比度

        if r[3] < self.p:
            image = self._saturation(image)  # 调整饱和度

        if r[4] < self.p:
            image = self._hue(image)  # 调整色调

        if not contrast_before and r[5] < self.p:
            image = self._contrast(image)  # 再次调整对比度

        if r[6] < self.p:
            channels, _, _ = F.get_dimensions(image)  # 获取通道数
            permutation = torch.randperm(channels)  # 随机重排通道

            is_pil = F._is_pil_image(image)
            if is_pil:
                image = F.pil_to_tensor(image)
                image = F.convert_image_dtype(image)
            image = image[..., permutation, :, :]
            if is_pil:
                image = F.to_pil_image(image)  # 如果原始为PIL图像，则转回PIL格式

        return image, target


class ScaleJitter(nn.Module):
    """
    随机调整图像大小，用于数据增强，根据目标大小和比例范围随机缩放图像。
    """
    def __init__(
        self,
        target_size: Tuple[int, int],  # 目标大小（高度，宽度）
        scale_range: Tuple[float, float] = (0.1, 2.0),  # 缩放比例范围
        interpolation: InterpolationMode = InterpolationMode.BILINEAR,  # 插值方式，默认双线性插值
        antialias=True,  # 是否使用抗锯齿功能
    ):
        super().__init__()  # 调用父类的构造函数
        self.target_size = target_size  # 初始化目标尺寸
        self.scale_range = scale_range  # 初始化缩放范围
        self.interpolation = interpolation  # 初始化插值方式
        self.antialias = antialias  # 初始化抗锯齿功能

    def forward(
        self, image: Tensor, target: Optional[Dict[str, Tensor]] = None
    ) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        if isinstance(image, torch.Tensor):  # 检查输入是否为张量
            if image.ndimension() not in {2, 3}:  # 检查图像维度是否为2或3
                raise ValueError(f"图像应为2/3维。当前维度为 {image.ndimension()}。")
            elif image.ndimension() == 2:  # 如果图像维度为2
                image = image.unsqueeze(0)  # 将图像维度扩展为3

        _, orig_height, orig_width = F.get_dimensions(image)  # 获取原始图像尺寸

        # 计算新的缩放比例
        scale = self.scale_range[0] + torch.rand(1) * (self.scale_range[1] - self.scale_range[0])
        # 计算缩放比例并限制在指定范围内：目标尺寸最小值'小于'原始尺寸，就缩小。目标尺寸'都'大于原始尺寸，就放大。
        # 这行代码计算实际使用的缩放比例 r。首先，它计算了目标尺寸与原始图像尺寸的高度和宽度之比，并取两者中较小的那个值。
        # 然后，将这个比例乘以随机生成的缩放比例 scale，得到最终的缩放比例 r。
        r = min(self.target_size[1] / orig_height, self.target_size[0] / orig_width) * scale
        new_width = int(orig_width * r)  # 计算新的宽度
        new_height = int(orig_height * r)  # 计算新的高度

        # 重新调整图像大小
        image = F.resize(image, [new_height, new_width], interpolation=self.interpolation, antialias=self.antialias)

        if target is not None:
            # 调整目标框大小
            target["boxes"][:, 0::2] *= new_width / orig_width  # 调整目标框的宽度
            target["boxes"][:, 1::2] *= new_height / orig_height  # 调整目标框的高度
            if "masks" in target:  # 如果存在掩码信息
                # 调整掩码大小
                target["masks"] = F.resize(
                    target["masks"],
                    [new_height, new_width],
                    interpolation=InterpolationMode.NEAREST,  # 使用最近邻插值
                    antialias=self.antialias,
                )

        return image, target  # 返回调整后的图像和目标信息


class RandomZoomOut(nn.Module):
    """随机放大图像"""

    def __init__(
            self, fill: Optional[List[float]] = None, side_range: Tuple[float, float] = (1.0, 4.0), p: float = 0.5
    ):
        super().__init__()
        if fill is None:
            fill = [0.0, 0.0, 0.0]  # 如果未指定填充色，默认为黑色
        self.fill = fill  # fill: 缩放时边缘填充的颜色。
        self.side_range = side_range  # side_range: 缩放的比例范围。
        if side_range[0] < 1.0 or side_range[0] > side_range[1]:
            raise ValueError(f"提供的画布大小范围无效 {side_range}.")
        self.p = p  # 应用此变换的概率

    @torch.jit.unused
    def _get_fill_value(self, is_pil):
        return tuple(int(x) for x in self.fill) if is_pil else 0

    def forward(
            self, image: Tensor, target: Optional[Dict[str, Tensor]] = None
    ) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        if isinstance(image, torch.Tensor):
            if image.ndimension() not in {2, 3}:
                raise ValueError(f"image should be 2/3 dimensional. Got {image.ndimension()} dimensions.")
            elif image.ndimension() == 2:
                image = image.unsqueeze(0)  # 升维处理，适应后续操作

        if torch.rand(1) >= self.p:
            return image, target  # 按概率不执行任何操作

        _, orig_h, orig_w = F.get_dimensions(image)  # 获取原始图像尺寸

        r = self.side_range[0] + torch.rand(1) * (self.side_range[1] - self.side_range[0])
        canvas_width = int(orig_w * r)
        canvas_height = int(orig_h * r)

        r = torch.rand(2)  # 随机确定新图像在画布中的位置
        left = int((canvas_width - orig_w) * r[0])
        top = int((canvas_height - orig_h) * r[1])
        right = canvas_width - (left + orig_w)
        bottom = canvas_height - (top + orig_h)

        if torch.jit.is_scripting():
            fill = 0
        else:
            fill = self._get_fill_value(F._is_pil_image(image))  # 根据是否为PIL图像获取填充值

        image = F.pad(image, [left, top, right, bottom], fill=fill)  # 扩展图像尺寸，填充边缘
        if isinstance(image, torch.Tensor):
            # PyTorch's pad supports only integers on fill. So we need to overwrite the colour
            v = torch.tensor(self.fill, device=image.device, dtype=image.dtype).view(-1, 1, 1)
            image[..., :top, :] = image[..., :, :left] = image[..., (top + orig_h):, :] = image[
                                                                                          ..., :, (left + orig_w):
                                                                                          ] = v

        if target is not None:
            target["boxes"][:, 0::2] += left  # 调整目标框的位置
            target["boxes"][:, 1::2] += top
            # 关键改动
            # 对masks的填充处理：
            if "masks" in target:
                masks = target["masks"]
                num_masks, mask_h, mask_w = masks.shape
                new_masks = torch.zeros((num_masks, canvas_height, canvas_width), dtype=masks.dtype,
                                        device=masks.device)
                new_masks[:, top:top + mask_h, left:left + mask_w] = masks
                target["masks"] = new_masks

        return image, target



class RandomRotatesmallangle:
    """随机旋转图像一定角度（-90到90度）并调整边界框。"""
    def __init__(self, angle_range=(-90, 90), prob=0.5):
        self.angle_range = angle_range
        self.prob = prob

    def __call__(self, image: torch.Tensor, target: Optional[Dict[str, torch.Tensor]] = None) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        if random.random() < self.prob:
            angle = random.uniform(self.angle_range[0], self.angle_range[1])
            height, width = image.shape[-2:]

            # 旋转图像
            image = F.rotate(image, angle, interpolation=InterpolationMode.BILINEAR)

            if 'boxes' in target:
                # 计算旋转中心
                cx, cy = width / 2, height / 2
                theta = math.radians(angle)
                rotation_matrix = torch.tensor([
                    [math.cos(theta), -math.sin(theta)],
                    [math.sin(theta), math.cos(theta)]
                ], dtype=torch.float32)
                bbox = target['boxes']
                corners = torch.stack([bbox[:, 0], bbox[:, 1], bbox[:, 2], bbox[:, 1], bbox[:, 2], bbox[:, 3], bbox[:, 0], bbox[:, 3]], dim=1).reshape(-1, 2)
                corners = corners - torch.tensor([cx, cy], dtype=torch.float32)
                rotated_corners = torch.matmul(corners, rotation_matrix)
                rotated_corners = rotated_corners + torch.tensor([cx, cy], dtype=torch.float32)
                rotated_corners = rotated_corners.reshape(-1, 4, 2)
                xmin, _ = rotated_corners[:, :, 0].min(dim=1)
                ymin, _ = rotated_corners[:, :, 1].min(dim=1)
                xmax, _ = rotated_corners[:, :, 0].max(dim=1)
                ymax, _ = rotated_corners[:, :, 1].max(dim=1)
                target['boxes'] = torch.stack([xmin, ymin, xmax, ymax], dim=1)

                # 检查边界框的有效性
                is_valid = (target['boxes'][:, 0] < target['boxes'][:, 2]) & (target['boxes'][:, 1] < target['boxes'][:, 3])
                if is_valid.any():
                    target['boxes'] = target['boxes'][is_valid]
                    target['labels'] = target['labels'][is_valid]
                    if 'masks' in target:
                        target['masks'] = target['masks'][is_valid]
                else:
                    return image, None

            if 'masks' in target:
                masks = target['masks']
                num_masks = masks.shape[0]
                rotated_masks = []
                for i in range(num_masks):
                    mask = masks[i]
                    if mask.dim() == 2:
                        mask = mask.unsqueeze(0)
                    rotated_mask = F.rotate(mask, angle, interpolation=InterpolationMode.NEAREST)
                    rotated_masks.append(rotated_mask.squeeze(0))
                target['masks'] = torch.stack(rotated_masks)


        return image, target


class RandomRotate45:
    """固定角度变换-向右旋转45°"""
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image: torch.Tensor, target: Optional[Dict[str, torch.Tensor]] = None) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        if random.random() < self.prob:
            angle = 45
            height, width = image.shape[-2:]

            # 旋转图像
            image = F.rotate(image, angle, interpolation=InterpolationMode.BILINEAR)

            if 'boxes' in target:
                # 计算旋转中心
                cx, cy = width / 2, height / 2
                theta = math.radians(angle)
                rotation_matrix = torch.tensor([
                    [math.cos(theta), -math.sin(theta)],
                    [math.sin(theta), math.cos(theta)]
                ], dtype=torch.float32)
                bbox = target['boxes']
                corners = torch.stack([bbox[:, 0], bbox[:, 1], bbox[:, 2], bbox[:, 1], bbox[:, 2], bbox[:, 3], bbox[:, 0], bbox[:, 3]], dim=1).reshape(-1, 2)
                corners = corners - torch.tensor([cx, cy], dtype=torch.float32)
                rotated_corners = torch.matmul(corners, rotation_matrix)
                rotated_corners = rotated_corners + torch.tensor([cx, cy], dtype=torch.float32)
                rotated_corners = rotated_corners.reshape(-1, 4, 2)
                xmin, _ = rotated_corners[:, :, 0].min(dim=1)
                ymin, _ = rotated_corners[:, :, 1].min(dim=1)
                xmax, _ = rotated_corners[:, :, 0].max(dim=1)
                ymax, _ = rotated_corners[:, :, 1].max(dim=1)
                target['boxes'] = torch.stack([xmin, ymin, xmax, ymax], dim=1)

                # 检查边界框的有效性
                is_valid = (target['boxes'][:, 0] < target['boxes'][:, 2]) & (target['boxes'][:, 1] < target['boxes'][:, 3])
                if is_valid.any():
                    target['boxes'] = target['boxes'][is_valid]
                    target['labels'] = target['labels'][is_valid]
                    if 'masks' in target:
                        target['masks'] = target['masks'][is_valid]
                else:
                    return image, None

            if 'masks' in target:
                # 旋转掩码
                masks = target['masks']
                num_masks = masks.shape[0]
                rotated_masks = []
                for i in range(num_masks):
                    mask = masks[i]
                    if mask.dim() == 2:
                        mask = mask.unsqueeze(0)
                    rotated_mask = F.rotate(mask, angle, interpolation=InterpolationMode.NEAREST)
                    rotated_masks.append(rotated_mask.squeeze(0))
                target['masks'] = torch.stack(rotated_masks)

        return image, target
class RandomRotatealittle:
    """随机旋转图像一定角度并调整边界框"""
    def __init__(self, angle_range=(-45, 45), prob=0.5):
        self.angle_range = angle_range
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            angle = random.uniform(self.angle_range[0], self.angle_range[1])
            height, width = image.shape[-2:]

            # 旋转图像
            image = F.rotate(image, angle, interpolation=InterpolationMode.BILINEAR)
            """
            在旋转操作中，对边界框进行旋转之后，检查哪些边界框是有效的，并过滤掉无效的边界框。
            """
            if 'boxes' in target:
                cx, cy = width / 2, height / 2
                theta = math.radians(angle)
                rotation_matrix = torch.tensor([
                    [math.cos(theta), -math.sin(theta)],
                    [math.sin(theta), math.cos(theta)]
                ], dtype=torch.float32)
                bbox = target['boxes']
                corners = torch.stack([bbox[:, 0], bbox[:, 1], bbox[:, 2], bbox[:, 1], bbox[:, 2], bbox[:, 3], bbox[:, 0], bbox[:, 3]], dim=1).reshape(-1, 2)
                corners = corners - torch.tensor([cx, cy], dtype=torch.float32)
                rotated_corners = torch.matmul(corners, rotation_matrix)
                rotated_corners = rotated_corners + torch.tensor([cx, cy], dtype=torch.float32)
                rotated_corners = rotated_corners.reshape(-1, 4, 2)
                xmin, _ = rotated_corners[:, :, 0].min(dim=1)
                ymin, _ = rotated_corners[:, :, 1].min(dim=1)
                xmax, _ = rotated_corners[:, :, 0].max(dim=1)
                ymax, _ = rotated_corners[:, :, 1].max(dim=1)
                target['boxes'] = torch.stack([xmin, ymin, xmax, ymax], dim=1)

                # 检查边界框的有效性 is_valid 是一个布尔张量，表示哪些边界框是有效的。is_valid 的每个元素是一个布尔值，表示对应的边界框是否有效。有效的边界框应满足 xmin < xmax 和 ymin < ymax。
                is_valid = (target['boxes'][:, 0] < target['boxes'][:, 2]) & (target['boxes'][:, 1] < target['boxes'][:, 3])
                if is_valid.any(): #is_valid.any() 检查是否有任何有效的边界框。
                    target['boxes'] = target['boxes'][is_valid] #保留有效的边界框。
                    target['labels'] = target['labels'][is_valid] #保留有效边界框对应的标签。
                    if 'masks' in target:
                        target['masks'] = target['masks'][is_valid] #保留有效边界框对应的掩码。
                else:
                    return image, None  #返回原始图像和 None 作为目标。这通常表示在这个图像中没有有效的目标。

            if 'masks' in target:
                # 旋转掩码
                masks = target['masks']  # 获取目标中的掩码张量
                num_masks = masks.shape[0]  # 获取掩码的数量
                rotated_masks = []  # 用于存储旋转后的掩码
                for i in range(num_masks):
                    mask = masks[i]  # 获取第 i 个掩码
                    if mask.dim() == 2:
                        mask = mask.unsqueeze(0)  # 如果掩码是二维的，将其扩展为三维 (1, H, W)
                    rotated_mask = F.rotate(mask, angle, interpolation=F.InterpolationMode.NEAREST)
                    rotated_masks.append(rotated_mask.squeeze(0))  # 将旋转后的掩码去掉第0维并存储在列表中
                target['masks'] = torch.stack(rotated_masks)  # 将旋转后的掩码堆叠成一个张量，并重新存储回目标中

        return image, target


class FixedSizeCrop(nn.Module):
    def __init__(self, size, fill=0, padding_mode="constant"):
        super().__init__()
        size = tuple(T._setup_size(size, error_msg="Please provide only two dimensions (h, w) for size."))
        self.crop_height = size[0]
        self.crop_width = size[1]
        self.fill = fill  # TODO: Fill is currently respected only on PIL. Apply tensor patch.
        self.padding_mode = padding_mode

    def _pad(self, img, target, padding):
        # Taken from the functional_tensor.py pad
        if isinstance(padding, int):
            pad_left = pad_right = pad_top = pad_bottom = padding
        elif len(padding) == 1:
            pad_left = pad_right = pad_top = pad_bottom = padding[0]
        elif len(padding) == 2:
            pad_left = pad_right = padding[0]
            pad_top = pad_bottom = padding[1]
        else:
            pad_left = padding[0]
            pad_top = padding[1]
            pad_right = padding[2]
            pad_bottom = padding[3]

        padding = [pad_left, pad_top, pad_right, pad_bottom]
        img = F.pad(img, padding, self.fill, self.padding_mode)
        if target is not None:
            target["boxes"][:, 0::2] += pad_left
            target["boxes"][:, 1::2] += pad_top
            if "masks" in target:
                target["masks"] = F.pad(target["masks"], padding, 0, "constant")

        return img, target

    def _crop(self, img, target, top, left, height, width):
        img = F.crop(img, top, left, height, width)
        if target is not None:
            boxes = target["boxes"]
            boxes[:, 0::2] -= left
            boxes[:, 1::2] -= top
            boxes[:, 0::2].clamp_(min=0, max=width)
            boxes[:, 1::2].clamp_(min=0, max=height)

            is_valid = (boxes[:, 0] < boxes[:, 2]) & (boxes[:, 1] < boxes[:, 3])

            target["boxes"] = boxes[is_valid]
            target["labels"] = target["labels"][is_valid]
            if "masks" in target:
                valid_masks = target["masks"][is_valid]
                # print(f"Debug: Number of valid masks: {valid_masks.size(0)}")
                # print(f"Debug: Shape of valid masks before cropping: {valid_masks.shape}")
                if valid_masks.numel() > 0:
                    try:
                        target["masks"] = torch.stack(
                            [F.crop(mask, top, left, height, width) for mask in valid_masks]
                        )
                    except Exception as e:
                        print(f"Error in cropping masks: {e}")
                        print(f"Top: {top}, Left: {left}, Height: {height}, Width: {width}")
                        print(f"Valid indices: {is_valid.nonzero(as_tuple=True)}")
                        raise
                else:
                    target["masks"] = valid_masks
                # print(f"Debug: Shape of valid masks after cropping: {target['masks'].shape}")

            return img, target

    def forward(self, img, target=None):
        _, height, width = F.get_dimensions(img)
        new_height = min(height, self.crop_height)
        new_width = min(width, self.crop_width)

        if new_height != height or new_width != width:
            offset_height = max(height - self.crop_height, 0)
            offset_width = max(width - self.crop_width, 0)

            r = torch.rand(1)
            top = int(offset_height * r)
            left = int(offset_width * r)

            img, target = self._crop(img, target, top, left, new_height, new_width)

        pad_bottom = max(self.crop_height - new_height, 0)
        pad_right = max(self.crop_width - new_width, 0)
        if pad_bottom != 0 or pad_right != 0:
            img, target = self._pad(img, target, [0, 0, pad_right, pad_bottom])

        return img, target


class RandomShortestSize(nn.Module):
    """
    随机调整图像到最短边的特定大小，这是常用的数据增强方法之一。
    抗锯齿（Antialiasing）：
    当图像缩小到较小的尺寸时，如果没有抗锯齿处理，边缘可能会出现明显的锯齿状效应。
    启用抗锯齿后，边缘会被平滑处理，以减少锯齿效应。
    使用场景：
    在图像分类、目标检测和其他计算机视觉任务中，预处理步骤可能会涉及到对图像的缩放、裁剪等操作。启用抗锯齿可以提高处理后的图像质量，从而可能提高模型的性能。
    影响：
    启用抗锯齿可能会增加一些计算开销，但通常在提高图像质量方面是值得的，特别是对于高精度要求的任务。
    """
    def __init__(
        self,
        min_size: Union[List[int], Tuple[int], int],  # 最短边的大小，可以是一个值或一个列表
        max_size: int,  # 最大尺寸限制
        interpolation: InterpolationMode = InterpolationMode.BILINEAR,  # 插值方式
        antialias: Optional[bool] = True  # 添加 antialias 参数
    ):
        super().__init__()
        self.min_size = [min_size] if isinstance(min_size, int) else list(min_size)  # 确保min_size是列表形式
        self.max_size = max_size  # 最大尺寸
        self.interpolation = interpolation  # 插值方法
        self.antialias = antialias  # antialias 参数

    def forward(
        self, image: Tensor, target: Optional[Dict[str, Tensor]] = None
    ) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        _, orig_height, orig_width = F.get_dimensions(image)  # 获取原始图像的尺寸

        min_size = self.min_size[torch.randint(len(self.min_size), (1,)).item()]  # 随机选择一个最短边大小
        r = min(min_size / min(orig_height, orig_width), self.max_size / max(orig_height, orig_width))  # 计算缩放比例

        new_width = int(orig_width * r)  # 计算新宽度
        new_height = int(orig_height * r)  # 计算新高度

        image = F.resize(image, [new_height, new_width], interpolation=self.interpolation)  # 调整图像大小

        if target is not None:
            target["boxes"][:, 0::2] *= new_width / orig_width  # 调整bounding box宽度
            target["boxes"][:, 1::2] *= new_height / orig_height  # 调整bounding box高度
            if "masks" in target:
                target["masks"] = F.resize(
                    target["masks"], [new_height, new_width], interpolation=InterpolationMode.NEAREST  # 调整掩模大小
                )

        return image, target

# def _copy_paste(
#         image: torch.Tensor,
#         target: Dict[str, torch.Tensor],
#         paste_image: torch.Tensor,
#         paste_target: Dict[str, torch.Tensor],
#         blending: bool = True,
#         resize_interpolation: F.InterpolationMode = F.InterpolationMode.BILINEAR
# ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
#     """
#     实现具体的复制粘贴逻辑。
#     """
#     num_masks = len(paste_target["masks"])
#     if num_masks < 1:
#         return image, target  # 如果没有掩码直接返回原图和目标
#
#     random_selection = torch.randperm(num_masks, device=paste_image.device)[:max(1, num_masks // 2)]
#
#     paste_masks = paste_target["masks"][random_selection]
#     paste_boxes = paste_target["boxes"][random_selection]
#     paste_labels = paste_target["labels"][random_selection]
#
#     size1 = image.shape[-2:]
#     size2 = paste_image.shape[-2:]
#     if size1 != size2:
#         paste_image = F.resize(paste_image, size1, interpolation=resize_interpolation)
#         paste_masks = F.resize(paste_masks, size1, interpolation=F.InterpolationMode.NEAREST)
#         ratios = torch.tensor([size1[1] / size2[1], size1[0] / size2[0]], device=paste_boxes.device)
#         paste_boxes = paste_boxes * ratios.repeat(2).view(1, 2)
#
#     paste_alpha_mask = paste_masks.sum(dim=0).bool()
#     if blending:
#         paste_alpha_mask = F.gaussian_blur(paste_alpha_mask.float().unsqueeze(0), kernel_size=(5, 5), sigma=[2.0]).bool().squeeze(0)
#
#     image = torch.where(paste_alpha_mask.unsqueeze(0), paste_image, image)
#
#     masks = target["masks"] * (~paste_alpha_mask)
#     non_all_zero_masks = masks.sum((-1, -2)) > 0
#     masks = masks[non_all_zero_masks]
#
#     out_target = {k: v for k, v in target.items()}
#     out_target["masks"] = torch.cat([masks, paste_masks], dim=0)
#     out_target["boxes"] = torch.cat([ops.masks_to_boxes(masks), paste_boxes], dim=0)
#
#     # 确保 labels 张量维度正确
#     target_labels = target["labels"][non_all_zero_masks]
#     out_target["labels"] = torch.cat([target_labels, paste_labels], dim=0)
#
#     return image, out_target
#
# class SimpleCopyPaste(nn.Module):
#     def __init__(self, blending=True, resize_interpolation=F.InterpolationMode.BILINEAR):
#         super().__init__()
#         self.resize_interpolation = resize_interpolation
#         self.blending = blending
#
#     def forward(self, images: List[torch.Tensor], targets: List[Dict[str, Tensor]]):
#         images_rolled = images[-1:] + images[:-1]
#         targets_rolled = targets[-1:] + targets[:-1]
#
#         output_images = []
#         output_targets = []
#         for image, target, paste_image, paste_target in zip(images, targets, images_rolled, targets_rolled):
#             output_image, output_target = _copy_paste(
#                 image, target, paste_image, paste_target,
#                 blending=self.blending, resize_interpolation=self.resize_interpolation
#             )
#             output_images.append(output_image)
#             output_targets.append(output_target)
#
#         return output_images, output_targets
#
#     def __repr__(self):
#         return f"{self.__class__.__name__}(blending={self.blending}, resize_interpolation={self.resize_interpolation})"

