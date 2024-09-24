import math
import sys
import time

import torch

import maskrcnn_ribfrac.utils.distributed_utils as utils
from .coco_eval import EvalCOCOMetric


def train_one_epoch(model, optimizer, data_loader, device, epoch,
                    print_freq=50, warmup=False, scaler=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    lr_scheduler = None
    if epoch == 0 and warmup is True:  # 当训练第一轮（epoch=0）时，启用warmup训练方式，可理解为热身训练
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    # 初始化用于记录平均损失的变量
    mloss = torch.zeros(1).to(device)  # mean losses
    # 初始化存储各个损失组件的字典
    loss_components = {'loss_classifier': 0, 'loss_box_reg': 0, 'loss_mask': 0, 'loss_objectness': 0, 'loss_rpn_box_reg': 0}

    for i, [images, targets] in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # 混合精度训练上下文管理器，如果在CPU环境中不起任何作用
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purpose
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        loss_value = losses_reduced.item()
        # 记录训练损失
        mloss = (mloss * i + loss_value) / (i + 1)  # update mean losses

        if not math.isfinite(loss_value):  # 当计算的损失为无穷大时停止训练
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            optimizer.step()

        if lr_scheduler is not None:  # 第一轮使用warmup训练方式
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        for name, value in loss_dict_reduced.items():
            # 更新各个损失组件，累积每个损失
            loss_components[name] += value.item() * images[0].size(0)  # Update loss components with normalized loss
        # 获取当前学习率
        now_lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=now_lr)

    # 规范化损失组件的计算，确保准确性
    num_samples = len(data_loader.dataset)
    for k in loss_components.keys():
        loss_components[k] /= num_samples
    # 返回平均损失、当前学习率和损失组件字典
    return mloss, now_lr, loss_components


# 在 train_eval_utils.py 文件中更新 evaluate 函数
@torch.no_grad()
def evaluate(model, data_loader, device):
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test: "

    det_metric = EvalCOCOMetric(data_loader.dataset.coco, iou_type="bbox", results_file_name="det_results.json")
    seg_metric = EvalCOCOMetric(data_loader.dataset.coco, iou_type="segm", results_file_name="seg_results.json")
    for image, targets in metric_logger.log_every(data_loader, 100, header):
        image = list(img.to(device) for img in image)

        # 当使用CPU时，跳过GPU相关指令
        if device != torch.device("cpu"):
            torch.cuda.synchronize(device)

        model_time = time.time()
        outputs = model(image)

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        det_metric.update(targets, outputs)
        seg_metric.update(targets, outputs)
        metric_logger.update(model_time=model_time)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    # 同步所有进程中的数据
    det_metric.synchronize_results()
    seg_metric.synchronize_results()

    if utils.is_main_process():
        coco_info = det_metric.evaluate()
        seg_info = seg_metric.evaluate()
    else:
        coco_info = None
        seg_info = None

    return coco_info, seg_info

