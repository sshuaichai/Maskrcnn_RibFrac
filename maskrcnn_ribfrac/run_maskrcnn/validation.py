import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import json
import torch
from tqdm import tqdm
import numpy as np
from maskrcnn_ribfrac.network_files import MaskRCNN
from maskrcnn_ribfrac.utils import EvalCOCOMetric, transforms

def summarize(self, catId=None):
    def _summarize(ap=1, iouThr=None, areaRng='all', maxDets=100):
        p = self.params
        iStr = ' {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}'
        titleStr = 'Average Precision' if ap == 1 else 'Average Recall'
        typeStr = '(AP)' if ap == 1 else '(AR)'
        iouStr = '{:0.2f}:{:0.2f}'.format(p.iouThrs[0], p.iouThrs[-1]) \
            if iouThr is None else '{:0.2f}'.format(iouThr)

        aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
        mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]

        if ap == 1:
            # dimension of precision: [TxRxKxAxM]
            s = self.eval['precision']
            # IoU
            if iouThr is not None:
                t = np.where(iouThr == p.iouThrs)[0]
                s = s[t]

            if isinstance(catId, int):
                s = s[:, :, catId, aind, mind]
            else:
                s = s[:, :, :, aind, mind]

        else:
            # dimension of recall: [TxKxAxM]
            s = self.eval['recall']
            if iouThr is not None:
                t = np.where(iouThr == p.iouThrs)[0]
                s = s[t]

            if isinstance(catId, int):
                s = s[:, catId, aind, mind]
            else:
                s = s[:, :, aind, mind]

        if len(s[s > -1]) == 0:
            mean_s = -1
        else:
            mean_s = np.mean(s[s > -1])

        print_string = iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets, mean_s)
        return mean_s, print_string

    stats, print_list = [0] * 12, [""] * 12
    stats[0], print_list[0] = _summarize(1)
    stats[1], print_list[1] = _summarize(1, iouThr=.1, maxDets=self.params.maxDets[2])
    stats[2], print_list[2] = _summarize(1, iouThr=.50, maxDets=self.params.maxDets[2])
    stats[3], print_list[3] = _summarize(1, areaRng='small', maxDets=self.params.maxDets[2])
    stats[4], print_list[4] = _summarize(1, areaRng='medium', maxDets=self.params.maxDets[2])
    stats[5], print_list[5] = _summarize(1, areaRng='large', maxDets=self.params.maxDets[2])
    stats[6], print_list[6] = _summarize(0, maxDets=self.params.maxDets[0])
    stats[7], print_list[7] = _summarize(0, maxDets=self.params.maxDets[1])
    stats[8], print_list[8] = _summarize(0, maxDets=self.params.maxDets[2])
    stats[9], print_list[9] = _summarize(0, areaRng='small', maxDets=self.params.maxDets[2])
    stats[10], print_list[10] = _summarize(0, areaRng='medium', maxDets=self.params.maxDets[2])
    stats[11], print_list[11] = _summarize(0, areaRng='large', maxDets=self.params.maxDets[2])

    print_info = "\n".join(print_list)

    if not self.eval:
        raise Exception('Please run_maskrcnn accumulate() first')

    return stats, print_info


def save_info(coco_evaluator,
              category_index: dict,
              save_name: str = "record_mAP.txt"):
    iou_type = coco_evaluator.params.iouType
    print(f"IoU metric: {iou_type}")
    # calculate COCO info for all classes
    coco_stats, print_coco = summarize(coco_evaluator)

    # calculate voc info for every classes(IoU=0.5)
    classes = [v for v in category_index.values() if v != "N/A"]
    voc_map_info_list = []
    for i in range(len(classes)):
        stats, _ = summarize(coco_evaluator, catId=i)
        voc_map_info_list.append(" {:15}: {}".format(classes[i], stats[2]))  # stats[1] : iouThr=.1

    print_voc = "\n".join(voc_map_info_list)
    print(print_voc)

    with open(save_name, "w") as f:
        record_lines = ["COCO results:",
                        print_coco,
                        "",
                        "mAP(IoU=0.5) for each category:",   # iouThr=.1
                        print_voc]
        f.write("\n".join(record_lines))


def main(parser_data):
    device = torch.device(parser_data.device if torch.cuda.is_available() else "cpu")
    print("Using {} device training.".format(device.type))

    data_transform = {
        "val": transforms.Compose([
            transforms.ToTensor(),
            transforms.ToDtype(torch.float32, scale=True)
       ])
    }

    # read class_indict
    label_json_path = parser_data.label_json_path
    assert os.path.exists(label_json_path), "json file {} dose not exist.".format(label_json_path)
    with open(label_json_path, 'r') as f:
        category_index = json.load(f)

    data_root = parser_data.data_path

    batch_size = parser_data.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using %g dataloader workers' % nw)

    from maskrcnn_ribfrac.run_maskrcnn.my_dataset_cocoRib import CocoDetection
    val_dataset = CocoDetection(data_root, "val", data_transform["val"])

    val_dataset_loader = torch.utils.data.DataLoader(val_dataset,
                                                     batch_size=batch_size,
                                                     shuffle=False,
                                                     pin_memory=True,
                                                     num_workers=nw,
                                                     collate_fn=val_dataset.collate_fn)

    from maskrcnn_ribfrac.backbone import resnet50_fpn_backbone
    backbone = resnet50_fpn_backbone()
    # from maskrcnn_ribfrac.backbone import resnet101_fpn_backbone
    # backbone = resnet101_fpn_backbone()
    # from maskrcnn_ribfrac.backbone import resnet152_fpn_backbone
    # backbone = resnet152_fpn_backbone()

    model = MaskRCNN(backbone, num_classes=args.num_classes + 1)

    weights_path = parser_data.weights_path
    assert os.path.exists(weights_path), "not found {} file.".format(weights_path)
    model.load_state_dict(torch.load(weights_path, map_location='cpu')['model_state_dict'])
    # print(model)

    model.to(device)

    # evaluate on the val dataset
    cpu_device = torch.device("cpu")

    det_metric = EvalCOCOMetric(val_dataset.coco, "bbox", "det_results.json")
    seg_metric = EvalCOCOMetric(val_dataset.coco, "segm", "seg_results.json")
    model.eval()
    with torch.no_grad():
        for image, targets in tqdm(val_dataset_loader, desc="validation..."):
            image = list(img.to(device) for img in image)

            # inference
            outputs = model(image)

            outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
            det_metric.update(targets, outputs)
            seg_metric.update(targets, outputs)

    det_metric.synchronize_results()
    seg_metric.synchronize_results()
    det_metric.evaluate()
    seg_metric.evaluate()

    save_info(det_metric.coco_evaluator, category_index, "det_record_mAP.txt")
    save_info(seg_metric.coco_evaluator, category_index, "seg_record_mAP.txt")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=__doc__)

    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('--num-classes', type=int, default=1, help='number of classes, not including the background')
    parser.add_argument('--data-path', default=r"", help='dataset root')
    parser.add_argument('--weights-path', default=r"", type=str, help='training weights')
    parser.add_argument('--batch-size', default=1, type=int, metavar='N',
                        help='batch size when validation.')
    parser.add_argument('--label-json-path', type=str, default="cocorib_indices.json")

    args = parser.parse_args()

    main(args)

