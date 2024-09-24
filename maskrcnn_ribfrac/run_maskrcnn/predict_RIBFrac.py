import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import os
import time
import json
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import argparse
from maskrcnn_ribfrac.utils.draw_box_utils import draw_objs
from tqdm import tqdm
from maskrcnn_ribfrac.utils import load_image,merge_two_boxes,box_overlap,merge_boxes_and_masks,save_jpg,save_dicom,time_synchronized,create_model,dicom_to_image,ensure_dir_exists

def load_weights(model, model_id):
    if model_id == "maskrcn50":
        model_path = r"../RibFrac50.pth"
    elif model_id == "maskrcn101":
        model_path = r"../RibFrac101.pth"
    elif model_id == "maskrcn152":
        model_path = r"../RibFrac152.pth"
    else:
        raise ValueError(f"Unsupported model_id: {model_id}")

    assert os.path.exists(model_path), "{} file does not exist.".format(model_path)
    weights_dict = torch.load(model_path, map_location='cpu')
    weights_dict = weights_dict["model_state_dict"] if "model_state_dict" in weights_dict else weights_dict
    model.load_state_dict(weights_dict)
    return model


def main():
    parser = argparse.ArgumentParser(description="Predict DICOM or NIfTI images with Mask R-CNN")
    parser.add_argument('--num_classes', type=int, default=1, help='Number of classes')
    parser.add_argument('--box_thresh', type=float, default=0.5, help='Box score threshold')
    parser.add_argument('--model_id', type=str, default="maskrcn152", help='Model identifier')
    parser.add_argument('--img_folder', type=str, default=r"", help='Path to input images')
    parser.add_argument('--output_folder', type=str,default=r"",  help='Path to save output')
    parser.add_argument('--label_json_path', type=str, default='cocorib_indices.json', help='Path to label JSON')
    parser.add_argument('--save_format', type=str, choices=['dicom', 'jpg'], default='jpg', help='Output format')

    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    model = create_model(model_id=args.model_id, num_classes=args.num_classes + 1, box_thresh=args.box_thresh)
    model = load_weights(model, args.model_id)
    model.to(device)

    assert os.path.exists(args.label_json_path), "json file {} does not exist.".format(args.label_json_path)
    with open(args.label_json_path, 'r') as json_file:
        category_index = json.load(json_file)

    patients = [root for root, dirs, files in os.walk(args.img_folder) if files]

    for root in patients:
        tqdm_bar = tqdm([file for file in os.listdir(root) if file.endswith((".IMA", ".jpg", ".jpeg", ".png"))],
                        desc=root, leave=True)
        for file in tqdm_bar:
            img_file = os.path.join(root, file)
            relative_path = os.path.relpath(root, args.img_folder)
            img_output_folder = os.path.join(args.output_folder, relative_path)
            ensure_dir_exists(img_output_folder)

            output_extension = 'dcm' if args.save_format == 'dicom' else 'jpg'
            img_output_path = os.path.join(img_output_folder, f"{os.path.splitext(file)[0]}.{output_extension}")

            # 使用 load_image 来加载图像
            original_img, dicom, img_min, img_max = load_image(img_file)

            data_transform = transforms.Compose([transforms.ToTensor()])
            img = data_transform(original_img)
            img = torch.unsqueeze(img, dim=0)

            model.eval()
            with torch.no_grad():
                img_height, img_width = img.shape[-2:]
                init_img = torch.zeros((1, 3, img_height, img_width), device=device)
                model(init_img)

                t_start = time_synchronized()
                predictions = model(img.to(device))[0]
                t_end = time_synchronized()

                predict_boxes = predictions["boxes"].to("cpu").numpy()
                predict_classes = predictions["labels"].to("cpu").numpy()
                predict_scores = predictions["scores"].to("cpu").numpy()
                predict_mask = predictions["masks"].to("cpu").numpy()
                predict_mask = np.squeeze(predict_mask, axis=1)

                if len(predict_boxes) == 0:
                    plot_img = original_img
                else:
                    merged_boxes, merged_masks, merged_scores = merge_boxes_and_masks(predict_boxes, predict_mask,
                                                                                      predict_scores)

                    plot_img = draw_objs(original_img,
                                         boxes=merged_boxes,
                                         classes=predict_classes[:len(merged_boxes)],  # 合并后的 classes
                                         scores=merged_scores,
                                         masks=merged_masks,  # 绘制合并后的 mask
                                         category_index=category_index,
                                         line_thickness=1,
                                         font_size=10)

                if args.save_format == 'dicom':
                    save_dicom(plot_img, dicom, img_output_path)
                else:
                    save_jpg(plot_img, img_output_path)

if __name__ == '__main__':
    main()
