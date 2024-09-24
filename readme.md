# 🌟 MaskRCNN Rib Fracture Detection

<img src="maskrcnn_ribfrac/assets/title.png" alt="" width="1200" height="" />

> 🦴 Rib location combined with rib fracture detection

<img src="maskrcnn_ribfrac/assets/2024-09-24-21-45-13.gif" alt="" width="500" height="auto" />

We have also conducted experiments on rib positioning, but it is not yet perfect. We have achieved good results through the fusion prediction of the positioning model and the fracture model. However, the rib area on both sides of the spine has not yet been well identified and detected. We will further improve this. 🚀

## 📁 Project Structure
Maskrcnn_RibFrac\
    ├── backbone : Backbone architecture  
    ├── data : Data directory  
    ├── network_files : MaskRCNN network architecture  
    ├── run_maskrcnn : Training, prediction, and evaluation scripts  
    └── utils : Utility functions  

First, create and activate a virtual environment:  
It is recommended to use a conda environment with `python3.9`. 🐍

# 🔧 Usage
## 📦 Install Required Packages
First, install required packages listed in `requirements.txt` using pip:

```bash
git clone https://github.com/sshuaichai/Maskrcnn_RibFrac.git
cd maskrcnn_ribfrac
pip install -r requirements.txt
```

# 📥 Download Dataset
We use a large-scale rib fracture CT dataset, named the RibFrac dataset, as a benchmark for developing rib fracture detection, segmentation, and classification algorithms. 
After free registration, you can access the public part of the RibFrac dataset through the RibFrac Challenge website https://ribfrac.grand-challenge.org/dataset/, which is the official challenge of MICCAI 2020. 
The public dataset in this document is in 2D format, processed from the official 3D format. 
Refer to the RibFrac Challenge website https://ribfrac.grand-challenge.org/tasks/ for more details. 

## 📂 Prepare the dataset

Please first download the RibFrac public dataset to the [data] folder, and place the images and labels in a ratio of 0.9:0.1 (that is, the first 450 are for training and the last 50 are for verification). 
The format is as follows:

```md
Dataset510_RibFrac/
├──imagesTr/    # Training set: 450 nii.gz images
│   ├── RibFrac1-image.nii     
│   ├── RibFrac2-image.nii
│   ├── ...
├── imagesTs/   # Validation set: 50 nii.gz images
│   └── Main/
│       ├── RibFrac451-image.nii 
│       ├── RibFrac452-image.nii
│       └── ...
├── labelsTr/    # Training set tag
│   ├── RibFrac1-label.nii     
│   ├── RibFrac2-label.nii
│   └── ...
└── labelsTs/    # Validation set tag
│   ├── RibFrac451-label.nii
│   ├── RibFrac452-label.nii
│   └──...
├── ribfrac-train-info.csv  # Label category and fracture correspondence for the first 450 images
└── ribfrac-val-info.csv    # Label category and fracture corresponding information for the last 50 images
```

After placing the data, running the following command will generate 2d slices that match the anatomy:

a. After generating the angular rotated nii image (this process is because the nii needs to perform appropriate axis conversion to correspond to the anatomical relationship when processing into 2d slices, it is necessary to run this step if you want to view 2d slices).
run the following code:

```bash
python data/RibFrac_preprocess.py --data-path data/Dataset510_RibFrac --output-dir data/Dataset510_RibFrac_preprocess
```

This will produce the nii image after the coordinate rotation, which we place in a separate file to prevent confusion with the original image. If you do not want to see a 2d slice of the correct anatomy, you can simply run the next command, which will not affect the performance of our model, because we have already added rotation mapping to the model.

Note: Don't forget to place the `ribfrac-train-info.csv` and `ribfrac-val-info.csv` in the [Dataset510_RibFrac_preprocess] folder.

b. Generate 2D slices

```bash
python data/Ribfrac_dataset.py --data-path data/Dataset510_RibFrac_preprocess --output-dir data/COCO_Rib2020_Ribfrac_v2
```
Some Warning⚠️: will be printed during the run. Ignore it. This is because we are checking the data and deleting the mismatched slices and label information.

This will generate our rib fracture 2d slices under our [data] folder, named COCO_Rib2020_Ribfrac_v2, with the following structure:

```md
COCO_Rib2020_Ribfrac_v2/
├── train/        
│   ├── patient1_slice_0.jpg
│   ├── patient1_slice_1.jpg
│   └── ...
├── val/
│   ├── patient1_slice_0.jpg
│   ├── patient1_slice_1.jpg
│   └── ...
├── annotations/
    ├── instances_train.json
    └── instances_val.json
```

In this example, `train` is a slice of the training set, `val` is a slice of the validation set, and the `annotations` list contains annotation information.

If you would like to use our data sets directly in 2D format, please download them to the [data] folder:

```md
Link: https://pan.baidu.com/s/1Z409IsidAG3-4p0sqdhbtA 
Extraction Code: RibF
```

## 🏋️‍♂️ Download Pytorch Pre-trained Weights

If you plan to use transfer learning to train the model, please download the Pytorch pre-trained weight file to the maskrcnn_ribfrac directory

- `resnet50.pth` imagenet weights url:"https://download.pytorch.org/models/resnet50-0676ba61.pth"
- `resnet101.pth` imagenet weights url: "https://download.pytorch.org/models/resnet101-63fe2227.pth"
- `resnet152.pth` imagenet weights url: "https://download.pytorch.org/models/resnet152-394f9c45.pth"
- `maskrcnn_resnet50_fpn_coco.pth` weights url: "https://download.pytorch.org/models/maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth"

## 🚀 Train the Model

To train the MaskRCNN_RibFrac model, run the following command in the terminal:

```bash
python run_maskrcnn/train_cocorib.py --data-path <image_path> --output-dir "save_weights_RibFrac" --num-classes 1 --batch_size 16 --epochs 300 --lr 0.01 --momentum 0.9 --weight-decay 1e-4  --validation-frequency 1 --patience 50 --delta 0.001 --lr-scheduler "StepLR" --step-size 50 --lr-gamma 0.33 --amp True
```
This will generate the [save_weights_RibFrac] folder containing det, seg, and tensorboard_logs subfolders to monitor all metrics during the training process.📈


If you want to train a different model architecture, find the corresponding architecture and modify it (prediction and evaluation scripts are the same). We have configured all architectures; you only need to comment out the unused architectures and select the desired one. For example:

```python
from maskrcnn_ribfrac.backbone import resnet50_fpn_backbone

backbone = resnet50_fpn_backbone()
# from backbone import resnet101_fpn_backbone
# backbone = resnet101_fpn_backbone()
# from maskrcnn_ribfrac.backbone import resnet152_fpn_backbone
# backbone = resnet152_fpn_backbone()
```

## 🔍 Make Predictions

You can directly download our model weights for inference：

MaksR-cnn50 model weights: `RibFrac50.pth`
Link：https://pan.baidu.com/s/1GNxUEMgthdcKjkp-_5byjQ
Extraction Code：RibF

MaksR-cnn101 model weights: `RibFrac101.pth`
Link：https://pan.baidu.com/s/1hGyJop5q6OwFj9zpH2tVag 
Extraction Code：RibF

MaksR-cnn152 model weights: `RibFrac152.pth`
Link：https://pan.baidu.com/s/15Yxefnj5BUC5gfHIRwy9Dg 
Extraction Code：RibF

Use the following command to make predictions:

```bash
python run_maskrcnn/predict_RIBFrac.py --img_folder /path/to/input/images --output_folder /path/to/output --label_json_path cocorib_indices.json --model_id maskrcn152 --save_format jpg
```
For example: `--file_type dicom` or `--file_type nii` to predict different image formats.
In the prediction file, we use 0.3 as the threshold for object detection because it filters out very low probability objects, retaining better results.


## Model Evaluation

The run_maskrcnn/validation.py script is mainly used to evaluate the performance of the Mask R-CNN model on a COCO format dataset. By running this script, evaluation metrics such as mAP (mean Average Precision) can be generated, and the results will be saved to a text file. First, generate the det_results.json and seg_results.json files, then use the following command for evaluation:

```bash
python run_maskrcnn/validation.py --device cuda --num-classes 1 --data-path <image_path>  --weights-path <weights_path>  --label-json-path ./data/cocorib_indices.json --batch-size 1
```

This will generate the [det_record_mAP.txt] and [seg_record_mAP.txt] files.

Model training process：
Bbox：
![det_metrics_comparison.png](maskrcnn_ribfrac/assets/det_metrics_comparison.png)
Mask：
![seg_metrics_comparison.png](maskrcnn_ribfrac/assets/seg_metrics_comparison.png)

Note: We provide the original 2D slices. Readers can view the data loading script [run_maskrcnn/my_dataset_cocoRib.py]and comment out the visualization code to view our original data and data augmentation.You only need to change the image address.
For example:
![visualized_samples_trans](maskrcnn_ribfrac/assets/visualized_samples_trans.png)


Reference code： https://github.com/WZMIAOMIAO/deep-learning-for-image-processing and https://github.com/pytorch/vision. 
Acknowledgements WZMIAOMIAO！
