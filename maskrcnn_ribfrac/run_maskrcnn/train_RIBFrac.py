import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import datetime
import torch
from maskrcnn_ribfrac.network_files import MaskRCNN
from maskrcnn_ribfrac.run_maskrcnn.my_dataset_cocoRib import CocoDetection
from maskrcnn_ribfrac.utils import train_eval_utils as utils, transforms
from maskrcnn_ribfrac.utils import GroupedBatchSampler, create_aspect_ratio_groups
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from torch.optim.lr_scheduler import StepLR
from torchvision.transforms import InterpolationMode

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

class EarlyStopping:
    """
    Early stopping the training if validation loss doesn't improve after a given patience.'
    """
    def __init__(self, patience=7, verbose=False, delta=0, trace_func=print):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.trace_func = trace_func
        self.best_score = None
        self.epochs_no_improve = 0
        self.early_stop = False

    def __call__(self, val_metric, model):
        score = val_metric
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.epochs_no_improve += 1
            if self.epochs_no_improve >= self.patience:
                self.early_stop = True
                if self.verbose:
                    self.trace_func("Early stopping")
        else:
            self.best_score = score
            self.epochs_no_improve = 0

def setup_seed(seed):
    """
    Sets seed for reproducibility.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def create_model(num_classes, load_pretrain_weights=False):
    """
    Here you can modify different network structures to train, if you want to use transfer learning, please download the pytorch official pre-training weights
    resnet50 imagenet weights url:"https://download.pytorch.org/models/resnet50-0676ba61.pth"
    resnet101 imagenet weights url: "https://download.pytorch.org/models/resnet101-63fe2227.pth"
    resnet152 imagenet weights url: "https://download.pytorch.org/models/resnet152-394f9c45.pth"

    Freeze one part of the backbone network and train only a few modules at the top to fine-tune.
    The trainable_layers=3 parameter does mean that only the last three layers of the backbone network (ResNet-50) are trained.

    - conv1: The original convolution layer.
    - layer1, layer2, layer3, layer4: These are subsequent residual modules, each containing multiple convolutional layers.
    When you set trainable_layers=3, it usually means the following: layer1 and layer2 are frozen and do not participate in training, and the parameters of these layers remain the pre-trained weights.
    layer3 and layer4 and subsequent layers (for example, FPN or RPN layers) are trainable, i.e. the weights of these layers are updated based on data for new tasks.

    Why only train the last three layers?
    1.Transfer learning efficiency: Freezing the weights of the first few layers can take advantage of the common features they learn, such as edges, textures, shapes, etc., that are common to most visual tasks.
    2.Reduce overfitting: By reducing the number of parameters that need to be trained, the risk of overfitting is reduced, especially when the training data is small.
    3.Accelerated training: Reduced computational effort makes model training faster because only a few parameters need to be updated.
    """
    from maskrcnn_ribfrac.backbone import resnet50_fpn_backbone
    backbone = resnet50_fpn_backbone(pretrain_path="../resnet50.pth", trainable_layers=3)
    # from maskrcnn_ribfrac.backbone import resnet101_fpn_backbone
    # backbone = resnet101_fpn_backbone(pretrain_path="resnet101.pth", trainable_layers=3)
    # from maskrcnn_ribfrac.backbone import resnet152_fpn_backbone
    # backbone = resnet152_fpn_backbone(pretrain_path="resnet152.pth", trainable_layers=3)

    model = MaskRCNN(backbone, num_classes=num_classes)

    # Further fine-tuning：Load the weights of the complete MaskR-CNN model trained on the COCO dataset.
    # Since pytorch does not provide pre-trained weights for maskrcnn101 and 152, you can simply comment out this part of the code when training 101 and 152 weights, which does not delay training.
    if load_pretrain_weights:
        # maskrcnn_resnet50_fpn_coco.pth weights url: "https://download.pytorch.org/models/maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth"
        weights_path = "../maskrcnn_resnet50_fpn_coco.pth"
        weights_dict = torch.load(weights_path, map_location="cpu")
        for k in list(weights_dict.keys()):
            if "box_predictor" in k or "mask_fcn_logits" in k:
                del weights_dict[k]
        new_state_dict = {}
        for k, v in weights_dict.items():
            if k in model.state_dict() and model.state_dict()[k].shape == v.shape:
                new_state_dict[k] = v
            else:
                print(f"Skipping {k} due to shape mismatch")
        missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
        if missing_keys:
            print(f"Missing keys: {missing_keys}")
        if unexpected_keys:
            print(f"Unexpected keys: {unexpected_keys}")
    # Let's comment it out here

    return model


def main(args):
    setup_seed(1)
    writer = SummaryWriter(log_dir=os.path.join(args.output_dir, 'tensorboard_logs'))
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    early_stopping = EarlyStopping(patience=args.patience, verbose=True, delta=args.delta)
    print(f"Using {device.type} device for training.")

    det_dir = os.path.join(args.output_dir, 'det')
    seg_dir = os.path.join(args.output_dir, 'seg')
    os.makedirs(det_dir, exist_ok=True)
    os.makedirs(seg_dir, exist_ok=True)
    det_results_file = os.path.join(det_dir, f"det_results_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}--{args.batch_size}-{args.lr}.txt")
    seg_results_file = os.path.join(seg_dir, f"seg_results_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}--{args.batch_size}-{args.lr}.txt")

    data_transform = {
        "train": transforms.Compose([
            transforms.ToTensor(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.RandomHorizontalFlip(prob=0.5),
            transforms.RandomVerticalFlip(prob=0.5),
            transforms.RandomRotateCounterClockwise90(prob=0.5),
            transforms.RandomRotatealittle(angle_range=(-90, 90), prob=0.5),
            transforms.RandomIoUCrop(),
            transforms.RandomPhotometricDistort(
                contrast=(0.5, 1.5),
                saturation=(0.5, 1.5),
                hue=(-0.05, 0.05),
                brightness=(0.875, 1.125),
                p=0.5
            ),
            transforms.ScaleJitter(
                target_size=(512, 512),
                scale_range=(0.7, 1.3),
                interpolation=InterpolationMode.BILINEAR,
                antialias=True
            ),
            transforms.RandomZoomOut(
                fill=[0, 0, 0],
                side_range=(1, 1.5),
                p=0.5
            ),
            transforms.RandomShortestSize(
                min_size=[512],
                max_size=1024,
                interpolation=InterpolationMode.BILINEAR,
                antialias=True
            ),
        ]),
        "val": transforms.Compose([
            transforms.ToTensor(),
            transforms.ToDtype(torch.float32, scale=True)
        ])
    }

    data_root = args.data_path
    train_dataset = CocoDetection(data_root, "train", data_transform["train"])
    train_sampler = None

    if args.aspect_ratio_group_factor >= 0:
        train_sampler = torch.utils.data.RandomSampler(train_dataset)
        group_ids = create_aspect_ratio_groups(train_dataset, k=args.aspect_ratio_group_factor)
        train_batch_sampler = GroupedBatchSampler(train_sampler, group_ids, args.batch_size)

    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    print('Using %g dataloader workers' % nw)

    if train_sampler:
        train_data_loader = torch.utils.data.DataLoader(train_dataset,
                                                        batch_sampler=train_batch_sampler,
                                                        pin_memory=True,
                                                        num_workers=nw,
                                                        collate_fn=train_dataset.collate_fn)
    else:
        train_data_loader = torch.utils.data.DataLoader(train_dataset,
                                                        batch_size=batch_size,
                                                        shuffle=True,
                                                        pin_memory=True,
                                                        num_workers=nw,
                                                        collate_fn=train_dataset.collate_fn)


    val_dataset = CocoDetection(data_root, "val", data_transform["val"])
    val_data_loader = torch.utils.data.DataLoader(val_dataset,
                                                  batch_size=1,
                                                  shuffle=False,
                                                  pin_memory=True,
                                                  num_workers=nw,
                                                  collate_fn=train_dataset.collate_fn)

    model = create_model(num_classes=args.num_classes + 1, load_pretrain_weights=args.pretrain)
    model.to(device)

    train_loss = []
    learning_rate = []
    val_map = []

    best_val_map = 0.0  # Initialize best validation mAP
    best_model_path = None  # Initialize path for best model

    # define optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    scaler = torch.cuda.amp.GradScaler() if args.amp else None
    lr_scheduler = StepLR(optimizer,
                          step_size=args.step_size,
                          gamma=args.lr_gamma)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1
        if args.amp and "scaler" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler"])

    for epoch in range(args.start_epoch, args.epochs):
        mean_loss, lr, losses = utils.train_one_epoch(model, optimizer, train_data_loader,
                                              device, epoch, print_freq=300,
                                              warmup=True, scaler=scaler)

        train_loss.append(mean_loss.item())
        learning_rate.append(lr)
        # 记录损失和学习率
        writer.add_scalar('Loss/train', mean_loss, epoch)
        writer.add_scalar('Learning_rate', lr, epoch)

        writer.add_scalar('Loss/loss_classifier', losses['loss_classifier'], epoch)
        writer.add_scalar('Loss/loss_box_reg', losses['loss_box_reg'], epoch)
        writer.add_scalar('Loss/loss_mask', losses['loss_mask'], epoch)
        writer.add_scalar('Loss/loss_objectness', losses['loss_objectness'], epoch)
        writer.add_scalar('Loss/loss_rpn_box_reg', losses['loss_rpn_box_reg'], epoch)

        lr_scheduler.step()

        if epoch % args.validation_frequency == 0:
            coco_info, seg_info = utils.evaluate(model, val_data_loader, device=device)
            if coco_info is not None:
                val_mAP = coco_info[1]
                early_stopping(val_mAP, model)
                if early_stopping.early_stop:
                    print(f"Stopping early at epoch {epoch + 1}")
                    break

                if val_mAP > best_val_map and epoch > 50:
                    best_val_map = val_mAP
                    save_path = os.path.join(args.output_dir, "best_model.pth")
                    if best_model_path and os.path.exists(best_model_path):
                        os.remove(best_model_path)
                    best_model_path = save_path
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'scaler': scaler.state_dict() if args.amp else None
                    }, save_path)
                    print(f"Saved best model checkpoint to {save_path}")

            log_model_parameters(writer, model, epoch)

        metric_names = [
            "IoU=0.50:0.95_all", "IoU=0.50_all", "IoU=0.75_all",
            "IoU=0.50:0.95_small", "IoU=0.50:0.95_medium", "IoU=0.50:0.95_large",
            "AR1_all", "AR10_all", "AR100_all",
            "AR100_small", "AR100_medium", "AR100_large",
            "loss", "lr"
        ]
        det_info_dict = dict(zip(metric_names, coco_info))
        if det_info_dict:
            for k, v in det_info_dict.items():
                writer.add_scalar(f'Val/det_{k}', v, epoch)

        seg_info_dict = dict(zip(metric_names, seg_info))
        if seg_info_dict:
            for k, v in seg_info_dict.items():
                writer.add_scalar(f'Val/seg_{k}', v, epoch)

        with open(det_results_file, "a") as f:
            result_info = [f"{i:.4f}" for i in coco_info + [mean_loss.item()]] + [f"{lr:.6f}"]
            txt = "epoch:{} {}".format(epoch, '  '.join(result_info))
            f.write(txt + "\n")

        with open(seg_results_file, "a") as f:
            result_info = [f"{i:.4f}" for i in seg_info + [mean_loss.item()]] + [f"{lr:.6f}"]
            txt = "epoch:{} {}".format(epoch, '  '.join(result_info))
            f.write(txt + "\n")

        val_map.append(coco_info[1])

    writer.close()

    if len(train_loss) != 0 and len(learning_rate) != 0:
        from maskrcnn_ribfrac.utils.plot_curve import plot_loss_and_lr
        plot_loss_and_lr(train_loss, learning_rate)

    if len(val_map) != 0:
        from maskrcnn_ribfrac.utils.plot_curve import plot_map
        plot_map(val_map)

def log_model_parameters(writer, model, epoch):
    for name, param in model.named_parameters():
        writer.add_histogram(f'Params/{name}', param, epoch)
        if param.grad is not None:
            writer.add_histogram(f'Gradients/{name}', param.grad, epoch)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=__doc__)

    parser.add_argument('--device', default='cuda:0', help='device')
    parser.add_argument('--data-path', default=r"", help='dataset')
    parser.add_argument('--num-classes', default=1, type=int, help='num_classes, not including the background')
    parser.add_argument('--output-dir', default='./save_weights_RibFrac', help='path where to save')
    parser.add_argument('--resume', default='', type=str, help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, help='start epoch')
    parser.add_argument('--epochs', default=300, type=int, metavar='N',
                        help='number of total epochs to run_maskrcnn')
    parser.add_argument('--lr', default=0.01, type=float,
                        help='initial learning rate, 0.01 is the default value for training')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--batch_size', default=16, type=int, metavar='N',
                        help='batch size when training.')
    parser.add_argument('--aspect-ratio-group-factor', default=3, type=int)
    parser.add_argument("--pretrain", type=bool, default=True, help="load COCO pretrain weights.")
    parser.add_argument("--amp", default=True, help="Use torch.cuda.amp for mixed precision training")
    parser.add_argument('--step-size', default=50, type=int, help='Step size for StepLR')
    parser.add_argument('--lr-gamma', default=0.33, type=float, help='decrease lr by a factor of lr-gamma')
    parser.add_argument('--validation-frequency', type=int, default=1, help='Frequency of validation in epochs')
    parser.add_argument('--lr-scheduler', default='', help='Type of scheduler to use')
    parser.add_argument('--patience', type=int, default=50, help='Patience for early stopping')
    parser.add_argument('--delta', type=float, default=0.001, help='Minimum change to qualify as an improvement')

    args = parser.parse_args()
    print(args)

    args.output_dir = os.path.join(args.output_dir, f"lr-{args.lr}_{args.lr_scheduler}")

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    main(args)


