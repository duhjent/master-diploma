from experiments.ssd.data import *
from experiments.ssd.utils.augmentations import SSDAugmentation
from experiments.ssd.layers.modules import MultiBoxLoss
from experiments.ssd.ssd import build_ssd, build_ssd_fractal, build_ssd_resnet
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.utils.data as data
import argparse
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import CocoDetection as TvCocoDetection, wrap_dataset_for_transforms_v2
from torchvision.transforms import v2 as transforms
from dataset import CocoWrapperDataset
from tqdm import tqdm
from torch.optim.lr_scheduler import MultiStepLR


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


def parse_array(s: str) -> list[int]:
    return [int(x) for x in s.split(",")]


parser = argparse.ArgumentParser(
    description="Single Shot MultiBox Detector Training With Pytorch"
)
train_set = parser.add_mutually_exclusive_group()
parser.add_argument(
    "--dataset", default="VOC", choices=["VOC", "COCO", "DFG"], type=str, help="VOC, COCO or DFG"
)
parser.add_argument(
    "--dataset_root", default=VOC_ROOT, help="Dataset root directory path"
)
parser.add_argument(
    "--basenet", default=None, type=str, help="Pretrained base model"
)
parser.add_argument("--num_epochs", default=100, type=int, help="Number of epochs")
parser.add_argument(
    "--batch_size", default=32, type=int, help="Batch size for training"
)
parser.add_argument(
    "--resume",
    default=None,
    type=str,
    help="Checkpoint state_dict file to resume training from",
)
parser.add_argument(
    "--start_iter", default=0, type=int, help="Resume training at this iter"
)
parser.add_argument(
    "--num_workers", default=4, type=int, help="Number of workers used in dataloading"
)
parser.add_argument(
    "--cuda", default=True, type=str2bool, help="Use CUDA to train model"
)
parser.add_argument(
    "--lr", "--learning-rate", default=1e-3, type=float, help="initial learning rate"
)
parser.add_argument(
    "--momentum", default=0.9, type=float, help="Momentum value for optim"
)
parser.add_argument(
    "--weight_decay", default=5e-4, type=float, help="Weight decay for SGD"
)
parser.add_argument("--gamma", default=0.1, type=float, help="Gamma update for SGD")
parser.add_argument(
    "--save_folder", default="weights/", help="Directory for saving checkpoint models"
)
parser.add_argument("--model", default="vgg", choices=["vgg", "fractalnet", "resnet"], type=str, help="vgg, fractalnet or resnet backbone")
parser.add_argument("--comment", type=str, help="TensorBoard comment for run")
parser.add_argument("--lr_schedule", type=parse_array, help="LR Scheduler steps", default=[])
args = parser.parse_args()


if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)


def train():
    if args.dataset == "COCO":
        if args.dataset_root == VOC_ROOT:
            if not os.path.exists(COCO_ROOT):
                parser.error("Must specify dataset_root if specifying dataset")
            print(
                "WARNING: Using default COCO dataset_root because "
                + "--dataset_root was not specified."
            )
            args.dataset_root = COCO_ROOT
        cfg = coco
        dataset = COCODetection(
            root=args.dataset_root, transform=SSDAugmentation(cfg["min_dim"], MEANS)
        )
    elif args.dataset == "VOC":
        if args.dataset_root == COCO_ROOT:
            parser.error("Must specify dataset if specifying dataset_root")
        cfg = voc
        dataset = VOCDetection(
            root=args.dataset_root,
            transform=SSDAugmentation(cfg["min_dim"], MEANS),
            image_sets=[("2007", "trainval")],
        )
    elif args.dataset == "DFG":
        dataset = CocoWrapperDataset(wrap_dataset_for_transforms_v2(TvCocoDetection(
            os.path.join(args.dataset_root, "JPEGImages"),
            os.path.join(args.dataset_root, "train.json"),
            transforms=transforms.Compose(
                [
                    transforms.ToImage(),
                    transforms.ToDtype(torch.float32, scale=True),
                    transforms.Normalize(
                        [0.4649, 0.4758, 0.4479], [0.2797, 0.2809, 0.2897]
                    ),
                    transforms.RandomIoUCrop(sampler_options=[0.1, 0.3, 0.5, 0.7, 0.9, 1.0]),
                    transforms.Resize((300, 300)),
                    transforms.SanitizeBoundingBoxes(),
                ]
            ),
        ), target_keys=('boxes', 'labels')))
        cfg = dfg

    writer = SummaryWriter(comment=args.comment)

    device = "cuda" if args.cuda else "cpu"

    if args.model == 'vgg':
        ssd_net = build_ssd("train", cfg, cfg["min_dim"], cfg["num_classes"])
    elif args.model == 'fractalnet':
        ssd_net = build_ssd_fractal("train", cfg, cfg["min_dim"], cfg["num_classes"])
    elif args.model == 'resnet':
        ssd_net = build_ssd_resnet("train", cfg, cfg["min_dim"], cfg["num_classes"])
    net = ssd_net

    if args.cuda:
        device = "cuda"
        cudnn.benchmark = True

    if args.resume is not None:
        print("Resuming training, loading {}...".format(args.resume))
        ssd_net.load_weights(args.resume)
    elif args.basenet is not None:
        vgg_weights = torch.load(args.save_folder + args.basenet)
        print("Loading base network...")
        ssd_net.vgg.load_state_dict(vgg_weights)

    net = net.to(device)

    if not args.resume:
        print("Initializing weights...")
        # initialize newly added layers' weights with xavier method
        ssd_net.extras.apply(weights_init)
        ssd_net.loc.apply(weights_init)
        ssd_net.conf.apply(weights_init)

    optimizer = optim.SGD(
        net.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    criterion = MultiBoxLoss(
        cfg["num_classes"], 0.5, True, 0, True, 3, 0.5, False, device
    ).to(device)

    print("Loading the dataset...")

    print("Training SSD on:", dataset.name)
    print("Using the specified args:")
    print(args)

    global_iteration = 0

    data_loader = data.DataLoader(
        dataset,
        args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        collate_fn=detection_collate,
        # pin_memory=True,
    )

    scheduler = MultiStepLR(
        optimizer,
        args.lr_schedule,
        0.1,
    )
    # create batch iterator
    # images, targets = next(iter(data_loader))
    for epoch in tqdm(
        range(args.num_epochs), desc="epochs", position=0
    ):
        net.train()
        running_loss = 0
        with tqdm(
            enumerate(data_loader),
            desc="iterations",
            position=1,
            leave=False,
            total=len(data_loader),
        ) as pbatch:
            for iteration, (images, targets) in pbatch:
            # for iteration in range(1):
                images = images.to(device)
                targets = [target.to(device) for target in targets]

                # forward
                out = net(images)
                # backprop
                optimizer.zero_grad()
                loss_l, loss_c = criterion(out, targets)
                loss = loss_l + loss_c
                loss.backward()
                optimizer.step()

                writer.add_scalar("ImmediateLoss/train", loss.item(), global_step=global_iteration)
                pbatch.set_postfix(loss=loss.item())
                global_iteration += 1
                running_loss += loss.item()

        epoch_loss = running_loss / (iteration + 1)
        writer.add_scalar("Loss/train", epoch_loss, epoch)

        scheduler.step()

    torch.save(ssd_net.state_dict(), os.path.join(args.save_folder, f'{args.comment}.pth'))


def xavier(param):
    init.xavier_uniform(param)


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
        if m.bias is not None:
            m.bias.data.zero_()


if __name__ == "__main__":
    train()
