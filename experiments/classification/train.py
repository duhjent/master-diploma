from fractalnet.layers.github.fractal_net import FractalNet
import torch
from torchvision.transforms import v2 as transforms
from .data import DFGClassification
from torch.utils.data import DataLoader, random_split
from torch.nn import functional as F
import argparse
from tqdm import tqdm
from torch import nn
from torch.utils.tensorboard import SummaryWriter

class_weights = torch.tensor(
    [
        1.00685,
        1.00685,
        1.00685,
        1.00685,
        1.00685,
        1.00685,
        1.00685,
        1.00685,
        1.00685,
        1.00685,
        1.00685,
        1.00685,
        1.00685,
        1.00685,
        1.00685,
        1.00685,
        1.00685,
        1.00685,
        1.00685,
        1.00685,
        1.00685,
        1.00685,
        1.00685,
        1.00685,
        1.00685,
        1.00685,
        1.00685,
        1.0018408,
        1.00685,
        1.00685,
        1.00685,
        1.00685,
        1.00685,
        1.00685,
        1.00685,
        1.00685,
        1.00685,
        0.58537791,
        1.00685,
        1.00685,
        1.00685,
        1.00685,
        1.00685,
        0.87552174,
        1.00685,
        1.01190955,
        1.01190955,
        1.00685,
        1.00685,
        1.00685,
        1.01190955,
        1.00685,
        1.00685,
        1.01190955,
        1.00685,
        1.00685,
        1.00685,
        1.00685,
        1.00685,
        1.0170202,
        1.00685,
        1.00685,
        1.00685,
        1.01190955,
        1.00685,
        1.00685,
        1.00685,
        1.00685,
        1.00685,
        1.01190955,
        1.00685,
        1.00685,
        1.00685,
        1.00685,
        1.00685,
        1.00685,
        1.00685,
        1.00685,
        1.00685,
        1.00685,
        1.00685,
        1.00685,
        1.00685,
        1.00685,
        1.00685,
        1.00685,
        0.95436019,
        1.00685,
        1.00685,
        1.00685,
        1.00685,
        1.00685,
        1.00685,
        1.00685,
        1.00685,
        1.00685,
        1.0018408,
        1.00685,
        1.00685,
        1.00685,
        1.00685,
        1.00685,
        1.00685,
        1.00685,
        1.00685,
        1.00685,
        1.00685,
        1.00685,
        1.00685,
        1.00685,
        1.00685,
        1.00685,
        1.00685,
        1.00685,
        1.00685,
        1.00685,
        1.00685,
        1.00685,
        1.00685,
        0.95436019,
        1.00685,
        1.00685,
        1.00685,
        0.99688119,
        1.00685,
        1.00685,
        1.00685,
        1.00685,
        1.00685,
        1.00685,
        1.00685,
        1.00685,
        1.00685,
        1.00685,
        1.00685,
        1.00685,
        0.97280193,
        1.00685,
        1.00685,
        1.00685,
        1.00685,
        1.00685,
        1.00685,
        1.00685,
        1.00685,
        1.00685,
        1.00685,
        1.00685,
        1.04336788,
        1.03266667,
        1.00685,
        1.00685,
        1.0018408,
        1.00685,
        1.00685,
        1.00685,
        1.00685,
        1.00685,
        1.00685,
        1.00685,
        1.00685,
        1.00685,
        1.00685,
        1.00685,
        1.00685,
        1.00685,
        1.00685,
        1.00685,
        1.00685,
        1.00685,
        1.00685,
        1.00685,
        1.00685,
        0.84609244,
        1.00685,
        1.00685,
        1.00685,
        1.00685,
        1.0018408,
        0.40928862,
        1.00685,
        1.00685,
        1.00685,
        1.00685,
        1.02218274,
        1.00685,
        1.00685,
        1.29083333,
        1.29083333,
        1.65057377,
        1.11872222,
        1.06544974,
        1.08848649,
        1.28261146,
        1.24302469,
        1.39840278,
        0.6580719,
        1.0018408,
        1.00685,
        1.00685,
    ],
    dtype=torch.float32,
)

parser = argparse.ArgumentParser()
parser.add_argument("--img_path", type=str, required=True)
parser.add_argument("--annot_path", type=str, required=True)
parser.add_argument("--model", type=str, default="fractal", choices=["fractal", "vgg"])
parser.add_argument("--num_epochs", type=int, required=True)
parser.add_argument("--num_workers", type=int, default=0)
parser.add_argument("--batch_size", type=int, required=True)
parser.add_argument("--lr", type=float, required=True)
parser.add_argument("--momentum", type=float, required=True)
parser.add_argument("--cuda", action="store_true")
parser.add_argument("--out_dir", type=str, default="./weights")
parser.add_argument("--train_ratio", type=float, default=0.8)
parser.add_argument("--val_ratio", type=float, default=0.2)

args = parser.parse_args()


def main():
    device = "cuda" if args.cuda else "cpu"

    if args.model == "fractal":
        model = FractalNet(
            data_shape=(3, 64, 64, 200),
            n_columns=4,
            init_channels=128,
            p_ldrop=0.15,
            dropout_probs=[0, 0.1, 0.2, 0.3],
            gdrop_ratio=0.5,
            # doubling=True,
        ).to(device)

    writer = SummaryWriter(log_dir="runs", comment="classification")

    print(args)

    dataset = DFGClassification(
        args.img_path,
        args.annot_path,
        transform=transforms.Compose(
            [
                transforms.ToDtype(torch.float32, scale=True),
                transforms.Normalize(
                    [0.4397, 0.4331, 0.4526], [0.2677, 0.2707, 0.2906]
                ),
                transforms.Resize((64, 64)),
            ]
        ),
    )

    train_ds, val_ds = random_split(dataset, [args.train_ratio, args.val_ratio])

    train_dl = DataLoader(
        train_ds, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True
    )
    val_dl = DataLoader(val_ds, batch_size=args.batch_size)

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    # criterion = nn.CrossEntropyLoss(class_weights)
    criterion = nn.CrossEntropyLoss()

    global_iter = 0

    for epoch in range(args.num_epochs):
        model.train()
        running_loss = 0
        img, tgt = next(iter(train_dl))
        tgt = F.one_hot(tgt.to(device), 200).to(torch.float32)
        # for iter, (img, tgt) in enumerate(train_dl):
        for iter_num in range(1):
            img = img.to(device)
            # tgt = F.one_hot(tgt.to(device), 200).to(torch.float32)

            out = model(img)

            loss = criterion(out, tgt)
            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

            running_loss += loss.item()
            print(loss.item())
            writer.add_scalar("ImmediateLoss/train", loss.item(), global_iter)
            global_iter += 1

        continue

        epoch_loss = running_loss / (iter_num + 1)
        writer.add_scalar("Loss/train", epoch_loss, epoch)

        model.eval()
        val_running_loss = 0
        with torch.no_grad():
            for iter_num, (img, tgt) in enumerate(val_dl):
                img = img.to(device)
                tgt = F.one_hot(tgt.to(device), 200)

                out = model(img)
                loss = criterion(out, tgt)
                val_running_loss = loss.item()

        writer.add_scalar("Loss/val", val_running_loss / (iter_num + 1), epoch)


if __name__ == "__main__":
    main()
