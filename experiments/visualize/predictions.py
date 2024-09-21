import torch
from torch import random
from torchvision.datasets import CocoDetection as TvCocoDetection
from torchvision.datasets import wrap_dataset_for_transforms_v2
from experiments.ssd.ssd import build_ssd
from experiments.ssd.data import detection_collate
from torchvision.transforms import v2 as transforms
from torch.utils.data import DataLoader
from dataset import CocoWrapperDataset
from torchvision.utils import draw_bounding_boxes
import os
import matplotlib.pyplot as plt

dataset_root = "./data/dfg"

# model = build_ssd('test', 300, 201)
# model.load_state_dict(torch.load('./weights/DFG-initial.pth', map_location='cpu'))


dataset = CocoWrapperDataset(
    wrap_dataset_for_transforms_v2(
        TvCocoDetection(
            os.path.join(dataset_root, "JPEGImages"),
            os.path.join(dataset_root, "train.json"),
            transforms=transforms.Compose(
                [
                    transforms.ToImage(),
                    transforms.ToDtype(torch.float32, scale=True),
                    transforms.Normalize(
                        [0.4649, 0.4758, 0.4479], [0.2797, 0.2809, 0.2897]
                    ),
                    transforms.Resize(300),
                    transforms.RandomIoUCrop(
                        sampler_options=[0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
                    ),
                    # transforms.RandomCrop(300),
                    transforms.Resize((300, 300)),
                    transforms.SanitizeBoundingBoxes(),
                ]
            ),
        ),
        target_keys=("boxes", "labels"),
    )
)

inverse_normalize = transforms.Compose(
    [
        transforms.Normalize([0.0, 0.0, 0.0], [1 / 0.2797, 1 / 0.2809, 1 / 0.2897]),
        transforms.Normalize(
            [-0.4649, -0.4758, -0.4479],
            [
                1.0,
                1.0,
                1.0,
            ],
        ),
        transforms.ToDtype(torch.uint8, scale=True)
    ]
)

idx = 137
img, tgt = dataset._dataset[idx]

print(tgt)
tgt["boxes"] += (torch.randn(tgt["boxes"].shape) * 5).type(torch.int)
print(tgt)

img1 = draw_bounding_boxes(inverse_normalize(img), tgt["boxes"][1:])
plt.imshow(transforms.functional.to_pil_image(img1))
plt.show()
exit()


loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=detection_collate)

img, tgt = dataset._dataset[0]
print(tgt)
exit()

img, tgt = next(iter(loader))

model(img)
