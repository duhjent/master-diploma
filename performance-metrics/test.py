from resnet import model as resnet_model
# from fractalnet import model as fractalnet_model
from fractalnet_v2 import model as fractalnet_model
import torch
import torch.utils.data
import torchvision
from torchvision import transforms
from PIL import Image
import os
import time
from tqdm import tqdm

class ImageNetCustomLoader(torch.utils.data.Dataset):
    def __init__(self, base_dir, transformers):
        self._base_dir = base_dir
        self._transformers = transformers
        images = os.listdir(base_dir)
        self._images = images

    def __getitem__(self, idx):
        img = Image.open(f'{self._base_dir}/{self._images[idx]}')
        img = self._transformers(img)

        return img

    def __len__(self):
        return len(self._images)

# transformers = transforms.Compose(
#     [
#         transforms.PILToTensor(),
#         transforms.Resize((256, 256))
#     ]
# )

transformers = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(256),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

imagenet_data = ImageNetCustomLoader('./imagenet', transformers)
data_loader = torch.utils.data.DataLoader(imagenet_data,
                                          batch_size=1,
                                          shuffle=False)

with torch.no_grad():
    i = 0
    fractalnet_model.eval()
    fractal_start_time = time.time()
    for batch in data_loader:
        res = fractalnet_model(batch)
        i += 1
        if i == 3:
            break

    fractal_end_time = time.time()
    print(f'ended fractal in {fractal_end_time - fractal_start_time}')

    i = 0
    resnet_model.eval()
    resnet_start_time = time.time()
    for batch in data_loader:
        res = resnet_model(batch)
        i += 1
        if i == 3:
            break

    resnet_end_time = time.time()
    print(f'ended resnet in {resnet_end_time - resnet_start_time}')
