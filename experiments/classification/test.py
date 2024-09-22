from experiments.classification.data import DFGClassification
from torchvision.transforms import v2 as transforms
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

def main():
    metrics = torch.load('./outs/metrics-vgg.pth')
    print(metrics)

if __name__ == "__main__":
    main()
