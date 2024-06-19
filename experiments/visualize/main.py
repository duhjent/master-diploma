from dataset import MTSDDetectionDataset, visualize
from torchvision.transforms import v2 as transforms

if __name__ == "__main__":
    ds = MTSDDetectionDataset(
        "./data",
        "val",
        transform=transforms.Compose(
            [
                transforms.ToImage(),
            ]
        ),
        skip_validation=True,
        objects_filter=lambda obj: obj["label"] != "other-sign",
        download=True
    )

    visualize(*ds[0], width=5)
