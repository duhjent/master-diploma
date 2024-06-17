import torchvision

# model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet101', pretrained=True)
model = torchvision.models.resnet101(weights='IMAGENET1K_V1')

