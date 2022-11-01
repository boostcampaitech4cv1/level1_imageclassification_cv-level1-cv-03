import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import timm


# Custom Model Template
class ModelMask(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = models.efficientnet_b0(pretrained=True)
        self.backbone.classifier[1] = nn.Linear(self.backbone.classifier[1].in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)


class Resnet50(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = models.resnet50(pretrained=True)
        n_input = self.backbone.fc.in_features
        last = nn.Linear(n_input, num_classes)
        self.backbone.fc = last 

    def forward(self, x):
        x = self.backbone(x)
        return x


class Resnet152(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = models.resnet152(pretrained=True)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)


class VGG19(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = models.vgg19(pretrained=True)
        self.backbone.classifier[6] = nn.Linear(self.backbone.classifier[6].in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)


class GoogLeNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = models.googlenet(pretrained=True)
        n_input = self.backbone.fc.in_features
        last = nn.Linear(n_input, num_classes)
        self.backbone.fc = last

    def forward(self, x):
        x = self.backbone(x)
        return x