import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


# Custom Model Template
class ModelAge(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = models.efficientnet_b0(pretrained=True)
        self.backbone.classifier[1] = nn.Linear(self.backbone.classifier[1].in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)

############################ My Model ############################
class EffNetB0(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = models.efficientnet_b0(pretrained=True)
        self.backbone.classifier[1] = nn.Linear(self.backbone.classifier[1].in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)

class EffNetB4(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        
        self.EffNetB4 = models.efficientnet_b4(pretrained=True)
        self.EffNetB4.classifier[1] = nn.Linear(in_features=1792, out_features=num_classes, bias=True)

    def forward(self, x):

        x = self.EffNetB4(x)
        return x


class VGG19(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        
        self.VGG19 = models.vgg19(pretrained=True)
        self.VGG19.classifier[6] = nn.Linear(in_features=4096, out_features=num_classes, bias=True)

    def forward(self, x):

        x = self.VGG19(x)
        return x


class ResNet50(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.ResNet50 = models.resnet50(pretrained=True)
        self.ResNet50.fc = nn.Linear(2048, 3)

    def forward(self, x):

        x = self.ResNet50(x)
        return x

class ResNet152(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        
        self.ResNet152 = models.resnet152(pretrained=True)
        self.ResNet152.fc = nn.Linear(in_features=2048, out_features=num_classes, bias=True)

    def forward(self, x):

        x = self.ResNet152(x)
        return x
##################################################################