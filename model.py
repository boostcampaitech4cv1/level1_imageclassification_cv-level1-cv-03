import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class BaseModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.25)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout2(x)

        x = self.avgpool(x)
        x = x.view(-1, 128)
        return self.fc(x)


# Custom Model Template
class MyResNet50(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        """
        1. 위와 같이 생성자의 parameter 에 num_claases 를 포함해주세요.
        2. 나만의 모델 아키텍쳐를 디자인 해봅니다.
        3. 모델의 output_dimension 은 num_classes 로 설정해주세요.
        """
        
        self.ResNet50 = models.resnet50(pretrained=True)
        self.ResNet50.fc = nn.Linear(2048, 3)

        # for param in self.ResNet50.parameters():
            # param.requires_grad = False
    
        # for param in self.ResNet50.fc.parameters():
            # param.requires_grad = True

    def forward(self, x):
        """
        1. 위에서 정의한 모델 아키텍쳐를 forward propagation 을 진행해주세요
        2. 결과로 나온 output 을 return 해주세요
        """
        x = self.ResNet50(x)
        return x


# Custom Model Template
class MyResNet152(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        """
        1. 위와 같이 생성자의 parameter 에 num_claases 를 포함해주세요.
        2. 나만의 모델 아키텍쳐를 디자인 해봅니다.
        3. 모델의 output_dimension 은 num_classes 로 설정해주세요.
        """
        
        self.ResNet152 = models.resnet152(pretrained=True)
        self.ResNet152.fc = nn.Linear(in_features=2048, out_features=num_classes, bias=True)

        # for param in self.ResNet50.parameters():
            # param.requires_grad = False
    
        # for param in self.ResNet50.fc.parameters():
            # param.requires_grad = True

    def forward(self, x):
        """
        1. 위에서 정의한 모델 아키텍쳐를 forward propagation 을 진행해주세요
        2. 결과로 나온 output 을 return 해주세요
        """
        x = self.ResNet152(x)
        return x

class MyEfficientNetB4(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        """
        1. 위와 같이 생성자의 parameter 에 num_claases 를 포함해주세요.
        2. 나만의 모델 아키텍쳐를 디자인 해봅니다.
        3. 모델의 output_dimension 은 num_classes 로 설정해주세요.
        """
        
        self.EffNetB4 = models.efficientnet_b4(pretrained=True)
        self.EffNetB4.classifier[1] = nn.Linear(in_features=1792, out_features=num_classes, bias=True)

        # for param in self.ResNet50.parameters():
            # param.requires_grad = False
    
        # for param in self.ResNet50.classifier.parameters():
            # param.requires_grad = True

    def forward(self, x):
        """
        1. 위에서 정의한 모델 아키텍쳐를 forward propagation 을 진행해주세요
        2. 결과로 나온 output 을 return 해주세요
        """
        x = self.EffNetB4(x)
        return x

class MyEfficientNetB0(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        """
        1. 위와 같이 생성자의 parameter 에 num_claases 를 포함해주세요.
        2. 나만의 모델 아키텍쳐를 디자인 해봅니다.
        3. 모델의 output_dimension 은 num_classes 로 설정해주세요.
        """
        
        self.EffNetB0 = models.efficientnet_b0(pretrained=True)
        self.EffNetB0.classifier[1] = nn.Linear(in_features=1280, out_features=num_classes, bias=True)

        # for param in self.ResNet50.parameters():
            # param.requires_grad = False
    
        # for param in self.ResNet50.classifier.parameters():
            # param.requires_grad = True

    def forward(self, x):
        """
        1. 위에서 정의한 모델 아키텍쳐를 forward propagation 을 진행해주세요
        2. 결과로 나온 output 을 return 해주세요
        """
        x = self.EffNetB0(x)
        return x