import torch
import numpy as np
import torch.nn as nn
from . import resnet
import random
import math
import torch.nn.functional as F
from torchvision.models import resnet34

def init_resnet50(num_classes=4, pretrained=True, heatmap=False, early=False):
    model = resnet.resnet50(pretrained=pretrained, heatmap=heatmap)
    model.avgpool.kernel_size = 14
    return model
def init_resnet34(num_classes=4, pretrained=True, heatmap=False, early=False):
    model = resnet.resnet34(pretrained=pretrained, heatmap=heatmap)
    model.avgpool.kernel_size = 14
    return model

def get_r50_cla_unet():
        import segmentation_models_pytorch as smp
        aux_params=dict(
            pooling='avg',             # one of 'avg', 'max'
            dropout=0.5,               # dropout ratio, default is None
            activation=None,           # activation function, default is None
            classes=3,                 # define number of output labels
        )
        model = smp.create_model(
            "Unet", encoder_name="resnet50", encoder_weights="imagenet",in_channels=3, classes=3,att=False,aux_params=None
        )
        return model

class CFP_OCT_Single(nn.Module):
    def __init__(
            self, fc_ins=1024, num_classes=3):

        super(CFP_OCT_Single, self).__init__()
        # 2D CFP
        self.model1 = init_resnet50(pretrained=True, heatmap=False, num_classes=3, early=False)
        # 3D OCT
        self.model2 = init_resnet50(pretrained=True, heatmap=False, num_classes=3, early=False)
        self.model2.conv1 = nn.Conv2d(256, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        self.fc_cat = nn.Linear(fc_ins // 2, num_classes)
        self.gap = nn.AvgPool2d(14, stride=1)

        self.conv1 = nn.Conv2d(fc_ins, fc_ins, kernel_size=3, stride=1, padding=1, groups=fc_ins, bias=False)
        self.bn1 = nn.BatchNorm2d(fc_ins)
        self.conv1_1 = nn.Conv2d(fc_ins, fc_ins // 2, kernel_size=1, groups=1, bias=False)
        self.bn1_1 = nn.BatchNorm2d(fc_ins // 2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x1, x2):
        x1, map_x1 = self.model1(x1)
        x2, map_x2 = self.model2(x2)
        map_x = torch.cat([map_x1, map_x2], 1)

        x = F.relu(self.bn1((self.conv1(map_x))))
        x = F.relu(self.bn1_1(self.conv1_1(x)))
        x = self.gap(x)

        x = x.view(x.size(0), -1)
        x = self.fc_cat(x)
        return x
    
class CFPMT_OCT(nn.Module):
    def __init__(
            self, fc_ins=1024, num_classes=3):

        super(CFPMT_OCT, self).__init__()
        # 3D OCT
        self.model2 = init_resnet50(pretrained=True, heatmap=False, num_classes=3, early=False)
        self.model2.conv1 = nn.Conv2d(256, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        self.fc_cat = nn.Linear(fc_ins // 2, num_classes)
        self.gap = nn.AvgPool2d(14, stride=1)

        self.conv1 = nn.Conv2d(fc_ins, fc_ins, kernel_size=3, stride=1, padding=1, groups=fc_ins, bias=False)
        self.bn1 = nn.BatchNorm2d(fc_ins)
        self.conv1_1 = nn.Conv2d(fc_ins, fc_ins // 2, kernel_size=1, groups=1, bias=False)
        self.bn1_1 = nn.BatchNorm2d(fc_ins // 2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, oct_img, cfp_map):
        map_x1 = cfp_map
        x2, map_x2 = self.model2(oct_img)
        map_x = torch.cat([map_x1, map_x2], 1)

        x = F.relu(self.bn1((self.conv1(map_x))))
        x = F.relu(self.bn1_1(self.conv1_1(x)))
        x = self.gap(x)

        x = x.view(x.size(0), -1)
        x = self.fc_cat(x)
        return x
    
class CFPMT_OCT_1(nn.Module):
    def __init__(
            self, fc_ins=1024, num_classes=3):

        super(CFPMT_OCT_1, self).__init__()
        # 3D OCT
        self.model2 = init_resnet50(pretrained=True, heatmap=False, num_classes=3, early=False)
        self.model2.conv1 = nn.Conv2d(256, 64, kernel_size=7, stride=2, padding=3, bias=False)

    def forward(self, oct_img, cfp_map):
        map_x1 = cfp_map
        x2, map_x2 = self.model2(oct_img)
        map_x = torch.cat([map_x1, map_x2], 1)

        x = F.relu(self.bn1((self.conv1(map_x))))
        x = F.relu(self.bn1_1(self.conv1_1(x)))
        x = self.gap(x)

        x = x.view(x.size(0), -1)
        x = self.fc_cat(x)
        return x
    
class classifier(nn.Module):
    def __init__(self, fc_ins=4096):
        super(classifier, self).__init__()
        
        self.fc_cat = nn.Linear(fc_ins // 2, 3)
        self.gap = nn.AvgPool2d(14, stride=1)

        self.conv1 = nn.Conv2d(fc_ins, fc_ins, kernel_size=3, stride=1, padding=1, groups=fc_ins, bias=False)
        self.bn1 = nn.BatchNorm2d(fc_ins)
        self.conv1_1 = nn.Conv2d(fc_ins, fc_ins // 2, kernel_size=1, groups=1, bias=False)
        self.bn1_1 = nn.BatchNorm2d(fc_ins // 2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, map_cfp, map_oct):
        map_x = torch.cat([map_cfp, map_oct], 1)

        x = F.relu(self.bn1((self.conv1(map_x))))
        x = F.relu(self.bn1_1(self.conv1_1(x)))
        x = self.gap(x)

        x = x.view(x.size(0), -1)
        x = self.fc_cat(x)
        return x

class MLP(nn.Module):
    def __init__(self, fc_ins=1000):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(fc_ins, fc_ins)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(fc_ins, fc_ins)
        self.relu2 = nn.ReLU()
        self.linear3 = nn.Linear(fc_ins, fc_ins)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu2(x)
        x = self.linear3(x)
        return x