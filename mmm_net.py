import torch
import numpy as np
import torch.nn as nn
from . import resnet
import random
import math
import torch.nn.functional as F
from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.base import modules as md

from segmentation_models_pytorch.base import (
    SegmentationHead,
    ClassificationHead,
    ClassificationHead_common
)

class DecoderBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels,
        use_batchnorm=True,
        attention_type=None,
    ):
        super().__init__()
        self.conv1 = md.Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.attention1 = md.Attention(attention_type, in_channels=in_channels + skip_channels)
        self.conv2 = md.Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.attention2 = md.Attention(attention_type, in_channels=out_channels)

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
            x = self.attention1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.attention2(x)
        return x

class CenterBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, use_batchnorm=True):
        conv1 = md.Conv2dReLU(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        conv2 = md.Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        super().__init__(conv1, conv2)
    
class UnetDecoder(nn.Module):
    def __init__(
        self,
        encoder_channels,
        decoder_channels,
        n_blocks=5,
        use_batchnorm=True,
        attention_type=None,
        center=False,
    ):
        super().__init__()

        if n_blocks != len(decoder_channels):
            raise ValueError(
                "Model depth is {}, but you provide `decoder_channels` for {} blocks.".format(
                    n_blocks, len(decoder_channels)
                )
            )

        # remove first skip with same spatial resolution
        encoder_channels = encoder_channels[1:]
        # reverse channels to start from head of encoder
        encoder_channels = encoder_channels[::-1]

        # computing blocks input and output channels
        head_channels = encoder_channels[0]
        in_channels = [head_channels] + list(decoder_channels[:-1])
        skip_channels = list(encoder_channels[1:]) + [0]
        out_channels = decoder_channels
        self.center = center
        if center:
            self.center = CenterBlock(head_channels, head_channels, use_batchnorm=use_batchnorm)
            # self.center = CenterBlock(in_channnel=encoder_channels[0])
        else:
            self.center = nn.Identity()

        # combine decoder keyword arguments
        kwargs = dict(use_batchnorm=use_batchnorm, attention_type=attention_type)
        blocks = [
            DecoderBlock(in_ch, skip_ch, out_ch, **kwargs)
            for in_ch, skip_ch, out_ch in zip(in_channels, skip_channels, out_channels)
        ]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, *features):
        features = features[1:]  # remove first skip with same spatial resolution
        features = features[::-1]  # reverse channels to start from head of encoder

        head = features[0]
        skips = features[1:]
        x = self.center(head)
        f_all = []
        f_all.append(x)
        for i, decoder_block in enumerate(self.blocks):
            skip = skips[i] if i < len(skips) else None
            x = decoder_block(x, skip)
            f_all.append(x)
        # return x
        return f_all

def get_r50_cla_unet():
        import segmentation_models_pytorch as smp
        aux_params=dict(
            pooling='avg',             # one of 'avg', 'max'
            dropout=0.5,               # dropout ratio, default is None
            activation=None,           # activation function, default is None
            classes=3,                 # define number of output labels
        )
        model = smp.create_model(
            "Unet", encoder_name="resnet50", encoder_weights="imagenet",in_channels=3, classes=3,att=False,aux_params=aux_params
        )
        return model
    
def initialize_decoder(module):
    for m in module.modules():

        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

def initialize_head(module):
    for m in module.modules():
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

class mmm_net(nn.Module):
    def __init__(
            self):

        super(mmm_net, self).__init__()
        # 已经初始化
        self.encoder_f = get_encoder(
            "resnet50",
            in_channels=3,
            depth=5,
            weights="imagenet",
        )
        # 已经初始化
        self.encoder_o = get_encoder(
            "resnet50",
            in_channels=256,
            depth=5,
            weights="imagenet",
        )
        
        self.decoder = UnetDecoder(
            encoder_channels=self.encoder_f.out_channels,
            decoder_channels=(256, 128, 64, 32, 16),
            n_blocks=5,
            use_batchnorm=True,
            center=False,
            attention_type=None,
        )
        
        self.segmentation_head = SegmentationHead(
            in_channels=16,
            out_channels=3,
            activation=None,
            kernel_size=3,
        )
        
        self.classification_head_f = ClassificationHead(in_channels=self.encoder_f.out_channels[-1],classes=3, pooling="avg", dropout=0.5, activation=None)
        
        self.classification_head_o = ClassificationHead(in_channels=self.encoder_o.out_channels[-1],classes=3, pooling="avg", dropout=0.5, activation=None)

        initialize_decoder(self.decoder)
        initialize_head(self.segmentation_head)
        initialize_head(self.classification_head_f)
        initialize_head(self.classification_head_o)

    # x1: cfp, x2: oct
    def forward(self, x1, x2):
        
        features_f = self.encoder_f(x1)
        features_o = self.encoder_o(x2)
        decoder_output = self.decoder(*features_f)
        masks = self.segmentation_head(decoder_output)
        labels_f = self.classification_head_f(features_f[-1])
        labels_o = self.classification_head_o(features_o[-1])
        
        return masks, labels_f, labels_o, features_f[-1], features_o[-1]
    
class mmm_net_1(nn.Module):
    def __init__(
            self, fc_ins=4096):

        super(mmm_net_1, self).__init__()
        # 已经初始化
        self.encoder_f = get_encoder(
            "resnet50",
            in_channels=3,
            depth=5,
            weights="imagenet",
        )
        # 已经初始化
        self.encoder_o = get_encoder(
            "resnet50",
            in_channels=256,
            depth=5,
            weights="imagenet",
        )
        
        self.decoder = UnetDecoder(
            encoder_channels=self.encoder_f.out_channels,
            decoder_channels=(256, 128, 64, 32, 16),
            n_blocks=5,
            use_batchnorm=True,
            center=False,
            attention_type=None,
        )
        
        self.segmentation_head = SegmentationHead(
            in_channels=16,
            out_channels=3,
            activation=None,
            kernel_size=3,
        )
        
        self.classification_head_f = ClassificationHead(in_channels=self.encoder_f.out_channels[-1],classes=3, pooling="avg", dropout=0.5, activation=None)
        
        self.classification_head_o = ClassificationHead(in_channels=self.encoder_o.out_channels[-1],classes=3, pooling="avg", dropout=0.5, activation=None)

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
    
    
        initialize_decoder(self.decoder)
        initialize_head(self.segmentation_head)
        initialize_head(self.classification_head_f)
        initialize_head(self.classification_head_o)
        
    

    # x1: cfp, x2: oct
    def forward(self, x1, x2):
        
        features_f = self.encoder_f(x1)
        features_o = self.encoder_o(x2)
        decoder_output = self.decoder(*features_f)
        masks = self.segmentation_head(decoder_output)
        labels_f = self.classification_head_f(features_f[-1])
        labels_o = self.classification_head_o(features_o[-1])
        
        return masks, labels_f, labels_o, features_f[-1], features_o[-1]
    
class classifier(nn.Module):
    def __init__(self, fc_ins=1024):
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

# common 无seg，每个模态的特征图直接下采样后concat
class mmm_net_common(nn.Module):
    def __init__(
            self):

        super(mmm_net_common, self).__init__()
        # 已经初始化
        self.encoder_f = get_encoder(
            "resnet34",
            in_channels=3,
            depth=5,
            weights="imagenet",
        )
        # 已经初始化
        self.encoder_o = get_encoder(
            "resnet34",
            in_channels=256,
            depth=5,
            weights="imagenet",
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        
        self.classification_head = ClassificationHead_common(in_channels=self.encoder_o.out_channels[-1]*2,classes=3, dropout=0.5)

        initialize_head(self.classification_head)

    # x1: cfp, x2: oct
    def forward(self, x1, x2):
        
        features_f = self.encoder_f(x1)
        features_o = self.encoder_o(x2)
        
        feat_f = self.avgpool(features_f[-1])
        feat_o = self.avgpool(features_o[-1])
        
        feat_f = feat_f.view(feat_f.size(0), -1)
        feat_o = feat_o.view(feat_o.size(0), -1)
        
        feat = torch.cat([feat_f, feat_o], 1)
        
        labels = self.classification_head(feat)
        return labels
    
class mmm_net_common_align(nn.Module):
    def __init__(
            self):

        super(mmm_net_common_align, self).__init__()
        # 已经初始化
        self.encoder_f = get_encoder(
            "resnet34",
            in_channels=3,
            depth=5,
            weights="imagenet",
        )
        # 已经初始化
        self.encoder_o = get_encoder(
            "resnet34",
            in_channels=256,
            depth=5,
            weights="imagenet",
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        
        self.classification_head = ClassificationHead_common(in_channels=self.encoder_o.out_channels[-1]*3,classes=3, dropout=0.5)

        self.align = classifier()
        
        initialize_head(self.classification_head)
        

    # x1: cfp, x2: oct
    def forward(self, x1, x2):
        
        features_f = self.encoder_f(x1)
        features_o = self.encoder_o(x2)
        
        feat_f = self.avgpool(features_f[-1])
        feat_o = self.avgpool(features_o[-1])
        
        feat_f = feat_f.view(feat_f.size(0), -1)
        feat_o = feat_o.view(feat_o.size(0), -1)
        feat_a = self.align(features_f[-1], features_o[-1])
        
        feat = torch.cat([feat_f, feat_o,feat_a], 1)
        
        labels = self.classification_head(feat)
        
        return labels
    
class mmm_net_common_align_only(nn.Module):
    def __init__(
            self):

        super(mmm_net_common_align_only, self).__init__()
        # 已经初始化
        self.encoder_f = get_encoder(
            "resnet34",
            in_channels=3,
            depth=5,
            weights="imagenet",
        )
        # 已经初始化
        self.encoder_o = get_encoder(
            "resnet34",
            in_channels=256,
            depth=5,
            weights="imagenet",
        )
        
        # self.avgpool = nn.AdaptiveAvgPool2d(1)
        
        # self.classification_head = ClassificationHead_common(in_channels=self.encoder_o.out_channels[-1]*3,classes=3, dropout=0.5)

        self.align = classifier()
        
        # initialize_head(self.classification_head)
        

    # x1: cfp, x2: oct
    def forward(self, x1, x2):
        
        print(x1.shape, x2.shape)
        
        features_f = self.encoder_f(x1)
        
        features_o = self.encoder_o(x2)
        
        labels = self.align(features_f[-1], features_o[-1])
        
        return labels

class mmm_net_seg_align(nn.Module):
    def __init__(
            self):

        super(mmm_net_seg_align, self).__init__()
        # 已经初始化
        self.encoder_f = get_encoder(
            "resnet34",
            in_channels=3,
            depth=5,
            weights="imagenet",
        )
        # 已经初始化
        self.encoder_o = get_encoder(
            "resnet34",
            in_channels=256,
            depth=5,
            weights="imagenet",
        )
        
        self.decoder = UnetDecoder(
            encoder_channels=self.encoder_f.out_channels,
            decoder_channels=(256, 128, 64, 32, 16),
            n_blocks=5,
            use_batchnorm=True,
            center=False,
            attention_type=None,
        )
        
        self.segmentation_head = SegmentationHead(
            in_channels=16,
            out_channels=3,
            activation=None,
            kernel_size=3,
        )
        self.avgpool = nn.AdaptiveAvgPool2d(1)    
        
        self.classification_head = ClassificationHead_common(in_channels=self.encoder_o.out_channels[-1]*3,classes=3, dropout=0.5)

        self.align = classifier()
        
        initialize_decoder(self.decoder)
        initialize_head(self.segmentation_head)
        initialize_head(self.classification_head)

    # x1: cfp, x2: oct
    def forward(self, x1, x2):
        
        features_f = self.encoder_f(x1)
        features_o = self.encoder_o(x2)
        
        decoder_output = self.decoder(*features_f)
        masks = self.segmentation_head(decoder_output)
        
        feat_f = self.avgpool(features_f[-1])
        feat_f = feat_f.view(feat_f.size(0), -1)
        
        feat_o = self.avgpool(features_o[-1])
        feat_o = feat_o.view(feat_o.size(0), -1)
        
        feat_a = self.align(features_f[-1], features_o[-1])
        
        feat = torch.cat([feat_f, feat_o,feat_a], 1)
        
        labels = self.classification_head(feat)        
        
        return masks, labels
   
class mmm_net_seg_align_FF(nn.Module):
    def __init__(
            self):

        super(mmm_net_seg_align_FF, self).__init__()
        # 已经初始化
        self.encoder_f = get_encoder(
            "resnet34",
            in_channels=3,
            depth=5,
            weights="imagenet",
        )
        # 已经初始化
        self.encoder_o = get_encoder(
            "resnet34",
            in_channels=256,
            depth=5,
            weights="imagenet",
        )
        
        self.decoder = UnetDecoder(
            encoder_channels=self.encoder_f.out_channels,
            decoder_channels=(256, 128, 64, 32, 16),
            n_blocks=5,
            use_batchnorm=True,
            center=False,
            attention_type=None,
        )
        
        self.segmentation_head = SegmentationHead(
            in_channels=16,
            out_channels=3,
            activation=None,
            kernel_size=3,
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.maxpool = nn.AdaptiveMaxPool2d(1)    
        
        self.classification_head = ClassificationHead_common(in_channels=256+512+512+512,classes=3, dropout=0.5)

        self.align = classifier()
        
        initialize_decoder(self.decoder)
        initialize_head(self.segmentation_head)
        initialize_head(self.classification_head)

    # x1: cfp, x2: oct
    def forward(self, x1, x2):
        
        # 提取特征
        features_f = self.encoder_f(x1)
        features_o = self.encoder_o(x2)
        
        decoder_output = self.decoder(*features_f)
        masks = self.segmentation_head(decoder_output[-1])
        
        feat_f3 = self.avgpool(features_f[-2])
        feat_f3 = feat_f3.view(feat_f3.size(0), -1)
        
        feat_f4 = self.avgpool(features_f[-1])
        feat_f4 = feat_f4.view(feat_f4.size(0), -1)
        
        feat_d5 = self.avgpool(decoder_output[0])
        feat_d5 = feat_d5.view(feat_d5.size(0), -1)
        
        # torch.Size([1, 256]) torch.Size([1, 512]) torch.Size([1, 512])
        # print(feat_f3.shape, feat_f4.shape, feat_d5.shape)
        
        feat_f = torch.cat([feat_f3, feat_f4, feat_d5], 1)
        
        feat_o = self.avgpool(features_o[-1])
        feat_o = feat_o.view(feat_o.size(0), -1)
        
        # feat_a = self.align(features_f[-1], features_o[-1])
        
        feat = torch.cat([feat_f, feat_o], 1)
        
        labels = self.classification_head(feat)        
        return masks, labels
    
class mmm_net_seg_align_MMFF(nn.Module):
    def __init__(
            self):

        super(mmm_net_seg_align_MMFF, self).__init__()
        # 已经初始化
        self.encoder_f = get_encoder(
            "resnet34",
            in_channels=3,
            depth=5,
            weights="imagenet",
        )
        # 已经初始化
        self.encoder_o = get_encoder(
            "resnet34",
            in_channels=256,
            depth=5,
            weights="imagenet",
        )
        
        self.decoder = UnetDecoder(
            encoder_channels=self.encoder_f.out_channels,
            decoder_channels=(256, 128, 64, 32, 16),
            n_blocks=5,
            use_batchnorm=True,
            center=False,
            attention_type=None,
        )
        
        self.segmentation_head = SegmentationHead(
            in_channels=16,
            out_channels=3,
            activation=None,
            kernel_size=3,
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.maxpool = nn.AdaptiveMaxPool2d(1)    
        
        self.classification_head = ClassificationHead_common(in_channels=256+512+512+512,classes=3, dropout=0.5)

        self.align = classifier()
        
        initialize_decoder(self.decoder)
        initialize_head(self.segmentation_head)
        initialize_head(self.classification_head)

    # x1: cfp, x2: oct
    def forward(self, x1, x2):
        
        # 提取特征
        features_f = self.encoder_f(x1)
        features_o = self.encoder_o(x2)
        
        decoder_output = self.decoder(*features_f)
        masks = self.segmentation_head(decoder_output[-1])
        
        feat_f3 = self.avgpool(features_f[-2])
        feat_f3 = feat_f3.view(feat_f3.size(0), -1)
        
        feat_f4 = self.avgpool(features_f[-1])
        feat_f4 = feat_f4.view(feat_f4.size(0), -1)
        
        feat_d5 = self.avgpool(decoder_output[0])
        feat_d5 = feat_d5.view(feat_d5.size(0), -1)
        
        # torch.Size([1, 256]) torch.Size([1, 512]) torch.Size([1, 512])
        # print(feat_f3.shape, feat_f4.shape, feat_d5.shape)
        
        feat_f = torch.cat([feat_f3, feat_f4, feat_d5], 1)
        
        feat_o = self.avgpool(features_o[-1])
        feat_o = feat_o.view(feat_o.size(0), -1)
        
        # feat_a = self.align(features_f[-1], features_o[-1])
        
        feat = torch.cat([feat_f, feat_o], 1)
        
        labels = self.classification_head(feat)        
        return masks, labels
 
# 纯净版 
class mmm_net_seg_baseline(nn.Module):
    def __init__(
            self):

        super(mmm_net_seg_baseline, self).__init__()
        # 已经初始化
        self.encoder_f = get_encoder(
            "resnet34",
            in_channels=3,
            depth=5,
            weights="imagenet",
        )
        # 已经初始化
        self.encoder_o = get_encoder(
            "resnet34",
            in_channels=256,
            depth=5,
            weights="imagenet",
        )
        
        self.decoder = UnetDecoder(
            encoder_channels=self.encoder_f.out_channels,
            decoder_channels=(256, 128, 64, 32, 16),
            n_blocks=5,
            use_batchnorm=True,
            center=False,
            attention_type=None,
        )
        
        self.segmentation_head = SegmentationHead(
            in_channels=16,
            out_channels=3,
            activation=None,
            kernel_size=3,
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classification_head = ClassificationHead_common(in_channels=512+512,classes=3, dropout=0.5)
        
        initialize_decoder(self.decoder)
        initialize_head(self.segmentation_head)
        initialize_head(self.classification_head)

    # x1: cfp, x2: oct
    def forward(self, x1, x2):
        
        # 提取特征
        features_f = self.encoder_f(x1)
        features_o = self.encoder_o(x2)
        
        decoder_output = self.decoder(*features_f)
        masks = self.segmentation_head(decoder_output[-1])
        
        feat_f = self.avgpool(features_f[-1])
        feat_f = feat_f.view(feat_f.size(0), -1)
        
        feat_o = self.avgpool(features_o[-1])
        feat_o = feat_o.view(feat_o.size(0), -1)
        
        feat = torch.cat([feat_f, feat_o], 1)
        
        labels = self.classification_head(feat)        
        return masks, labels
 
class oct_single(nn.Module):
    def __init__(
            self):

        super(oct_single, self).__init__()

        # 已经初始化
        self.encoder_o = get_encoder(
            "resnet34",
            in_channels=256,
            depth=5,
            weights="imagenet",
        )
    
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classification_head = ClassificationHead_common(in_channels=512,classes=3, dropout=0.5)
    
        initialize_head(self.classification_head)

    # x1: cfp, x2: oct
    def forward(self, x2):
        
        # 提取特征
        features_o = self.encoder_o(x2)
        
        feat_o = self.avgpool(features_o[-1])
        feat_o = feat_o.view(feat_o.size(0), -1)
        
        feat = feat_o
        labels = self.classification_head(feat)        
        
        return labels