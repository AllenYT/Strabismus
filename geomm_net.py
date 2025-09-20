import torch
import numpy as np
import torch.nn as nn
from . import resnet
import random
import math
import torch.nn.functional as F
from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.base import modules as md
from typing import Optional

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

# 提取slice的特征，暂时不添加分割以及模态交互模块 
class geomm_net1(nn.Module):
    def __init__(
            self):
        super(geomm_net1, self).__init__()
        
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
            in_channels=1,
            depth=5,
            weights="imagenet",
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classification_head = ClassificationHead_common(in_channels=512+512,classes=3, dropout=0.5)
    
        initialize_head(self.classification_head)

    # x1: cfp, x2: oct
    def forward(self, x1, x2):
        
        # 提取特征
        features_f = self.encoder_f(x1)
        feat_f = self.avgpool(features_f[-1])
        feat_f = feat_f.view(feat_f.size(0), -1)
        
        oct_list = list(x2.split(1, dim=1))  # 将 OCT 切片拆分成列表，len(oct_list) = M
        feature_vectors = []  # 用于存储每个切片的特征向量
        masked_feature_maps = []  # 用于存储每个切片的特征图
        # Step 1: 对每个 OCT 切片提取特征并计算全局平均池化向量
        for oct_img in oct_list:
            oct_img = oct_img.squeeze(1)  # 去掉单通道维度，shape = (B, H, W)
            features = self.encoder_o(oct_img)  # 提取特征图，features[-1] shape = (B, C, H', W')
            feat_map = features[-1]  # 最后一层特征图
            
            # 全局平均池化并展平成 (B, C)
            feature_vector = self.avgpool(feat_map).view(feat_map.size(0), -1)  # shape = (B, C)
            feature_vectors.append(feature_vector)  # 存储特征向量
            masked_feature_maps.append(feat_map)  # 存储特征图
            
        # Step 2: 将所有特征向量堆叠成 (B, M, C)，M 是切片数量
        feature_vectors = torch.stack(feature_vectors, dim=1)  # shape = (B, M, C)

        # Step 3: 逐通道找到每个通道上响应最大的切片索引
        channel_indices = torch.argmax(feature_vectors, dim=1)  # shape = (B, C)    

        # Step 4: 根据每个通道的索引，从对应的切片中提取特征图
        B, C, H, W = masked_feature_maps[0].shape  # 获取特征图的形状
        selected_feature_maps = torch.zeros((B, C, H, W), device=feature_vectors.device)  # 初始化选择的特征图
        # # 遍历每个通道，选择对应的特征图
        for b in range(B):
            for c in range(C):
                # 获取样本 b、通道 c 的最佳切片索引
                best_slice_idx = channel_indices[b, c]
                # 从该切片中选择特征图对应的通道
                selected_feature_maps[b, c] = masked_feature_maps[best_slice_idx][b, c]

        # 最终的 selected_feature_maps shape = (B, C, H, W)
        combined_feature_map = selected_feature_maps  # 直接使用选中的特征图    
        
        # 对聚合后的 OCT 特征图进行全局平均池化，得到最终的特征向量
        feat_o = self.avgpool(combined_feature_map)
        feat_o = feat_o.view(feat_o.size(0), -1)
        
        # -----------------------------------------------------------
        # 将两种模态的特征向量拼接后送入分类头
        # -----------------------------------------------------------
        feat = torch.cat([feat_f, feat_o], dim=1)
        labels = self.classification_head(feat)     
        
        return labels

# geomm_net基础上添加CFP分割分支
class geomm_net_MT(nn.Module):
    def __init__(
            self):
        super(geomm_net_MT, self).__init__()
        
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
            in_channels=1,
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
    
        initialize_head(self.classification_head)
        initialize_decoder(self.decoder)
        initialize_head(self.segmentation_head)
        
        
    # x1: cfp, x2: oct
    def forward(self, x1, x2):
        
        # 提取特征
        features_f = self.encoder_f(x1)
        feat_f = self.avgpool(features_f[-1])
        feat_f = feat_f.view(feat_f.size(0), -1)
        
        decoder_output = self.decoder(*features_f)
        masks = self.segmentation_head(decoder_output[-1])
        
        oct_list = list(x2.split(1, dim=1))  # 将 OCT 切片拆分成列表，len(oct_list) = M
        feature_vectors = []  # 用于存储每个切片的特征向量
        masked_feature_maps = []  # 用于存储每个切片的特征图
        
        # Step 1: 对每个 OCT 切片提取特征并计算全局平均池化向量
        for oct_img in oct_list:
            oct_img = oct_img.squeeze(1)  # 去掉单通道维度，shape = (B, H, W)
            features = self.encoder_o(oct_img)  # 提取特征图，features[-1] shape = (B, C, H', W')
            feat_map = features[-1]  # 最后一层特征图
            
            # 全局平均池化并展平成 (B, C)
            feature_vector = self.avgpool(feat_map).view(feat_map.size(0), -1)  # shape = (B, C)
            feature_vectors.append(feature_vector)  # 存储特征向量
            masked_feature_maps.append(feat_map)  # 存储特征图
            
        # Step 2: 将所有特征向量堆叠成 (B, M, C)，M 是切片数量
        feature_vectors = torch.stack(feature_vectors, dim=1)  # shape = (B, M, C)

        # Step 3: 逐通道找到每个通道上响应最大的切片索引
        channel_indices = torch.argmax(feature_vectors, dim=1)  # shape = (B, C)    

        # Step 4: 根据每个通道的索引，从对应的切片中提取特征图
        B, C, H, W = masked_feature_maps[0].shape  # 获取特征图的形状
        selected_feature_maps = torch.zeros((B, C, H, W), device=feature_vectors.device)  # 初始化选择的特征图
        # 遍历每个通道，选择对应的特征图
        for b in range(B):
            for c in range(C):
                # 获取样本 b、通道 c 的最佳切片索引
                best_slice_idx = channel_indices[b, c]
                # 从该切片中选择特征图对应的通道
                selected_feature_maps[b, c] = masked_feature_maps[best_slice_idx][b, c]

        # 最终的 selected_feature_maps shape = (B, C, H, W)
        combined_feature_map = selected_feature_maps  # 直接使用选中的特征图    
        
        # 对聚合后的 OCT 特征图进行全局平均池化，得到最终的特征向量
        feat_o = self.avgpool(combined_feature_map)
        feat_o = feat_o.view(feat_o.size(0), -1)
        
        # -----------------------------------------------------------
        # 将两种模态的特征向量拼接后送入分类头
        # -----------------------------------------------------------
        feat = torch.cat([feat_f, feat_o], dim=1)
        labels = self.classification_head(feat)     
        
        return masks,labels

# best 0.9208
class geomm_net_tmi2024(nn.Module):
    def __init__(
            self):
        super(geomm_net_tmi2024, self).__init__()
        
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
            in_channels=1,
            depth=5,
            weights="imagenet",
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        # 1D 卷积计算 OCT 切片权重
        self.conv1d = nn.Conv1d(in_channels=1024, out_channels=64, kernel_size=64,padding=0)
        self.sigmoid = nn.Sigmoid()
        
        # CFP 几何注意力 1D 卷积
        self.conv1d_fundus = nn.Conv1d(in_channels=64, out_channels=13, kernel_size=5, stride=4, padding=2)
        
        self.classification_head = ClassificationHead_common(in_channels=512+1024,classes=3, dropout=0.5)
        initialize_head(self.classification_head)

    # x1: cfp, x2: oct
    def forward(self, x1, x2):
        
        B, M, _,H, W = x2.shape  # 获取 OCT 形状 # torch.Size([2, 64, 1, 224, 224])
        
        # 提取特征
        # -------------------------
        # Step 1: CFP 特征提取
        # -------------------------
        features_f = self.encoder_f(x1)  # CFP 提取特征
        feat_f = self.avgpool(features_f[-1]).view(B, -1)  # CFP 特征向量 (B, C)
        
        # -------------------------
        # Step 2: OCT 特征提取 & 权重计算
        # -------------------------
        feature_vectors = []
        masked_feature_maps = []
        
        feature_vectors_max = []
        masked_feature_maps_max = []
        for i in range(M):
            oct_img = x2[:, i, :, :, :]  # 取出第 i 张 OCT 切片, shape = (B, H, W)
            features = self.encoder_o(oct_img)
            feat_map = features[-1]  
            
            feature_vector = self.avgpool(feat_map).view(B, -1)  # shape = (B, C)
            feature_vectors.append(feature_vector)
            masked_feature_maps.append(feat_map)
            
            feature_vector_max = self.maxpool(feat_map).view(B, -1)  
            feature_vectors_max.append(feature_vector_max)
            masked_feature_maps_max.append(feat_map)
        
        # 将所有切片的特征向量堆叠 (B, M, C)
        feature_vectors = torch.stack(feature_vectors, dim=1) # torch.Size([B, 64（M）, 512(C)]
        feature_vectors_max = torch.stack(feature_vectors_max, dim=1) # torch.Size([B, 64（M）, 512(C)])
        
        # 计算 1D 卷积权重 (B, M, C) -> (B, M, 1)
        slice_weights = self.sigmoid((self.conv1d(torch.cat([feature_vectors, feature_vectors_max], dim=2).permute(0, 2, 1))))  # 正确的 permute  (B, 1, M) torch.Size([B, 1, 64])
        # -------------------------
        # Step 3: OCT 切片选择 对的
        # -------------------------
        channel_indices = torch.argmax(feature_vectors, dim=1)  # shape = (B, M，C)
        channel_indices_max = torch.argmax(feature_vectors_max, dim=1)  # shape = (B, M，C)
        
        # 根据通道选择最佳切片
        B, C, H_out, W_out = masked_feature_maps[0].shape  # 获取特征图的形状
        selected_feature_maps = torch.zeros((B, C, H_out, W_out), device=feature_vectors.device)  # 初始化选择的特征图
        selected_feature_maps_max = torch.zeros((B, C, H_out, W_out), device=feature_vectors.device)  # 初始化选择的特征图
        
        # 遍历 batch 和通道，选择最佳的切片
        for b in range(B):
            for c in range(C):
                best_slice_idx = channel_indices[b, c]  # 获取最佳切片索引
                best_slice_idx_max = channel_indices_max[b, c]  
                
                selected_feature_maps[b, c] = masked_feature_maps[best_slice_idx][b, c]  # 选取最佳切片的通道
                selected_feature_maps_max[b, c] = masked_feature_maps_max[best_slice_idx_max][b, c]
        
        
        # 对选择的 OCT 特征图进行池化 (B, C, H, W) -> (B, C)
        feat_o_avg = self.avgpool(selected_feature_maps).view(B, -1)
        feat_o_max = self.avgpool(selected_feature_maps_max).view(B, -1)
        feat_o = torch.cat([feat_o_avg, feat_o_max], dim=1)
        
        # -------------------------
        # Step 4: CFP 特征注意力加权
        # -------------------------
        # 1D 卷积处理权重
        
        # print(slice_weights.shape) torch.Size([2, 64, 1])
        r = self.conv1d_fundus(slice_weights)  # shape: (B, h, 1)  slice_weights: torch.Size([B, 64, 1])
        # print(r.shape) torch.Size([2, 13, 1])
        r = self.sigmoid(r)  # 归一化 torch.Size([2, 13, 1])
        cfp_feature_map = features_f[-1]  # (B, C, H, W)
        
        r = r.unsqueeze(1)
        # print( r.shape) torch.Size([2, 1, 13, 1]
        r = r.expand(-1, C, -1, -1)
        # print( r.shape) torch.Size([2, 512, 13, 1])
        attended_cfp_feature_map = cfp_feature_map + r * cfp_feature_map  # (B, C, H, W)

        # -------------------------
        # Step 5: 分类任务
        # -------------------------
        feat_f = self.avgpool(attended_cfp_feature_map).view(B, -1)  # 计算加权 CFP 特征
        feat = torch.cat([feat_f, feat_o], dim=1)  # 拼接 CFP 和 OCT 特征
        labels = self.classification_head(feat)  # 分类
        
        return labels

# class CDRExtractor(nn.Module):
#     def __init__(self):
#         super().__init__()
#         # 杯盘比计算模块
#         self.cdr_pool = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),
#             nn.Flatten()
#         )
        
#     def forward(self, segmentation_mask):
#         """
#         输入：分割mask (B, 3, H, W) 
#         (背景=0，视盘=1，视杯=2)
#         输出：形态学特征向量
#         """
#         # 计算杯盘比
#         # torch.Size([2, 3, 448, 448])
#         # 像素的类别是0,1,2
#         pred_mask = torch.argmax(segmentation_mask, dim=1)  # (B, H, W)
#         # torch.Size([2, 448, 448])
        
#         cup_area = (pred_mask == 1).float().sum(dim=(1,2)).unsqueeze(1)  # (B,1)
#         disc_area = (pred_mask).float().sum(dim=(1,2)).unsqueeze(1)  # (B,1)
#         cdr = cup_area / (disc_area + 1e-6)  # (B,1)

#         # 概率图
#         prob_map = torch.softmax(segmentation_mask, dim=1)  # (B,3,H,W)
#         # disc_prob = prob_map[:,2]  # 视盘概率图 (B,H,W)
#         # cup_prob = prob_map[:,1]  # 视杯概率图 (B,H,W)

#         # 提取形状特征
#         mask_feat = self.cdr_pool(prob_map)  # (B,3)
#         return torch.cat([cdr, mask_feat], dim=1)  # (B,4)

class CDRExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        # 杯盘比计算模块
        self.cdr_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )
        
    def forward(self, segmentation_mask):
        """
        输入：分割mask (B, 3, H, W) 
        (背景=0，视盘=1，视杯=2)
        输出：形态学特征向量
        """
        # 计算杯盘比
        # torch.Size([2, 3, 448, 448])
        # 像素的类别是0,1,2
        pred_mask = torch.argmax(segmentation_mask, dim=1)  # (B, H, W)
        # torch.Size([2, 448, 448])

        def get_vertical_bounds(mask, label):
            """
            获取 batch 内指定类别的垂直边界 (ymin, ymax)
            """
            B = mask.shape[0]  # 获取 batch 大小
            bounds = torch.zeros(B, 2, device=mask.device)  # 初始化为 0

            for i in range(B):
                coords = torch.nonzero(mask[i] == label, as_tuple=False)  # 获取当前样本的非零坐标
                if coords.shape[0] == 0:  # 如果没有找到该类别的像素点
                    continue  # 直接跳过

                ymin = coords[:, 0].min()
                ymax = coords[:, 0].max()
                bounds[i] = torch.tensor([ymin, ymax], device=mask.device)

            return bounds  # (B, 2) [ymin, ymax]


        # 获取视杯和视盘的上下边界
        cup_bounds = get_vertical_bounds(pred_mask, 1)  # (B, 2) [top, bottom]
        disc_bounds = get_vertical_bounds(pred_mask, 2)  # (B, 2) [top, bottom]

        # 计算垂直高度
        cup_height = (cup_bounds[:, 1] - cup_bounds[:, 0]).unsqueeze(1)  # (B,1)
        disc_height = (disc_bounds[:, 1] - disc_bounds[:, 0]).unsqueeze(1)  # (B,1)

        cdr = cup_height / (disc_height + 1e-6)  # (B,1)

        # 概率图
        prob_map = torch.softmax(segmentation_mask, dim=1)  # (B,3,H,W)
        cup_prob = prob_map[:,1]  # 视杯概率图 (B,H,W)
        disc_prob = prob_map[:,2]  # 视盘概率图 (B,H,W)

        # 提取形状特征
        # 计算形态特征
        shape_features = torch.cat([
            self.cdr_pool(disc_prob.unsqueeze(1)),  # 视盘形状特征 (B,1)
            self.cdr_pool(cup_prob.unsqueeze(1)),   # 视杯形状特征 (B,1)
            disc_prob.mean(dim=(1,2)).unsqueeze(1),     # 平均概率 (B,1)
            cup_prob.mean(dim=(1,2)).unsqueeze(1)        # 平均概率 (B,1)
        ], dim=1)  # (B,4)
        
        return torch.cat([cdr, shape_features], dim=1)  # (B,5)

class AnatomyAwareHead(nn.Module):
    def __init__(self, in_channels, anatomy_dim, classes, dropout=0.5):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_channels+ anatomy_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, classes))
        
    def forward(self, x, anatomy_feat):
        return self.fc(torch.cat([x, anatomy_feat], dim=1))

class geomm_net_tmi2024_MT(nn.Module):
    def __init__(
            self):
        super(geomm_net_tmi2024_MT, self).__init__()
        
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
            in_channels=1,
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
        
        self.cdr_extractor = CDRExtractor()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        # 1D 卷积计算 OCT 切片权重
        self.conv1d = nn.Conv1d(in_channels=1024, out_channels=64, kernel_size=64,padding=0)
        # self.conv1d = nn.Conv1d(in_channels=4096, out_channels=64, kernel_size=64,padding=0)
        self.sigmoid = nn.Sigmoid()
        
        # CFP 几何注意力 1D 卷积
        self.conv1d_fundus = nn.Conv1d(in_channels=64, out_channels=14, kernel_size=5, stride=4, padding=2)
        # self.classification_head = ClassificationHead_common(in_channels=512+1024,classes=3, dropout=0.5)
        self.classification_head = AnatomyAwareHead(
            in_channels=512+1024,
            anatomy_dim=5,  # CDR特征维度
            classes=3,
            dropout=0.5
        )
        # self.classification_head = AnatomyAwareHead(
        #     in_channels=2048+4096,
        #     anatomy_dim=5,  # CDR特征维度
        #     classes=3,
        #     dropout=0.5
        # )
        initialize_head(self.classification_head)
        initialize_decoder(self.decoder)
        initialize_head(self.segmentation_head)

    # x1: cfp, x2: oct
    def forward(self, x1, x2):
        
        B, M, _,H, W = x2.shape  # 获取 OCT 形状 # torch.Size([2, 64, 1, 224, 224])
        
        # 提取特征
        # -------------------------
        # Step 1: CFP 特征提取
        # -------------------------
        features_f = self.encoder_f(x1)  # CFP 提取特征
        feat_f = self.avgpool(features_f[-1]).view(B, -1)  # CFP 特征向量 (B, C)
        
        decoder_output = self.decoder(*features_f)
        masks = self.segmentation_head(decoder_output[-1])
        
        with torch.no_grad():  # 防止分割梯度影响主任务
            anatomy_feat = self.cdr_extractor(masks.detach())     # 2.4
        # -------------------------
        # Step 2: OCT 特征提取 & 权重计算
        # -------------------------
        feature_vectors = []
        masked_feature_maps = []
        
        feature_vectors_max = []
        masked_feature_maps_max = []
        for i in range(M):
            oct_img = x2[:, i, :, :, :]  # 取出第 i 张 OCT 切片, shape = (B, H, W)
            features = self.encoder_o(oct_img)
            feat_map = features[-1]  
            
            feature_vector = self.avgpool(feat_map).view(B, -1)  # shape = (B, C)
            feature_vectors.append(feature_vector)
            masked_feature_maps.append(feat_map)
            
            feature_vector_max = self.maxpool(feat_map).view(B, -1)  
            feature_vectors_max.append(feature_vector_max)
            masked_feature_maps_max.append(feat_map)
        
        # 将所有切片的特征向量堆叠 (B, M, C)
        feature_vectors = torch.stack(feature_vectors, dim=1) # torch.Size([B, 64（M）, 512(C)]
        feature_vectors_max = torch.stack(feature_vectors_max, dim=1) # torch.Size([B, 64（M）, 512(C)])
        
        # 计算 1D 卷积权重 (B, M, C) -> (B, M, 1)
        slice_weights = self.sigmoid((self.conv1d(torch.cat([feature_vectors, feature_vectors_max], dim=2).permute(0, 2, 1))))  # 正确的 permute  (B, 1, M) torch.Size([B, 1, 64])
        # -------------------------
        # Step 3: OCT 切片选择 对的
        # -------------------------
        channel_indices = torch.argmax(feature_vectors, dim=1)  # shape = (B, M，C)
        channel_indices_max = torch.argmax(feature_vectors_max, dim=1)  # shape = (B, M，C)
        
        # 根据通道选择最佳切片
        B, C, H_out, W_out = masked_feature_maps[0].shape  # 获取特征图的形状
        selected_feature_maps = torch.zeros((B, C, H_out, W_out), device=feature_vectors.device)  # 初始化选择的特征图
        selected_feature_maps_max = torch.zeros((B, C, H_out, W_out), device=feature_vectors.device)  # 初始化选择的特征图
        
        # 遍历 batch 和通道，选择最佳的切片
        for b in range(B):
            for c in range(C):
                best_slice_idx = channel_indices[b, c]  # 获取最佳切片索引
                best_slice_idx_max = channel_indices_max[b, c]  
                
                selected_feature_maps[b, c] = masked_feature_maps[best_slice_idx][b, c]  # 选取最佳切片的通道
                selected_feature_maps_max[b, c] = masked_feature_maps_max[best_slice_idx_max][b, c]
        
        # 对选择的 OCT 特征图进行池化 (B, C, H, W) -> (B, C)
        feat_o_avg = self.avgpool(selected_feature_maps).view(B, -1)
        feat_o_max = self.avgpool(selected_feature_maps_max).view(B, -1)
        feat_o = torch.cat([feat_o_avg, feat_o_max], dim=1)
        
        # -------------------------
        # Step 4: CFP 特征注意力加权
        # -------------------------
        # 1D 卷积处理权重
        r = self.conv1d_fundus(slice_weights)  # shape: (B, h, 1)  slice_weights: torch.Size([B, 64, 1])
        r = self.sigmoid(r)  # 归一化 torch.Size([2, 13, 1])
        cfp_feature_map = features_f[-1]  # (B, C, H, W)
        
        r = r.unsqueeze(1)
        r = r.expand(-1, C, -1, -1)
        attended_cfp_feature_map = cfp_feature_map + r * cfp_feature_map  # (B, C, H, W)

        # -------------------------
        # Step 5: 分类任务
        # -------------------------
        feat_f = self.avgpool(attended_cfp_feature_map).view(B, -1)  # 计算加权 CFP 特征
        feat = torch.cat([feat_f, feat_o], dim=1)  # 拼接 CFP 和 OCT 特征
        
        labels =  self.classification_head(feat, anatomy_feat)
        
        return masks, labels

# 修改点：slice_weight通过自注意力学习权重；切片选取采用加权平均方式
class geomm_net_tmi2024_MT2(nn.Module):
    def __init__(
            self):
        super(geomm_net_tmi2024_MT2, self).__init__()
        
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
            in_channels=1,
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
        
        self.attention = nn.MultiheadAttention(embed_dim=1024, num_heads=8)  # 这里假设512是特征维度
        
        self.cdr_extractor = CDRExtractor()
        
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        # 1D 卷积计算 OCT 切片权重
        self.conv1d = nn.Conv1d(in_channels=1024, out_channels=64, kernel_size=64,padding=0)
        self.sigmoid = nn.Sigmoid()
        
        # CFP 几何注意力 1D 卷积
        self.conv1d_fundus = nn.Conv1d(in_channels=64, out_channels=14, kernel_size=5, stride=4, padding=2)
        self.classification_head = AnatomyAwareHead(
            # in_channels=512+1024,
            in_channels=512+512,
            anatomy_dim=5,  # CDR特征维度
            classes=3,
            dropout=0.5
        )

        initialize_head(self.classification_head)
        initialize_decoder(self.decoder)
        initialize_head(self.segmentation_head)

    # x1: cfp, x2: oct
    def forward(self, x1, x2):
        
        B, M, _, H, W = x2.shape  # 获取 OCT 形状 # torch.Size([2, 64, 1, 224, 224])
        # 提取特征
        features_f = self.encoder_f(x1)  # CFP 提取特征
        feat_f = self.avgpool(features_f[-1]).view(B, -1)  # CFP 特征向量 (B, C)
        decoder_output = self.decoder(*features_f)
        masks = self.segmentation_head(decoder_output[-1])     
        with torch.no_grad():  # 防止分割梯度影响主任务
            anatomy_feat = self.cdr_extractor(masks.detach())     # 2.4
            

        oct_input = x2.view(B*M, 1, H, W)  # (B*M, 1, H, W)
        features = self.encoder_o(oct_input)  # 输出列表，最后一层为 (B*M, C, H', W')
        feat_map_all = features[-1].view(B, M, -1, H//32, W//32)  # (B, M, C, H', W')       
        
        feature_vectors = []
        masked_feature_maps = []
        feature_vectors_max = []
        masked_feature_maps_max = []
        
        for i in range(M):
            # oct_img = x2[:, i, :, :, :]  # 取出第 i 张 OCT 切片, shape = (B, H, W)
            # features = self.encoder_o(oct_img)
            # feat_map = features[-1]  
            feat_map = feat_map_all[:, i, :, :, :]
            
            feature_vector = self.avgpool(feat_map).view(B, -1)  # shape = (B, C)
            feature_vectors.append(feature_vector)
            masked_feature_maps.append(feat_map)
            
            feature_vector_max = self.maxpool(feat_map).view(B, -1)  
            feature_vectors_max.append(feature_vector_max)
            masked_feature_maps_max.append(feat_map)
        
        # 将所有切片的特征向量堆叠 (B, M, C)
        feature_vectors = torch.stack(feature_vectors, dim=1) # torch.Size([B, 64（M）, 512(C)]
        feature_vectors_max = torch.stack(feature_vectors_max, dim=1) # torch.Size([B, 64（M）, 512(C)])
        
        # 为每个OCT切片分配权重
        attention_input = torch.cat([feature_vectors, feature_vectors_max], dim=2).permute(1, 0, 2)  # (M, B, 2*C)  (sequence_length, batch_size, embed_dim)
        attn_output, _ = self.attention(attention_input, attention_input, attention_input)  # (M, B, 2*C)
        attn_output = attn_output.permute(1, 0, 2).permute(0, 2, 1)  # (B, 2*C, M)
        
        # 计算 1D 卷积权重 (B, M, C) -> (B, M, 1)
        slice_weights = self.conv1d(attn_output)  # 使用1D卷积得到切片权重 (B, M, 1)
        slice_weights = self.sigmoid(slice_weights)  # 权重归一化 
        
        # # 根据通道选择最佳切片
        B, C, H_out, W_out = masked_feature_maps[0].shape  # 获取特征图的形状
        slice_weights_o = slice_weights.unsqueeze(2).unsqueeze(3)
        weighted_feature_maps = torch.sum(slice_weights_o * torch.stack(masked_feature_maps, dim=1), dim=1)
        
        # 对选择的 OCT 特征图进行池化 (B, C, H, W) -> (B, C)
        feat_o_avg = self.avgpool(weighted_feature_maps).view(B, -1)
        feat_o = feat_o_avg
        # feat_o_max = self.maxpool(weighted_feature_maps).view(B, -1)
        # feat_o = torch.cat([feat_o_avg, feat_o_max], dim=1)
        
        
        # 1D 卷积处理权重
        r = self.conv1d_fundus(slice_weights)  # shape: (B, h, 1)  slice_weights: torch.Size([B, 64, 1])
        r = self.sigmoid(r)  # 归一化 torch.Size([2, 13, 1])
        cfp_feature_map = features_f[-1]  # (B, C, H, W)
        
        r = r.unsqueeze(1)
        r = r.expand(-1, C, -1, -1)
        attended_cfp_feature_map = cfp_feature_map + r * cfp_feature_map  # (B, C, H, W)

        feat_f = self.avgpool(attended_cfp_feature_map).view(B, -1)  # 计算加权 CFP 特征
        feat = torch.cat([feat_f, feat_o], dim=1)  # 拼接 CFP 和 OCT 特征
        
        labels =  self.classification_head(feat, anatomy_feat)
        
        return masks, labels


import torch
import torch.nn as nn

# class XS_MultiNet(nn.Module):
#     def __init__(self):
#         super(XS_MultiNet, self).__init__()
        
#         # 图像特征编码器（保持原始实现）
#         self.encoder_f = get_encoder(
#             "mit_b2",
#             in_channels=3,
#             depth=5,
#             weights="imagenet",
#         )
        
#         # 特征池化层
#         self.avgpool = nn.AdaptiveAvgPool2d(1)
#         self.maxpool = nn.AdaptiveMaxPool2d(1)
        
#         # 双分类头（保持原始结构）
#         self.classification_head_1 = nn.Sequential(
#             nn.Linear(512, 512),
#             nn.BatchNorm1d(512),
#             nn.ReLU(),
#             nn.Dropout(0.5),
#             nn.Linear(512, 3))
        
#         self.classification_head_2 = nn.Sequential(
#             nn.Linear(512, 512),
#             nn.BatchNorm1d(512),
#             nn.ReLU(),
#             nn.Dropout(0.5),
#             nn.Linear(512, 2))
        
#         # 初始化分类头（假设这是自定义函数）
#         initialize_head(self.classification_head_1)
#         initialize_head(self.classification_head_2)

#     def forward(self, x1):
#         """处理形状为 (batch_size, 9, 3, 44, 224) 的输入"""
#         # 维度处理
#         B, M, C, H, W = x1.shape  # batch_size, 9, 3, 44, 224
        
#         # 特征提取（保持原始逻辑）
#         x1_reshaped = x1.view(B*M, C, H, W)  # (B*9, 3, 44, 224)
#         features_f = self.encoder_f(x1_reshaped)  # 获取各层特征
        
#         # 获取最后一层特征
#         last_feature = features_f[-1]  # (B*9, 512, h, w)
        
#         # 特征池化
#         pooled_feature = self.avgpool(last_feature).view(B*M, -1)  # (B*9, 512)
        
#         # 恢复原始batch维度
#         pooled_feature = pooled_feature.view(B, M, -1)  # (B, 9, 512)
        
#         # 特征聚合（使用简单平均）
#         aggregated_feature = torch.mean(pooled_feature, dim=1)  # (B, 512)
        
#         # 双分类头输出
#         output1 = self.classification_head_1(aggregated_feature)  # (B, 3)
#         output2 = self.classification_head_2(aggregated_feature)  # (B, 2)
        
#         return output1, output2
    
    
class CrossAttentionLayer(nn.Module):
    """交叉注意力层（支持残差连接和层归一化）"""
    def __init__(self, embed_dim=512, num_heads=8, dropout=0.1):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # 输入形状: (batch_size, num_images, feature_dim)
        residual = x
        
        # 调整维度适应PyTorch多头注意力（需要seq_len在前）
        x = x.permute(1, 0, 2)  # (num_images, batch_size, feature_dim)
        attn_output, _ = self.multihead_attn(x, x, x)
        attn_output = attn_output.permute(1, 0, 2)  # 恢复维度 (batch_size, num_images, feature_dim)
        
        # 残差连接 + 层归一化
        output = self.layer_norm(residual + self.dropout(attn_output))
        return output

class XS_MultiNet(nn.Module):
    def __init__(self):
        super(XS_MultiNet, self).__init__()
        
        # 图像特征编码器（保持原始实现）
        self.encoder_f = get_encoder(
            "mit_b1",
            in_channels=3,
            depth=5,
            weights="imagenet",
        )
        
        # 特征池化层
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        
        # 交叉注意力模块（添加3层增强交互能力）
        self.cross_attention = nn.Sequential(
            CrossAttentionLayer(embed_dim=512),
            CrossAttentionLayer(embed_dim=512),
            CrossAttentionLayer(embed_dim=512)
        )
        
        # 双分类头（保持原始结构）
        self.classification_head_1 = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 3))
        
        self.classification_head_2 = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 2))
        
        # 初始化分类头
        initialize_head(self.classification_head_1)
        initialize_head(self.classification_head_2)

    def forward(self, x1):
        """处理形状为 (batch_size, 9, 3, 44, 224) 的输入"""
        B, M, C, H, W = x1.shape  # batch_size, 9, 3, 44, 224
        
        # 特征提取
        x1_reshaped = x1.view(B*M, C, H, W)
        features_f = self.encoder_f(x1_reshaped)
        
        # 获取并池化最后一层特征
        last_feature = features_f[-1]  # (B*9, 512, h, w)
        pooled_feature = self.avgpool(last_feature).view(B*M, -1)  # (B*9, 512)
        pooled_feature = pooled_feature.view(B, M, -1)  # (B, 9, 512)
        
        # 交叉注意力处理（9张图像特征交互）
        cross_features = self.cross_attention(pooled_feature)  # (B, 9, 512)
        
        # 自适应特征聚合（带可学习权重的注意力聚合）
        aggregation_weights = torch.softmax(cross_features.mean(dim=-1), dim=-1)  # (B, 9)
        aggregated_feature = torch.sum(cross_features * aggregation_weights.unsqueeze(-1), dim=1)  # (B, 512)
        
        # 双分类头输出
        output1 = self.classification_head_1(aggregated_feature)
        output2 = self.classification_head_2(aggregated_feature)
        
        return output1, output2
