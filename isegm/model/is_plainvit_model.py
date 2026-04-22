import math
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.nn import functional as F
from isegm.utils.serialization import serialize
from .is_model import ISModel
from .modeling.models_vit import VisionTransformer, PatchEmbed
from .modeling.swin_transformer import SwinTransfomerSegHead


class SimpleFPN(nn.Module):
    def __init__(self, in_dim=768, out_dims=[128, 256, 512, 1024]):
        super().__init__()
        self.down_4_chan = max(out_dims[0]*2, in_dim // 2)
        self.down_4 = nn.Sequential(
            nn.ConvTranspose2d(in_dim, self.down_4_chan, 2, stride=2),
            nn.GroupNorm(1, self.down_4_chan),
            nn.GELU(),
            nn.ConvTranspose2d(self.down_4_chan, self.down_4_chan // 2, 2, stride=2),
            nn.GroupNorm(1, self.down_4_chan // 2),
            nn.Conv2d(self.down_4_chan // 2, out_dims[0], 1),
            nn.GroupNorm(1, out_dims[0]),
            nn.GELU()
        )
        self.down_8_chan = max(out_dims[1], in_dim // 2)
        self.down_8 = nn.Sequential(
            nn.ConvTranspose2d(in_dim, self.down_8_chan, 2, stride=2),
            nn.GroupNorm(1, self.down_8_chan),
            nn.Conv2d(self.down_8_chan, out_dims[1], 1),
            nn.GroupNorm(1, out_dims[1]),
            nn.GELU()
        )
        self.down_16 = nn.Sequential(
            nn.Conv2d(in_dim, out_dims[2], 1),
            nn.GroupNorm(1, out_dims[2]),
            nn.GELU()
        )
        self.down_32_chan = max(out_dims[3], in_dim * 2)
        self.down_32 = nn.Sequential(
            nn.Conv2d(in_dim, self.down_32_chan, 2, stride=2),
            nn.GroupNorm(1, self.down_32_chan),
            nn.Conv2d(self.down_32_chan, out_dims[3], 1),
            nn.GroupNorm(1, out_dims[3]),
            nn.GELU()
        )

        self.init_weights()

    def init_weights(self):
        pass

    def forward(self, x):
        x_down_4 = self.down_4(x)
        x_down_8 = self.down_8(x)
        x_down_16 = self.down_16(x)
        x_down_32 = self.down_32(x)

        return [x_down_4, x_down_8, x_down_16, x_down_32]



class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch, channels, _, _ = x.size()
        scale = self.global_avg_pool(x).view(batch, channels)
        scale = self.fc(scale).view(batch, channels, 1, 1)
        return x * scale

    
class PlainVitModel(ISModel):
    @serialize
    def __init__(
        self,
        backbone_params={},
        neck_params={}, 
        seg_head_params={},
        edge_head_params={},
        #fusion_params={},
        random_split=False,
        **kwargs
        ):

        super().__init__(**kwargs)
        self.random_split = random_split
        
        self.patch_embed_coords = PatchEmbed(
            img_size= backbone_params['img_size'],
            patch_size=backbone_params['patch_size'], 
            #in_chans=3 if self.with_prev_mask else 2, 
            in_chans = (
                4 if self.with_prev_mask and self.with_prev_edges
                #else 5 if self.use_disk_maps and self.use_connectedness
                else 3 if self.with_prev_mask or self.with_prev_edges 
                else 2
            ),
            embed_dim=backbone_params['embed_dim'],
        )

        self.backbone = VisionTransformer(**backbone_params)
        self.neck = SimpleFPN(**neck_params)
        
        self.edge_head = SwinTransfomerSegHead(**edge_head_params)         
        
        self.head = SwinTransfomerSegHead(**seg_head_params)
        
        self.seg = None
        if self.with_aux_output:
            self.seg = self.head.conv_seg

    def backbone_forward(self, image, coord_features=None):
        ## Forward pass disk_maps and prev mask to patch embedding
        coord_features = self.patch_embed_coords(coord_features)
        ## Forward pass disk_maps and prev mask to patch embedding result and image
        backbone_features = self.backbone.forward_backbone(image, coord_features, self.random_split)

        # Extract 4 stage backbone feature map: 1/4, 1/8, 1/16, 1/32
        B, N, C = backbone_features.shape
        grid_size = self.backbone.patch_embed.grid_size

        backbone_features = backbone_features.transpose(-1,-2).view(B, C, grid_size[0], grid_size[1])
        
        #edge_multi_scale_features = self.resnet(torch.concat([image,image_grad],dim=1))
        
        seg_multi_scale_features = self.neck(backbone_features)
        
        seg, seg_feat = self.head(seg_multi_scale_features)
        edge, edge_feat = self.edge_head(seg_multi_scale_features)
        
        seg2 = self.seg(seg_feat + edge_feat) if self.seg is not None else None  # Safe calculation
        
        return {'instances':seg,
                'edges': edge, 
                'instances_aux': seg2 if seg2 is not None else None
                }#None}
