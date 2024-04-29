import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from model.Mossformer.mossformer import MossFormerBlock
from model.RadarMossFormer.cross_flash import CorssFLASHTransformer

from ..Mossformer.conv_modules import *


class FeatureExtractor(nn.Module):
    def __init__(self,
                 channel:int,
                 depth:int) -> None:
        super().__init__()
        self.feature_exturctor = nn.Sequential()
        for i in range(depth):
            temp_downsample_conv = Conv1dBlock(in_channels=channel,
                                               out_channels=channel,
                                               kernel_size=8,
                                               stride=4,
                                               padding=0)
            temp_res_conv = ResBlock(Conv1dBlock(in_channels=channel,
                                                 out_channels=channel,
                                                 kernel_size=3,
                                                 stride=1,
                                                 padding=1))
            self.feature_exturctor.add_module(f"conv_{i}",
                                              nn.Sequential(temp_downsample_conv,
                                                            temp_res_conv))
    def forward(self, x:torch.tensor) -> torch.tensor:
        return self.feature_exturctor(x)
    
class RadarNet(nn.Module):


    def __init__(self,
                 in_channels:int=1,
                 out_channels:int=128,
                 audio_channels:int=256,
                 MFB_num:int=4) -> None:
        super().__init__()
        self.MFB_num = MFB_num
        self.radar_in_conv = nn.Sequential(Conv1dBlock(in_channels=in_channels,
                                                       out_channels=out_channels,
                                                       kernel_size=3,
                                                       stride=1,
                                                       padding=1),
                                            Conv1dBlock(in_channels=out_channels,
                                                        out_channels=out_channels,
                                                        kernel_size=8,
                                                        stride=4,
                                                        padding=0),
                                    )

        self.feature_extractor = FeatureExtractor(channel=audio_channels,
                                                  depth=3)
        self.fusion_conv = Conv1dBlock(in_channels=out_channels + audio_channels,
                                       out_channels=audio_channels,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)
        self.mossformer_block = nn.Sequential()
        for i in range(MFB_num):
            self.mossformer_block.add_module(f"MFB_{i}", MossFormerBlock(dim=audio_channels,
                                                                            group_size=256,
                                                                            query_key_dim=128,
                                                                            expansion_factor=2.,
                                                                            dropout=0.1))

        self.average_pool = nn.AdaptiveAvgPool1d(1)
        self.adpter = nn.Sequential(
            nn.Linear(256, 20),
            nn.LogSoftmax())
        self.person_embedding = nn.Embedding(20, audio_channels)


    def forward(self,
                radar:torch.tensor,
                audio_feature:torch.tensor,
                label:torch.tensor) -> torch.tensor:

        radar = rearrange(radar, "b l -> b () l")
        radar_feature = self.radar_in_conv(radar)
        audio_feature = F.interpolate(audio_feature, size=radar_feature.size(-1), mode='nearest')
        fusion_feature = torch.cat([radar_feature, audio_feature], dim=1)
        fusion_feature = self.fusion_conv(fusion_feature)
        person_feature = self.feature_extractor(fusion_feature)

        fusion_feature = fusion_feature.transpose(-1, -2)
        fusion_feature = self.mossformer_block(fusion_feature)
        fusion_feature = fusion_feature.transpose(-1, -2)

        # extrac person feature
        person_feature = self.average_pool(person_feature)
        person_feature = person_feature.squeeze(-1)
        person_logit = self.adpter(person_feature)
        
        if not self.training:
            label = torch.argmax(person_logit, dim=-1)
        
        person_embedding = self.person_embedding(label)
        return person_logit, fusion_feature, person_embedding