import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from model.Mossformer.mossformer import MossFormerBlock

from ..conformer import ConformerBlock
from ..Mossformer.conv_modules import *


class RadarNet(nn.Module):


    def __init__(self,
                 in_channels:int=1,
                 out_channels:int=256,
                 MFB_num:int=1) -> None:
        super().__init__()
        self.MFB_num = MFB_num
        self.in_conv = nn.Sequential(Conv1dBlock(in_channels=in_channels,
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

        self.ln1 = nn.LayerNorm(out_channels)

        for i in range(MFB_num):
            setattr(self, f"1_MFB_{i}", MossFormerBlock(dim=out_channels,
                                                      group_size=256,
                                                      query_key_dim=128,
                                                      expansion_factor=2.,
                                                      dropout=0.1))
        self.person_embedding = nn.Embedding(20, 256)
        # end for
        self.average_pool = nn.AdaptiveAvgPool1d(1)
        self.cosin_threshold = 0.1
        self.adpter = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256))
    
    def extract_radar_feature(self, x):
        x = rearrange(x, "b l -> b () l")
        x = self.in_conv(x)
        x = x.transpose(-1, -2)
        x = self.ln1(x)
        for i in range(1):
            x = getattr(self, f"1_MFB_{i}")(x)
        person_feature = x.transpose(-1, -2)
        person_feature = self.average_pool(person_feature)
        person_feature = person_feature.squeeze(-1)
        person_feature = self.adpter(person_feature)
        return person_feature, x

    def mask_embedding(self, time_feature, embedding):

        embedding = embedding.unsqueeze(1)
        cosin_similarity = F.cosine_similarity(time_feature, embedding, dim=-1)
        mask = cosin_similarity > self.cosin_threshold
        mask = mask.unsqueeze(-1)
        time_feature = time_feature * mask + embedding * (~mask)
        return time_feature

    def few_shot_init_embedding(self, x, label):

        person_feature, _ = self.extract_radar_feature(x)
        self.person_embedding.weight.data[label] = person_feature

    def forward(self, x, label):

        person_feature, time_feature = self.extract_radar_feature(x)
        gt_person_feature = self.person_embedding(label)
        time_feature = self.mask_embedding(time_feature, person_feature)
        embendding_loss = -F.cosine_similarity(person_feature, gt_person_feature)
        return time_feature, person_feature, embendding_loss