import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from model.Mossformer.mossformer import MossFormerBlock

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

        self.average_pool = nn.AdaptiveAvgPool1d(1)
        self.cosin_threshold = 0.3
        self.adpter = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256))
        
        self.person_embedding = nn.Embedding(20, 256)

    def extract_radar_feature(self,
                              x:torch.tensor) -> torch.tensor:

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

    def mask_embedding(self,
                       time_feature:torch.tensor,
                       embedding:torch.tensor) -> torch.tensor:
        """
        Applies a mask to the time feature based on cosine similarity with the embedding.

        Args:
            time_feature (torch.tensor): The time feature tensor with shape of [B, T, C].
            embedding (torch.tensor): The embedding tensor with shape of [1, C].

        Returns:
            torch.tensor: The masked time feature tensor with shape of [B, T, C].
        """
        embedding = embedding.unsqueeze(1)
        cosin_similarity = F.cosine_similarity(time_feature, embedding, dim=-1)
        mask = cosin_similarity > self.cosin_threshold
        mask = mask.unsqueeze(-1)
        time_feature = time_feature * mask + embedding * (~mask)
        return time_feature

    def init_embedding(self,
                                embedding: torch.tensor,
                                label: torch.tensor) -> None:
        """
        Initializes the embedding of new person for a few-shot learning scenario.

        Args:
            x (torch.tensor): The input tensor.
            label (torch.tensor): The label tensor.

        Returns:
            None
        """
        self.person_embedding.weight.data[label] = embedding
    
    def inference(self,
                  x:torch.tensor,
                  label: torch.tensor) -> torch.tensor:

        person_feature, time_feature = self.extract_radar_feature(x)
        # weight = self.person_embedding.weight
        # cosin_similarity = F.cosine_similarity(person_feature, weight, dim=-1)
        # label = cosin_similarity.argmax(dim=-1)
        # person_embedding = weight[label]
        # person_embedding = person_embedding.unsqueeze(0)
        # person_feature, time_feature = self.extract_radar_feature(x)
        person_embedding = self.person_embedding(label)
        time_feature = self.mask_embedding(time_feature, person_embedding)
        return time_feature, person_embedding

    def forward(self,
                x:torch.tensor,
                label:torch.tensor) -> torch.tensor:

        person_feature, time_feature = self.extract_radar_feature(x)
        person_embedding = self.person_embedding(label)
        time_feature = self.mask_embedding(time_feature, person_embedding)
        mask = F.one_hot(label, num_classes=20)
        mask = mask.unsqueeze(-1)
        other_embedding = self.person_embedding.weight * (1 - mask)
        other_cosin_similarity = torch.mean(F.cosine_similarity(person_feature, other_embedding, dim=-1))
        embendding_loss = - F.cosine_similarity(person_feature, person_embedding) + \
                            other_cosin_similarity
        return time_feature, person_embedding, embendding_loss