import torch
import torch.nn as nn
import torch.nn.functional as F

from ..Mossformer.mossformer import (
    GLU,
    FusionGLULiner,
    MossFormerBlock,
    ScaledSinuEmbedding,
)
from .cross_flash import CorssFLASHTransformer
from .radar_net import RadarNet


class RadarMossFormer(nn.Module):
    def __init__(self,
                 in_dim:int=1,
                 hidden_dim:int=256,
                 kernel_size:int=8,
                 stride:int=4,
                 MFB_num:int=1,
                 drop_out_rate:float=0.1,
                 ) -> None:
        """
        MossFormer model implementation.

        Args:
            in_dim (int): Number of input dimensions.
            hidden_dim (int): Dimension of hidden layers.
            kernel_size (int): Size of the convolutional kernel.
            stride (int): Stride value for the convolutional layers.
            speaker_num (int): Number of speakers.
            MFB_num (int): Number of MossFormer blocks.
            drop_out_rate (float): Dropout rate.
        """
        super().__init__()
        self.MFB_num = MFB_num
        self.MFB_num1 = MFB_num
        self.MFB_num2 = MFB_num
        self.in_conv = nn.Sequential(
            nn.Conv1d(in_dim,
                      hidden_dim,
                      kernel_size=kernel_size,
                      stride=stride),
            nn.ReLU()
        )
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.abs_pos_emb = ScaledSinuEmbedding(hidden_dim)
        self.in_point_wise_conv = nn.Sequential(nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1),
                                                nn.ReLU())
        self.mossforer_block1 = nn.Sequential()
        self.mossforer_block2 = nn.Sequential()
        for i in range(self.MFB_num):
            self.mossforer_block1.add_module(f"1_MFB_{i}", MossFormerBlock(dim=hidden_dim,
                                                                           group_size=256,
                                                                           query_key_dim=128,
                                                                           expansion_factor=2.,
                                                                           dropout=drop_out_rate))
            self.mossforer_block2.add_module(f"2_MFB_{i}", MossFormerBlock(dim=2*hidden_dim,
                                                                            group_size=256,
                                                                            query_key_dim=128,
                                                                            expansion_factor=2.,
                                                                            dropout=drop_out_rate))
        self.radar_net = RadarNet(audio_channels=hidden_dim)

        # fusion person_embedding and audio
        self.select_glu = FusionGLULiner(hidden_dim)
        
        self.glu = GLU(2*hidden_dim)
        self.out_point_wise_conv = nn.Sequential(
            nn.Conv1d(hidden_dim*2, hidden_dim, kernel_size=1),
            nn.ReLU()
        )
        self.out_conv = nn.ConvTranspose1d(hidden_dim,
                                           in_dim,
                                           kernel_size=kernel_size,
                                           stride=stride)

    def forward(self,
                x:torch.Tensor,
                radar:torch.Tensor,
                label:torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the MossFormer model.

        Args:
            x (torch.Tensor): Input tensor with shape [B, 1, T].

        Returns:
            torch.Tensor: Output tensor with shape [BxC, 1, T].
        """
        # extract audio feature
        in_len = x.shape[-1]
        x_in = self.in_conv(x)

        x_trans = x_in.transpose(-1, -2)
        x_norm = self.ln1(x_trans)
        abs_pos_emb = self.abs_pos_emb(x_norm)
        x_pos = abs_pos_emb + x_norm
        x_pos = x_pos.transpose(-1, -2)
        
        x_MFB_in = self.in_point_wise_conv(x_pos)
        x_MFB_in = x_MFB_in.transpose(-1, -2)
        x_MFB1_out = self.mossforer_block1(x_MFB_in)

        person_logit, time_feature, person_feature,  = self.radar_net(radar,
                                                                      x_MFB1_out.transpose(-1, -2),
                                                                      label)

        x_extract = self.select_glu(x_MFB1_out, person_feature)
    
        time_feature = F.interpolate(time_feature, x_extract.shape[-2], mode='nearest')
        # end extract radar feature
        time_feature = time_feature.transpose(-1, -2)
        x_split = torch.cat([x_extract, time_feature], dim=-1)

        x_split = self.mossforer_block2(x_split)
        x_split = x_split.transpose(-1, -2)
        # end fusion audio and radar

        # rebuild audio
        x_split = self.glu(x_split)
        mask = self.out_point_wise_conv(x_split)
        split_sound =  self.out_conv(mask * x_in)[...,:in_len]
        return split_sound, person_logit