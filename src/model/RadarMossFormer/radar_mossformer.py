import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from ..Mossformer.mossformer import GLU, FusionGLU, MossFormerBlock, ScaledSinuEmbedding
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
            in_dim (int): Number of input dimensions. Default is 1.
            hidden_dim (int): Dimension of hidden layers. Default is 512.
            kernel_size (int): Size of the convolutional kernel. Default is 8.
            stride (int): Stride value for the convolutional layers. Default is 4.
            speaker_num (int): Number of speakers. Default is 4.
            MFB_num (int): Number of MossFormer blocks. Default is 4.
            drop_out_rate (float): Dropout rate. Default is 0.1.
        """
        super().__init__()
        self.MFB_num = MFB_num
        self.MFB_num1 = 2
        self.MFB_num2 = 4
        self.in_conv = nn.Sequential(
            nn.Conv1d(in_dim,
                      hidden_dim,
                      kernel_size=kernel_size,
                      stride=stride),
            nn.ReLU()
        )
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.abs_pos_emb = ScaledSinuEmbedding(hidden_dim)

        self.in_point_wise_conv = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1)
        for i in range(self.MFB_num1):
            setattr(self, f"1_MFB_{i}", MossFormerBlock(dim=hidden_dim,
                                                      group_size=256,
                                                      query_key_dim=128,
                                                      expansion_factor=2.,
                                                      dropout=drop_out_rate))
        for i in range(self.MFB_num2):
            setattr(self, f"2_MFB_{i}", MossFormerBlock(dim=hidden_dim,
                                                      group_size=256,
                                                      query_key_dim=128,
                                                      expansion_factor=2.,
                                                      dropout=drop_out_rate))
        self.radar_net = RadarNet()

        self.select_glu = FusionGLU(hidden_dim)
        self.cross_flash = CorssFLASHTransformer(dim=hidden_dim,
                                                 depth=2)
        self.glu = GLU(hidden_dim)
        self.out_point_wise_conv = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1),
            nn.ReLU()
        )
        self.out_conv = nn.ConvTranspose1d(hidden_dim,
                                           in_dim,
                                           kernel_size=kernel_size,
                                           stride=stride)

    def forward(self, x:torch.Tensor, radar:torch.Tensor, label) -> torch.Tensor:
        """
        Forward pass of the MossFormer model.

        Args:
            x (torch.Tensor): Input tensor with shape [B, 1, T].

        Returns:
            torch.Tensor: Output tensor with shape [BxC, 1, T].
        """
        in_len = x.shape[-1]
        x_in = self.in_conv(x)
        
        x_trans = x_in.transpose(-1, -2)
        x_norm = self.ln1(x_trans)
        abs_pos_emb = self.abs_pos_emb(x_norm)
        x_pos = abs_pos_emb + x_norm
        x_pos = x_pos.transpose(-1, -2)
        
        x_MFB_in = self.in_point_wise_conv(x_pos)
        x_MFB_in = x_MFB_in.transpose(-1, -2)
        for i in range(self.MFB_num1):
            x_MFB_in = getattr(self, f"1_MFB_{i}")(x_MFB_in)
        x_MFB_out = x_MFB_in.transpose(-1, -2)
        x_MFB_out = F.relu(x_MFB_out)
        # radar net
        time_feature, person_feature, embedding_loss = self.radar_net(radar, label)
        person_feature = person_feature.transpose(-1, -2)
        x_extract = self.select_glu(x_MFB_out, person_feature)
        time_feature = time_feature.transpose(-1, -2)
        time_feature = F.interpolate(time_feature, x_extract.shape[-1], mode='nearest')
        x_split = x_extract.transpose(-1, -2)
        time_feature = time_feature.transpose(-1, -2)
        x_split = self.cross_flash(x_split, time_feature)
        for i in range(self.MFB_num2):
            x_split = getattr(self, f"2_MFB_{i}")(x_split)
        x_split = x_split.transpose(-1, -2)

        x_split = self.glu(x_split)
        mask = self.out_point_wise_conv(x_split)
        split_sound =  self.out_conv(mask * x_in)[...,:in_len]
        return split_sound, embedding_loss