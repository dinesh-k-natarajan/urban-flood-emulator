from typing import List, Tuple, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from .concatenation import ConcatenationFusion

DEVICE = torch.device('cuda:0')

## Define typical double convolution block
class conv_block(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(conv_block,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.LeakyReLU(negative_slope=0.02),
            nn.Conv2d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.LeakyReLU(negative_slope=0.02),
        )

    def forward(self,x):
        x = self.conv(x)
        return x

class DecoderLayer_bottleneck(nn.Module):
    def __init__(self, in_channels_concat: int, in_channels_up: int, out_channels: int) -> None:
        super(DecoderLayer_bottleneck, self).__init__()

        self.transpose = nn.ConvTranspose2d(in_channels_up, out_channels, 2, 2, bias=True)
        self.conv = conv_block(in_channels_concat, out_channels)

    def forward(self, skip_x: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        x = self.transpose(x)
        x = torch.cat((skip_x, x), dim=1)
        return self.conv(x)

class DecoderLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super(DecoderLayer, self).__init__()

        self.transpose = nn.ConvTranspose2d(in_channels, out_channels, 2, 2, bias=True)
        self.conv = conv_block(in_channels, out_channels)

    def forward(self, skip_x: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        x = self.transpose(x)
        x = torch.cat((skip_x, x), dim=1)
        return self.conv(x)
    

class UNetTransformer(nn.Module):
    """ 
    Same as UNetTransformer but contains ReLU activation in the output layer.
    """
    def __init__(self, num_heads = 2, n_input_channels:int=1,
                fusion_module: str = 'ConcatenationFusion',
                device=torch.device('cuda:0')
            ) -> None:
        super(UNetTransformer, self).__init__()
        self.device = device
        self.pos_encoding = PositionalEncoding(self.device)
        
        ##Pooling operation
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        ## Encoder 1: DEM operations in linear format
        self.enc1 = conv_block(ch_in=n_input_channels,ch_out=8)
        self.enc2 = conv_block(ch_in=8,ch_out=16)
        self.enc3 = conv_block(ch_in=16,ch_out=32)
        self.enc4 = conv_block(ch_in=32,ch_out=64)
        self.enc5 = conv_block(ch_in=64, ch_out=128)
        self.enc6= conv_block(ch_in=128, ch_out=256)

        ##Bottle neck positional encoding and Multi-headed Self Attention
        #self.bottleneck = conv_block(ch_in=128, ch_out=256)
        self.pos_encoding = PositionalEncoding(self.device)
        self.mhsa = MultiHeadSelfAttention(256, num_heads)

        # Encoder 2: Rain
        self.rain_encoder = nn.Sequential(
           nn.Linear(in_features=36, out_features=24),
           nn.LeakyReLU(negative_slope=0.02),
           nn.Linear(in_features=24, out_features=12),
           nn.LeakyReLU(negative_slope=0.02),
           nn.Linear(in_features=12, out_features=8),
           nn.LeakyReLU(negative_slope=0.02),
        )

        ##Fusion Module: DEM & rain
        self.fusion = ConcatenationFusion()
        fused_channels = self.fusion.get_fused_channels(dem_channels=256, rain_channels=8)

        #Decoder operations
        self.decode6 = DecoderLayer_bottleneck(in_channels_concat=512, in_channels_up=fused_channels, out_channels=256)
        self.decode5 = DecoderLayer(in_channels=256, out_channels=128)
        self.decode4 = DecoderLayer(in_channels=128, out_channels=64)
        self.decode3 = DecoderLayer(in_channels=64, out_channels=32)
        self.decode2 = DecoderLayer(in_channels=32, out_channels=16)
        self.decode1 = DecoderLayer(in_channels=16, out_channels=8)

        self.outputlayer = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=1, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True) # to avoid negative predictions of water depth
        )

    def forward(self, dem: torch.Tensor, rain: torch.Tensor) -> Dict[str, torch.Tensor]:

        enc1 = self.enc1(dem)
        encP1 = self.pool(enc1)

        enc2 = self.enc2(encP1)
        encP2 = self.pool(enc2)

        enc3 = self.enc3(encP2)
        encP3 = self.pool(enc3)

        enc4 = self.enc4(encP3)
        encP4 = self.pool(enc4)

        enc5 = self.enc5(encP4)
        encP5 = self.pool(enc5)
        
        enc6 = self.enc6(encP5)
        encP6= self.pool(enc6)

        bottle_neck = self.pos_encoding(encP6)
        bottle_neck = self.mhsa(bottle_neck)

        # Rain encoding
        enc_rain   = self.rain_encoder(rain)
        #print(f"öööö Encoded features shape: enc_dem = {encP6.shape}, enc_rain = {enc_rain.shape}")
        fused = self.fusion(bottle_neck, enc_rain)
        #print(f"fused shape is {fused.shape}")

        dec5 = self.decode6(enc6, fused)
        #print(f"dec5 shape is {dec5.shape}")
        dec4 = self.decode5(enc5, dec5)
        dec3 = self.decode4(enc4, dec4)
        dec2 = self.decode3(enc3, dec3)
        dec1 = self.decode2(enc2, dec2)
        op = self.decode1(enc1, dec1)

        output = self.outputlayer(op)

        return {
            'water_depth': output, 
        }

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, bias=False) -> None:
        super(MultiHeadSelfAttention, self).__init__()

        self.mha = nn.MultiheadAttention(embed_dim, num_heads, bias=bias, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.size()
        x = x.permute(0, 2, 3, 1).view((b, h * w, c))
        x, _ = self.mha(x, x, x, need_weights=False)
        return x.view((b, h, w, c)).permute(0, 3, 1, 2)

class PositionalEncoding(nn.Module):
    def __init__(self, device=torch.device('cuda:0')) -> None:
        super(PositionalEncoding, self).__init__()
        self.device = device

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.size()
        pos_encoding = self.positional_encoding(h * w, c)
        pos_encoding = pos_encoding.permute(1, 0).unsqueeze(0).repeat(b, 1, 1)
        x_view = x.view((b, c, h * w))
        x = x_view + pos_encoding
        return x.view((b, c, h, w))

    def positional_encoding(self, length: int, depth: int) -> torch.Tensor:
        depth = depth / 2

        positions = torch.arange(length, device=self.device)
        depths = torch.arange(depth, device=self.device) / depth

        angle_rates = 1 / (10000**depths)
        angle_rads = torch.einsum('i,j->ij', positions, angle_rates)

        pos_encoding = torch.cat((torch.sin(angle_rads), torch.cos(angle_rads)), dim=-1)

        return pos_encoding

def main():
    arch = UNetTransformer()
    
if __name__ == '__main__':
    main()