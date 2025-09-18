from typing import Union, Dict, List, Tuple
import torch
from torch import nn
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

## Define decoder layers
class DecoderLayer_bottleneck(nn.Module):
    def __init__(self, in_channels_skip: int, in_channels_up: int, out_channels: int) -> None:
        super(DecoderLayer_bottleneck, self).__init__()

        self.transpose = nn.ConvTranspose2d(in_channels_up, out_channels, 2, 2, bias=True)
        self.conv = conv_block(in_channels_skip, out_channels)

    def forward(self, skip_x: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        #print(f"shape of x is {x.shape}")
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
    
class UNet_Large(nn.Module):
    def __init__(self, n_input_channels:int=1,
                fusion_module: str = 'ConcatenationFusion',
                device=torch.device('cuda:0')
            ) -> None:
        super(UNet_Large, self).__init__()
        
        ##Pooling operation
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        ## Encoder 1: DEM operations in linear format
        self.enc1 = conv_block(ch_in=n_input_channels,ch_out=16)
        self.enc2 = conv_block(ch_in=16,ch_out=32)
        self.enc3 = conv_block(ch_in=32,ch_out=64)
        self.enc4 = conv_block(ch_in=64,ch_out=128)
        self.enc5 = conv_block(ch_in=128, ch_out=256)
        self.enc6= conv_block(ch_in=256, ch_out=512)

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
        fused_channels = self.fusion.get_fused_channels(dem_channels=512, rain_channels=8)

        #Decoder operations
        self.decode6 = DecoderLayer_bottleneck(in_channels_skip=1024, in_channels_up=fused_channels, out_channels=512)
        self.decode5 = DecoderLayer(in_channels=512, out_channels=256)
        self.decode4 = DecoderLayer(in_channels=256, out_channels=128)
        self.decode3 = DecoderLayer(in_channels=128, out_channels=64)
        self.decode2 = DecoderLayer(in_channels=64, out_channels=32)
        self.decode1 = DecoderLayer(in_channels=32, out_channels=16)

        ##self.outputlayer = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=3, stride=1, padding=1, bias=True)

        self.outputlayer = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=1, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True)
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

        # Rain encoding
        enc_rain   = self.rain_encoder(rain)
        #print(f"öööö Encoded features shape: enc_dem = {encP6.shape}, enc_rain = {enc_rain.shape}")
        fused = self.fusion(encP6, enc_rain)
        #print(f"fused shape is {fused.shape}")

        dec5 = self.decode6(enc6,fused)
        #print(f"dec5 shape is {dec5.shape}")
        dec4 = self.decode5(enc5,dec5)
        dec3 = self.decode4(enc4, dec4)
        dec2 = self.decode3(enc3, dec3)
        dec1 = self.decode2(enc2, dec2)
        op = self.decode1(enc1, dec1)

        output = self.outputlayer(op)

        return {
            'water_depth': output, 
        }
    
def main():
    arch = UNet_Large()
    
if __name__ == '__main__':
    main()