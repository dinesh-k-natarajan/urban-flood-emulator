import torch
from torch import nn

class ConcatenationFusion(nn.Module):
    def __init__(self):
        super(ConcatenationFusion, self).__init__()
        ## Concatenation of the two images (n_samples, n_image_channels, height, width) along the image channel dimension
        self.concatenator = lambda enc1, enc2: torch.cat((enc1, enc2), dim=-3) 
    
    def forward(self, enc_dem, enc_rain):
        ## Convert enc_rain into an image with similar height and width as enc_dem, but with only 1 image channel
        enc_dem_ones = torch.ones_like(enc_dem)[...,:1,:,:]
        ## Using torch.einsum on enc_dem_ones of shape (n_samples, 1, height, width) with enc_rain of shape (n_samples, n_rain_enc)
        ## to produce an output of shape (n_samples, n_rain_enc, height, width)
        enc_rain_image = torch.einsum("ijkl, im -> imkl", enc_dem_ones, enc_rain)
        # Concatenate
        fused = self.concatenator(enc_dem, enc_rain_image)
        
        return fused
    
    def get_fused_channels(self, dem_channels, rain_channels):
        # Returns the expected channels in fused of shape [N, C, H, W]
        return dem_channels + rain_channels
        