## Denoising Convolutional Auto-Encoder
import torch.nn as nn
from torch.nn import functional as F


class SimpleDCAE(nn.Module):
    def __init__(self, s = 1, d = 1):
         super(SimpleDCAE, self).__init__()
         ## encoder: 1 -> 128 -> 64 -> 32
         self.encoder = nn.ModuleDict({
             'enc_block1': self.create_encoder_block(1, 128, stride = s, dilation =d, padding = 0 ),
             'enc_block2': self.create_encoder_block(128, 64, stride = s, dilation =d, padding = 0),
              'enc_block3': self.create_encoder_block(64, 32, stride = s, dilation =d, padding = 0) 
            })
         
         ## decoder: 32 -> 64 -> 128 -> 1
         self.decoder =  nn.ModuleDict(
             {
              'dec_block1': self.create_decoder_block(32, 64, stride = s, dilation = d, padding = 0),
              'dec_block2' : self.create_decoder_block(64, 128, stride = s, dilation = d, padding = 0),
              'dec_block3': nn.ConvTranspose2d(in_channels=128, out_channels=1,
                                               dilation = d,
                                               kernel_size=3,stride = s, padding = 0)
             }
              )
    def create_encoder_block(self, in_channels, out_channels, **kwargs):
        block = nn.ModuleList([nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, **kwargs),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU()
                ])
        return nn.Sequential(*block)
    
    def create_decoder_block(self, in_channels, out_channels, **kwargs):
        block = nn.ModuleList([
            nn.ConvTranspose2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=3, **kwargs),
                nn.LeakyReLU()
                ])
        return nn.Sequential(*block)
    
    def forward(self, x):
        ## encoder layer
        for e in self.encoder:
            x = self.encoder[e](x)
        
        ## decoder layer
        for d in self.decoder:
            x = self.decoder[d](x)
            
        return x


