import math
import torch as th
import torch.nn as nn
from .modules import *


class UNet(nn.Module):

    def __init__(self,
                 input_channels,
                 model_channels,
                 output_channels,
                 n_resblocks,
                 n_heads,
                 attention_resolution,
                 channels_mult):
        super().__init__()
        self.input_channels = input_channels
        self.model_channels = model_channels
        self.output_channels = output_channels
        self.n_resblocks = n_resblocks
        self.n_heads = n_heads
        self.attention_resolution = attention_resolution
        self.channels_mult = channels_mult
        time_channels = 4 * model_channels

        self.encoder = nn.ModuleList(
            [TimeSequentialEmbedding(nn.Conv1d(input_channels, model_channels, kernel_size=3, padding=1))]
        )
        channels = int(channels_mult[0] * model_channels)
        encoder_channels = [model_channels]
        for (i, mult) in enumerate(channels_mult):
            for j in range(n_resblocks):
                layers = [Residual_block(input_channels=channels, output_channels=int(mult * model_channels), emb_channels=time_channels)]
                channels = int(mult * model_channels)
                encoder_channels.append(channels)
                if mult in attention_resolution:
                    layers.append(AttentionBlock(channels, n_heads))
                self.encoder.append(TimeSequentialEmbedding(*layers))
            if i != len(channels_mult)-1 :
                self.encoder.append(TimeSequentialEmbedding(DownSample(channels, channels_mult[i+1]*model_channels)))
                channels = channels_mult[i+1]*model_channels
                encoder_channels.append(channels)
                

        self.bottleneck = TimeSequentialEmbedding(
            Residual_block(input_channels=channels, output_channels=channels, emb_channels=time_channels),
            AttentionBlock(channels, n_heads),
            Residual_block(input_channels=channels, output_channels=channels, emb_channels=time_channels)
        )

        self.decoder = nn.ModuleList([])
        for (i, mult) in enumerate(channels_mult[::-1]):
            for j in range(n_resblocks+1):
                concat_channels = channels + encoder_channels.pop()
                layers = [Residual_block(input_channels=concat_channels, output_channels=int(mult * model_channels), emb_channels=time_channels)]
                channels = int(mult * model_channels)
                if mult in attention_resolution:
                    layers.append(AttentionBlock(channels, n_heads))
                if i != len(channels_mult) - 1 and j == n_resblocks:
                    layers.append(Upsample(channels, channels_mult[::-1][i+1]*model_channels))
                    channels = channels_mult[::-1][i+1]*model_channels
                self.decoder.append(TimeSequentialEmbedding(*layers))

        self.output_layers = nn.Sequential(
            nn.GroupNorm(32, channels),
            nn.SiLU(),
            zero_module(nn.Conv1d(channels, output_channels, kernel_size=3, padding=1))
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(model_channels, model_channels*4),
            nn.SiLU(),
            nn.Linear(model_channels*4, model_channels*4),
        )          
        
    def forward(self, x, diffusion_steps):
        embeddings = self.time_embedding(timestep_embedding(diffusion_steps, self.model_channels))
        skip_connections = []
        for layer in self.encoder:
            x = layer(x, embeddings)
            skip_connections.append(x)
        x = self.bottleneck(x, embeddings)
        for layer in self.decoder:
            skip = skip_connections.pop()
            x = th.cat([x, skip], dim=1)
            x = layer(x, embeddings)
        output = self.output_layers(x)
        return output