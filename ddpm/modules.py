import torch as th
import torch.nn as nn
import torch.functional as F
import math
from abc import ABC, abstractmethod


class DownSample(nn.Module):

    def __init__(self,
                 input_channels,
                 output_channels,
                 kernel_size=2,
                 stride=2):
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.conv = nn.Conv1d(input_channels, output_channels, kernel_size=kernel_size, stride=stride)
    
    def forward(self, x):
        return self.conv(x)
    

class Upsample(nn.Module):
    
    def __init__(self,
                 input_channels,
                 output_channels,
                 kernel_size=3,
                 padding=1):
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv = nn.Conv1d(input_channels, output_channels, kernel_size=kernel_size, padding=padding)
    
    def forward(self, x):
        return self.conv(self.upsample(x))
    

def timestep_embedding(timesteps, emb_dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.
    """
    half = emb_dim // 2
    freqs = th.exp(-math.log(max_period) * th.arange(start=0, end=half, dtype=th.float32) / half).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = th.cat([th.cos(args), th.sin(args)], dim=-1)
    if emb_dim % 2:
        embedding = th.cat([embedding, th.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class TimestepBlock(nn.Module, ABC):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """
    @abstractmethod
    def forward(self, x, embeddings):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimeSequentialEmbedding(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that support it as an extra input.
    """
    def forward(self, x, emb):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x
    

def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


class Residual_block(TimestepBlock):
    """
    Residual block with optional timestep embedding conditioning.

    This block applies a sequence of normalization, activation, and convolution layers to the input tensor, adds a timestep embedding (after linear projection and SiLU activation), and includes a skip/residual connection to facilitate gradient flow. 
    It is designed for use in UNet-like architectures in diffusion models.
    """
    def __init__(self,
                 input_channels,
                 output_channels,
                 emb_channels,):
        super().__init__()
        self.input_channels = input_channels
        self.emb_channels = emb_channels
        self.output_channels = output_channels
        self.in_layers = nn.Sequential(
            nn.GroupNorm(32, input_channels),
            nn.SiLU(),
            nn.Conv1d(in_channels=input_channels, out_channels=output_channels, kernel_size=3, padding=1)
        )
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_channels, output_channels)
        )
        self.out_layers = nn.Sequential(
            nn.GroupNorm(32, output_channels),
            nn.SiLU(),
            zero_module(nn.Conv1d(in_channels=output_channels, out_channels=output_channels, kernel_size=3, padding=1))
        )
        if input_channels == output_channels:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = nn.Conv1d(in_channels=input_channels, out_channels=output_channels, kernel_size=1)
        
        
    def forward(self, x, embeddings):
        """
        Inputs:
            x[th.tensor]: Input feature map tensor of shape [B, C, ...]
            embeddings[th.tensor]: Timestep embeddings of shape [B, emb_channels] for each batch element.
        Output:
            output[th.tensor]: Tensor of shape [B, C, ...] after residual addition.
        """
        residual = self.skip_connection(x)
        x = self.in_layers(x)
        embeddings = self.emb_layers(embeddings)
        while len(embeddings.shape) < len(x.shape):
            embeddings = embeddings[...,None]
        output = x + embeddings
        output = self.out_layers(output)
        return output + residual
    

class SelfAttention(nn.Module):
    """
    Multi-head self-attention mechanism used in AttentionBlock.

    This module splits the input tensor into queries, keys, and values, applies multi-head attention, and returns the aggregated attention output.
    It is designed to operate on tensors of shape [B, T, 3*C] where queries, keys, and values are concatenated along the channel dimension.
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Input:
            qkv[th.tensor]: Input tensor of shape [B, 3*C, T]
        Output:
            attentions [th.tensor]: Output tensor of shape [B, C, T], representing the attention-weighted sum of values across all heads.
        """
        n_samples, channels, timesteps = qkv.shape
        assert channels % (3 * self.n_heads) == 0
        q, k, v = qkv.chunk(3, dim=1)
        c_heads = channels // (3 * self.n_heads)
        new_shape = n_samples, self.n_heads, c_heads, timesteps
        q, k, v = q.view(new_shape).transpose(2,3), k.view(new_shape).transpose(2,3), v.view(new_shape).transpose(2,3)
        weights = th.einsum('bhqd,bhkd->bhqk', q, k)
        weights /= math.sqrt(c_heads)
        weights = th.softmax(weights, dim=-1)
        attentions = th.einsum('bhqk,bhkd->bhqd', weights, v)
        attentions = attentions.transpose(2,3).reshape(n_samples, -1, timesteps)
        return attentions


class AttentionBlock(nn.Module):
    """
    Self-attention block for feature maps.

    This block applies group normalization followed by a linear projection to produce query, key, and value tensors, then performs multi-head self-attention, and finally projects the result back to the original channel dimension. 
    A residual connection adds the input to the attention output. Designed for use in UNet-like architectures.
    """

    def __init__(self,
                 channels, 
                 n_heads):
        super().__init__()
        self.channels = channels
        self.n_heads = n_heads
        assert channels % n_heads == 0
        self.c_heads = channels // n_heads
        self.in_layers = nn.Sequential(
            nn.GroupNorm(num_groups=32, num_channels=channels),
            nn.Conv1d(channels, 3*channels, kernel_size=1)
        )
        self.attention_layer = SelfAttention(n_heads)
        self.out_layers = nn.Conv1d(channels, channels, kernel_size=1)

    def forward(self, x):
        """
        Input:
            x[th.tensor]: Input feature map tensor of shape [B, C, ...]
        Output:
            output[th.tensor]: tensor of shape [B, C, ...]  after applying self-attention and residual addition.
        """
        b, c, *size = x.shape
        x = x.reshape(b, c, -1) # x: [B, C, T]
        qkv = self.in_layers(x) # qkv: [B, 3C, T]
        a = self.attention_layer(qkv) # a: [B, C, T]
        a = self.out_layers(a) # a: [B, C, T]
        return (a + x).reshape(b, c, *size)