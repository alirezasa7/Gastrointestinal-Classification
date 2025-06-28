
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadCrossAttention(nn.Module):
    def __init__(self, dim, num_heads=4, reduction=2, use_mul=True):
        super(MultiHeadCrossAttention, self).__init__()
        assert dim % num_heads == 0, 
        self.use_mul = use_mul

        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.query_conv = nn.Conv2d(dim, dim, kernel_size=1)
        self.key_conv = nn.Conv2d(dim, dim, kernel_size=1)
        self.value_conv = nn.Conv2d(dim, dim, kernel_size=1)

        self.out_proj = nn.Conv2d(dim, dim, kernel_size=1)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x_high, x_low):

        B, C, H, W = x_high.shape

        query = self.query_conv(x_high)  
        key = self.key_conv(x_low)
        value = self.value_conv(x_low)
        query = query.view(B, self.num_heads, self.head_dim, H * W).permute(0, 1, 3, 2)  
        key = key.view(B, self.num_heads, self.head_dim, H * W)                         
        value = value.view(B, self.num_heads, self.head_dim, H * W).permute(0, 1, 3, 2)  

        attn_scores = torch.matmul(query, key) / (self.head_dim ** 0.5)  
        attn = self.softmax(attn_scores)

        out = torch.matmul(attn, value)  

        out = out.permute(0, 1, 3, 2).contiguous() 
        out = out.view(B, C, H, W)  
        out = self.out_proj(out)  

        if self.use_mul:
            out = out * x_high
        else:
            out = out + x_high

        return out
