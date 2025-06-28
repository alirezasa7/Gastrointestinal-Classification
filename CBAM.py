import torch
import torch.nn as nn
import torch.nn.functional as F

class EnhancedSpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(EnhancedSpatialAttention, self).__init__()
        padding = kernel_size // 2

        self.spatial_attention_1 = nn.Sequential(
            nn.Conv2d(2 , 1 , kernel_size = kernel_size, padding = padding, bias = False ),
            #nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.spatial_attention_2 = nn.Sequential(
            nn.Conv2d(2 , 1 , kernel_size = kernel_size // 2, padding = padding // 2, bias = False ),
            #nn.BatchNorm2d(1),
            nn.Sigmoid()
    
        )
        

    def forward(self, x):
        
        avg_map = torch.mean(x, dim=1, keepdim=True)       
        max_map, _ = torch.max(x, dim=1, keepdim=True)     

        # Concatenate maps and apply attention
        spatial_feat = torch.cat([avg_map, max_map], dim=1)  
        attn_1 = self.spatial_attention_1(spatial_feat)          
        attn_2 = self.spatial_attention_2(spatial_feat)          

        final = attn_1 + attn_2
        return x * final
