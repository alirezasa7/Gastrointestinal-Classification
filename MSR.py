
import torch
import torch.nn as nn
from backbone_utils import CustomConvMixerWithTransformer
from transformer import TransformerEncoderLayer, TransformerDecoderLayer, PositionEmbeddingSine
from CBAM import EnhancedSpatialAttention
from CROSS_ATTENTION import MultiHeadCrossAttention


class MSR(nn.Module):
    def __init__(self, num_classes=6, reduce_dim=64, transformer_depth=1):
        super(MSR, self).__init__()

        self.reduce_dim = reduce_dim
        self.embedding_dim = reduce_dim // 2

        self.backbone = CustomConvMixerWithTransformer(num_classes=num_classes)
        self.cbam = EnhancedSpatialAttention(kernel_size = 7)
        self.activations = {}

        self.dim_reduction = nn.Sequential(
            nn.Conv2d(self.backbone.feature_dim, reduce_dim * 4, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(reduce_dim * 4, reduce_dim, kernel_size=1)
        )

        self.pe_layer = PositionEmbeddingSine(reduce_dim // 2, normalize=True)

        self.cross_attn = MultiHeadCrossAttention(dim=reduce_dim, num_heads=4, use_mul=False)

        self.feed_forward = nn.Sequential(
            nn.Linear(reduce_dim, reduce_dim),
            nn.GELU()
        )

        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(
                d_model=reduce_dim,
                nhead=4,
                dim_feedforward=reduce_dim * 4,
                dropout=0.1
            ) for _ in range(transformer_depth)
        ])

        self.classifier = nn.Sequential(
            nn.LayerNorm(reduce_dim),
            nn.Linear(reduce_dim, self.embedding_dim),
            nn.BatchNorm1d(self.embedding_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.embedding_dim, num_classes)
        )

        self.merge_conv = nn.Sequential(
            nn.Conv2d(reduce_dim * 2, reduce_dim, kernel_size=1),
            #nn.BatchNorm2d(reduce_dim),
            nn.GELU()
        )

        
        
        self.fpn_fuse = nn.Sequential(
            nn.Conv2d(reduce_dim*2, reduce_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(reduce_dim),
            nn.GELU()
        )

    def forward(self, x):
        B = x.size(0)

        x_low, x_medium, x_high = self.backbone(x)
    
        x_low = self.dim_reduction(x_low)
        x_medium = self.dim_reduction(x_medium)
        x_high = self.dim_reduction(x_high)

        self._hook_outputs = {
        'x_low': x_low,
        'x_medium': x_medium,
        'x_high': x_high, 
        }

        self.activations['x_low'] = x_low
        self.activations['x_medium'] = x_medium
        self.activations['x_high'] = x_high

        x_low_medium = torch.cat([x_low , x_medium], dim=1)
        x_low_medium = self.fpn_fuse(x_low_medium)

        pos = self.pe_layer(x_low_medium)
        x_low_medium = x_low_medium + pos
        x_high = x_high + pos

        x_attn = self.cross_attn(x_high, x_low_medium)
        x_feat_map = x_attn + x_high
        self.activations['cross_attn'] = x_feat_map

        x_cbam = self.cbam(x_feat_map)
        self.activations['cbam'] = x_cbam

        x_concat = torch.cat([x_feat_map, x_cbam], dim=1)
        
        x_feat_map = self.merge_conv(x_concat)

        query_input = nn.functional.adaptive_avg_pool2d(x_feat_map, 1).squeeze(-1).squeeze(-1)  
        query_input = query_input.unsqueeze(0)  
        memory = x_feat_map.flatten(2).permute(2, 0, 1)  

        query_attended = query_input
        for dec in self.decoder_layers:
            query_attended = dec(query_attended, memory=memory)

        query_residual = query_attended.squeeze(0) + query_input.squeeze(0)
        out_t = self.feed_forward(query_residual)

        self.activations['transformer_decoder'] = out_t.unsqueeze(-1).unsqueeze(-1)  

        skip = nn.functional.adaptive_avg_pool2d(x_high, 1).squeeze(-1).squeeze(-1)
        query_fused = out_t + skip
        out = self.classifier(query_fused)
        return out

    def get_fc_params(self):
        return list(self.dim_reduction.parameters()) + \
               list(self.cross_attn.parameters()) + \
               list(self.feed_forward.parameters()) + \
               list(self.decoder_layers.parameters()) + \
               list(self.classifier.parameters()) + \
               list(self.cbam.parameters()) + \
               list(self.merge_conv.parameters()) + \
               list(self.fpn_fuse.parameters())
