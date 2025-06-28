
import timm
import torch
import torch.nn as nn


class FrozenBatchNorm2d(nn.Module):
    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def forward(self, x):
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        eps = 1e-4
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


def convert_batchnorm_to_frozen(module):
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            frozen_bn = FrozenBatchNorm2d(child.num_features)
            frozen_bn.weight.data.copy_(child.weight.data)
            frozen_bn.bias.data.copy_(child.bias.data)
            frozen_bn.running_mean.data.copy_(child.running_mean.data)
            frozen_bn.running_var.data.copy_(child.running_var.data)
            setattr(module, name, frozen_bn)
        else:
            convert_batchnorm_to_frozen(child)


class CustomConvMixerWithTransformer(nn.Module):
    def __init__(self, model_name='convmixer_768_32', num_classes=6):
        super().__init__()

        
        self.base_model = timm.create_model(model_name, pretrained=True, num_classes=0)
        convert_batchnorm_to_frozen(self.base_model)
        self.stem = self.base_model.stem
        self.blocks = self.base_model.blocks

        for param in self.stem.parameters():
            param.requires_grad = False
        for param in self.blocks.parameters():
            param.requires_grad = False

    
        self.feature_dim = self.base_model.num_features  

    def forward(self, x):
        x = self.stem(x)  # [B, C, H, W]

        x1 = x2 = x3 = None

        for idx, block in enumerate(self.blocks):
            x = block(x)
            if idx == 10:
                x1 = x  
            if idx == 20:
                x2 = x  

        x3 = x  

        return x1, x2, x3
