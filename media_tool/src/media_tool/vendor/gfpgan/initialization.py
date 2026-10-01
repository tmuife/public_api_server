"""Adapted from BasicSR v1.4.2 arch_util.default_init_weights (Apache-2.0)."""

import torch
from torch import nn


@torch.no_grad()
def default_init_weights(module_list, scale=1, bias_fill=0, **kwargs):
    if not isinstance(module_list, list):
        module_list = [module_list]
    for module in module_list:
        for layer in module.modules():
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(layer.weight, **kwargs)
                layer.weight.data *= scale
                if layer.bias is not None:
                    layer.bias.data.fill_(bias_fill)
            elif isinstance(layer, nn.BatchNorm2d):
                nn.init.constant_(layer.weight, 1)
                if layer.bias is not None:
                    layer.bias.data.fill_(bias_fill)
