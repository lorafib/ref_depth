# Copyright 2024, Laura Fink

import numpy as np
import torch

#----------------------------------------------------------------------------

# from https://pytorch.org/vision/main/_modules/torchvision/ops/misc.html#MLP
import warnings
from typing import Callable, List, Optional, Sequence, Tuple, Union
class MLP(torch.nn.Sequential):
    """This block implements the multi-layer perceptron (MLP) module.

    Args:
        in_channels (int): Number of channels of the input
        hidden_channels (List[int]): List of the hidden channel dimensions
        norm_layer (Callable[..., torch.nn.Module], optional): Norm layer that will be stacked on top of the linear layer. If ``None`` this layer won't be used. Default: ``None``
        activation_layer (Callable[..., torch.nn.Module], optional): Activation function which will be stacked on top of the normalization layer (if not None), otherwise on top of the linear layer. If ``None`` this layer won't be used. Default: ``torch.nn.ReLU``
        inplace (bool, optional): Parameter for the activation layer, which can optionally do the operation in-place.
            Default is ``None``, which uses the respective default values of the ``activation_layer`` and Dropout layer.
        bias (bool): Whether to use bias in the linear layer. Default ``True``
        dropout (float): The probability for the dropout layer. Default: 0.0
    """

    def init_weights(self, m):
        if isinstance(m, torch.nn.Linear):
            # Initialize all weights to have minimal impact by setting them close to zero
            torch.nn.init.normal_(m.weight, mean=0.0, std=self.init_std)
            if m.bias is not None:
                torch.nn.init.normal_(m.bias, mean=0.0, std=self.init_std)
            
    def __init__(
        self,
        in_channels: int,
        hidden_channels: List[int],
        norm_layer: Optional[Callable[..., torch.nn.Module]] = None,
        activation_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.ReLU,
        inplace: Optional[bool] = None,
        bias: bool = True,
        dropout: float = 0.0,
        init_std=0.025
    ):
        # The addition of `norm_layer` is inspired from the implementation of TorchMultimodal:
        # https://github.com/facebookresearch/multimodal/blob/5dec8a/torchmultimodal/modules/layers/mlp.py
        params = {} if inplace is None else {"inplace": inplace}

        layers = []
        in_dim = in_channels
        for hidden_dim in hidden_channels[:-1]:
            layers.append(torch.nn.Linear(in_dim, hidden_dim, bias=bias))
            if norm_layer is not None:
                layers.append(norm_layer(hidden_dim))
            layers.append(activation_layer(**params))
            # layers.append(torch.nn.Dropout(dropout, **params))
            in_dim = hidden_dim

        layers.append(torch.nn.Linear(in_dim, hidden_channels[-1], bias=bias))
        # # layers.append(torch.nn.Dropout(dropout, **params))

        super().__init__(*layers)
        
        # initialization
        self.init_std = init_std
        self.apply(self.init_weights)


#----------------------------------------------------------------------------


class DepthTransferFunc(MLP):
    def __init__(self, args, offset_scale):
        activation = None
        if args.transfer_func_acti == "relu": activation = torch.nn.ReLU
        elif args.transfer_func_acti == "tanh": activation = torch.nn.Tanh
        elif args.transfer_func_acti == "sigmoid": activation = torch.nn.Sigmoid
        else: 
            print("invalid acitvation")
            exit()
        
        self.offset_scale = offset_scale
        self.num_posencs_uv_z = args.transfer_func_posencs_uv_z
        self.in_channels = 2*self.num_posencs_uv_z[0]+2 + self.num_posencs_uv_z[1]+1
        self.normalize_z = args.transfer_func_normalize_z
        
        super().__init__(in_channels=self.in_channels, hidden_channels=[*args.transfer_func_hlayers,2],
                         activation_layer=activation, init_std=args.transfer_func_init_std)
        
        # test forward pass
        test = -27*torch.ones([5,self.in_channels])
        test[0,2] = -5.354
        test[4,2] = 6.354
        test[1,2] = -9.354
        print("Test MLP forward pass")
        super().forward(test)
        print("Done")
        
        
    def pos_encoding(self, uv, z):
        in_z = z.clone()
        if self.normalize_z:
            in_z = (in_z - self.offset_scale[0]) /self.offset_scale[1] 
            # print("in_z", torch.std_mean(in_z))
        uv_encs     = [torch.sin(np.pi * pow(2,i) * uv)  for i in range(1, self.num_posencs_uv_z[0]+1)]
        z_encs      = [torch.sin(np.pi * pow(2,i) * z)   for i in range(1, self.num_posencs_uv_z[1]+1)]
        input = torch.cat([uv, *uv_encs, z, *z_encs], dim=1) 
        # input = torch.cat([uv, torch.sin(7*uv), z, torch.sin(3*z), torch.sin(7*z), torch.sin(29*z), torch.sin(100*z)], dim=1)
        return input
              
    
    def forward(self, uv, z):
        abs_z = torch.abs(z)
        input = self.pos_encoding(uv, abs_z)
        o_s = super().forward(input)
        scale = torch.clamp_min(torch.abs(1+o_s[:,0,None]), 1e-8)
        positive_transferred_depth = abs_z*scale+o_s[:,1,None] 
        
        # positive_transferred_depth = abs_z*(1+o_s[:,0,None])+(1.0/(1.0-o_s[:,1,None])-1.0) 
        return positive_transferred_depth
    
    