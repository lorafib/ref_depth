# from https://github.com/NVlabs/nvdiffrec/blob/main/dataset/dataset.py

# Copyright (c) 2020-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved. 
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction, 
# disclosure or distribution of this material and related documentation 
# without an express license agreement from NVIDIA CORPORATION or 
# its affiliates is strictly prohibited.

import torch

class Dataset(torch.utils.data.Dataset):
    """Basic dataset interface"""
    def __init__(self): 
        super().__init__()

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self):
        raise NotImplementedError

    def collate(self, batch):
        iter_res = batch[0]['resolution']
        coll_batch = {k: torch.cat(list([item[k] for item in batch]), dim=0) for k in batch[0] if not k in ['resolution']}
        coll_batch["resolution"] = iter_res
        return coll_batch