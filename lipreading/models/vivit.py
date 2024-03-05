#!/usr/bin/env python3
#  2023, Sogang University;  Jeongkyun Park
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""
    Vision Transformer for visual front-end.
    Refer to "Transformer-Based Video Front-Ends for Audio-Visual Speech Recognition for Single and Multi-Person Video"
"""

import math
import logging
from typing import List, Optional, Union
from itertools import accumulate

import torch
import torch.nn as nn

from lipreading.models.transformer.embedding import PositionalEncoding
from lipreading.models.transformer.transformer_encoder import TransformerEncoder


class ViViT(nn.Module):
    def __init__(
        self, 
        in_channels: int,
        out_channels: int= 512,
        input_size: list= [88, 88],
        patch_size: list= [22, 22, 7],
        patch_stride: list= [22, 22, 1],
        patch_pad: list= [0, 0, 3],
        cls_token: bool= True,
        pos_enc_layer_type: str= "pos",
        positional_dropout_rate: float= 0.1,
        transformer_conf: dict= dict(
            attention_heads= 8,
            linear_units= 2048,
            num_blocks= 6,
            dropout_rate= 0.1,
            attention_dropout_rate= 0.1,
            input_layer= None,
            normalize_before= True,
        ),
        fps: int= 25,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.patch_per_frame = tuple(ins // ps for ins, ps in zip(input_size, self.patch_size[:2]))
        patch_channels = in_channels * math.prod(self.patch_size)

        if cls_token:
            self.cls_token = torch.nn.Parameter(torch.randn(out_channels))
        else:
            self.cls_token = None

        if pos_enc_layer_type == "pos":
            pos_enc_class = PositionalEncoding
        else:
            pos_enc_class = None
        
        if pos_enc_class:
            self.pos_enc = pos_enc_class(out_channels, positional_dropout_rate, max_len=math.prod(self.patch_per_frame)+int(cls_token))
        else:
            self.pos_enc = None

        self.embed = nn.Sequential(
            nn.LayerNorm(patch_channels),
            nn.Linear(patch_channels, out_channels),
            nn.LayerNorm(out_channels)
        )
        
        self.transformer = TransformerEncoder(
            input_size = out_channels,
            output_size = out_channels,
            **transformer_conf,
        )

    def _patchfy(self, x, ilens):
        """This Function Has been Deprecated
           Separate video into tubelets before perturbating the projection layer.
        """
        
        B, T, H, W, C = x.shape
        
        assert H % self.patch_size[0] == 0 and W % self.patch_size[1] == 0, \
            f"The size of the input image `{x.shape}` doesn't match to the patch size, `{self.patch_size}`"

        h, w, l = self.patch_size
        N = math.prod(self.patch_per_frame)

        # Visualize Pathces
        # import matplotlib.pyplot as plt
        
        # plt.imshow(x[0,0,:,:,0].numpy())
        # plt.savefig("debug/org.png")

        # Patchfy each frame
        x = torch.stack([
                torch.stack([
                    x[:, :, hi:hi+h, wi:wi+w, :] 
                        for wi in range(0, W, self.patch_stride[1])], axis=2) 
                            for hi in range(0, H, self.patch_stride[0])], axis=2)
        x = x.reshape(B, T, N, h, w, C) # B, T, N, h, w, C
        
        # f, ax = plt.subplots(4,4, figsize=(16,16))
        # for r in range(4):
        #     for c in range(4):
        #         ax[r,c].imshow(x[0,0,r*4+c,:,:,0].numpy())
        # f.savefig("debug/patches.png")
        
        # Create tubelets
        pad_len = self.patch_size[2] - 1
        iterrange = range(0,T,self.patch_stride[2])
        T_ = (T + pad_len - self.patch_size[2]) // self.patch_stride[2] + 1
        olens = (ilens + pad_len - self.patch_size[2]) // self.patch_stride[2] + 1

        lpad = pad_len//2
        rpad = pad_len - lpad
        x = torch.nn.functional.pad(x, (0,0, 0,0, 0,0, 0,0, lpad, rpad))
        x = torch.stack([
            x[:, i:i+self.patch_size[2]] for i in iterrange
        ], axis=1)              # B, T', t, N_h * N_w, h, w, C
        x = x.transpose(2,3)    # B, T', N_h * N_w, t, h, w, C
        
        # import matplotlib.pyplot as plt
        # for fid in range(x.shape[1]-3, x.shape[1]):
        #     for t in range(x.shape[3]):
        #         f, ax = plt.subplots(4,4, figsize=(16,16))
        #         for r in range(4):
        #             for c in range(4):
        #                 ax[r,c].imshow(x[0,fid,r*4+c,t,:,:,0].numpy())
        #         import os
        #         os.makedirs(f"debug/tube_{fid}", exist_ok=True)
        #         f.savefig(f"debug/tube_{fid}/patches_{t}.png")
        # exit()
        
        # Flatten
        x = x.reshape(B, T_, N, -1)
        
        return x, olens

    def forward(self, x, input_lengths):
        assert x.dim() == 5, f"shape of inputs is {x.shape}"
        
        B, C, T, H, W = x.shape
        x = x.permute(0, 2, 3, 4, 1)
        input_lengths = x.new(input_lengths).to(int)
        
        B, T, H, W, C = x.shape
        
        # # Visualize Pathces
        # import matplotlib.pyplot as plt
        
        # for i, x_ in enumerate(x.detach().numpy()):
        #     plt.imshow(x_[0,:,:,0])
        #     plt.savefig(f"debug/org_{i}.png")
        # import pdb;pdb.set_trace()

        #########################################
        x, input_lengths = self._patchfy(x, input_lengths)
        # Option #1
        x = torch.cat([x[bid, :ilen] for bid, ilen in enumerate(input_lengths)], axis=0)
        x = self.embed(x)
        #########################################
        # x = x.transpose(1,4)    # (B, C, H, W, T)
        # x = self.embed(x)       # (B, D, H, W, T')
        # x = x.transpose(1,4)    # (B, T', Nh, Nw, D)
        # x = x[:, :T]            # (B, T, Nh, Nw, D)
        # x = torch.flatten(x, start_dim=2, end_dim=3)  # (B, T, Nh * Nw, D)
        #########################################
        # Option #2
        # x = torch.flatten(x, end_dim=1)               # (B x T, Nh * Nw, D)
        _, N, D = x.shape
        
        plens = torch.tensor([N for _ in range(x.shape[0])], device=x.device).to(int)
        # Add [CLS] token if needed
        if self.cls_token is not None:
            x = torch.cat([self.cls_token[None,None,:].repeat(x.shape[0],1,1), x], dim=1)
            plens = plens + 1
        
        if self.pos_enc:
            x = self.pos_enc(x)
        
        x, _, _ = self.transformer(x, plens)

        # Take only the first output (from N)
        x = x[:, 0]
        
        # Restore the original batch
        x = torch.stack([torch.nn.functional.pad(x[sid:sid+ilen], (0,0,0,input_lengths.max()-ilen))
                         for sid, ilen in zip([0, *accumulate(input_lengths[:-1])], input_lengths)], axis=0) # B, T, D
        x = x.reshape(B, T, -1)
        
        return x, input_lengths
