'''
Author: ll
LastEditTime: 2024-06-06 00:48:58
LastEditors: ll
无问西东
'''

from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, trunc_normal_
from timm.models.vision_transformer import Mlp
 
from torchvision.transforms.functional import resize
from torchvision import transforms as T

from pyiqa.archs.arch_util import  random_crop
from pyiqa.archs.clipiqa_arch import  CLIPIQA as clipiqa


class CLIPIQA(nn.Module):
    def __init__(self):
        super().__init__() 
        self.net = clipiqa( 
                 model_type='clipiqa+',
                 backbone='RN50',
                 pretrained=False,
                 pos_embedding=False,

        )   
 
    def forward(self, batch):

        x, t,index,gt = batch['images'],batch['moss'] ,batch['indexs']  ,batch['mossgt']

        x=T.Resize(512)(x)   
        x=T.CenterCrop(512)(x)   

        scores=self.net(x).squeeze(1)  
        lossfun=partial(F.mse_loss,scores) 
        loss=lossfun(target=t)
        logdict = {'mse_score':[scores,gt] ,  
                   'rmse_score':[scores,gt],  
                   'mean_loss':[torch.zeros_like(loss),loss.detach()],
                   'srcc':[scores,gt],
                   'plcc':[scores,gt],
                   } 

        out={'scores':scores,'loss':loss,'logdict':logdict,'lossfun':lossfun}
        return out