'''
Author: ll
LastEditTime: 2024-06-05 14:19:55
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

class PatchEmbed(nn.Module):
    """2D Image to Patch Embedding"""

    def __init__(
        self,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        norm_layer=None,
        flatten=True,
    ):
        super().__init__()
        self.flatten = flatten

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )

        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x


class Attention(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale

        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        Attention_block=Attention,
        Mlp_block=Mlp,
        init_values=1e-4,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention_block(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp_block(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class Experts_MOS(nn.Module):
    def __init__(
        self,
        embed_dim=768,
        juery_nums=6,
    ):
        super().__init__()
        self.juery = juery_nums
        bunch_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            dropout=0.0,
            nhead=6,
            activation=F.gelu,
            batch_first=True,
            dim_feedforward=(embed_dim * 4),
            norm_first=True,
        )
        self.bunch_decoder = nn.TransformerDecoder(bunch_layer, num_layers=1)
        self.bunch_embedding = nn.Parameter(torch.randn(1, self.juery, embed_dim))
        self.heads = nn.Sequential( 
            nn.Linear(embed_dim, embed_dim//2 ),

            nn.ReLU(),
            nn.Linear(embed_dim//2, 1 ),
            # nn.Sigmoid()
                                   )
        trunc_normal_(self.bunch_embedding, std=0.02)

    def forward(self, x, ref):
        B, L, D = x.shape
        bunch_embedding = self.bunch_embedding.expand(B, -1, -1)
        ref = ref.view(B, 1, -1)
        ref = ref.expand(B, self.juery, -1)
        output_embedding = bunch_embedding + ref
        x = self.bunch_decoder(output_embedding, x)
        x = self.heads(x)
        x = x.view(B, -1).mean(dim=1)
        return x.view(B, 1)


class Layer_scale_init_Block(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    # with slight modifications
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        Attention_block=Attention,
        Mlp_block=Mlp,
        init_values=1e-4,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention_block(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp_block(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )
        self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
        self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)

    def forward(self, x):
        x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
        x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


class Layer_scale_init_Block_paralx2(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    # with slight modifications
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        Attention_block=Attention,
        Mlp_block=Mlp,
        init_values=1e-4,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.norm11 = norm_layer(dim)
        self.attn = Attention_block(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.attn1 = Attention_block(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.norm21 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp_block(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )
        self.mlp1 = Mlp_block(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )
        self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
        self.gamma_1_1 = nn.Parameter(
            init_values * torch.ones((dim)), requires_grad=True
        )
        self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
        self.gamma_2_1 = nn.Parameter(
            init_values * torch.ones((dim)), requires_grad=True
        )

    def forward(self, x):
        x = (
            x
            + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
            + self.drop_path(self.gamma_1_1 * self.attn1(self.norm11(x)))
        )
        x = (
            x
            + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
            + self.drop_path(self.gamma_2_1 * self.mlp1(self.norm21(x)))
        )
        return x


class Block_paralx2(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    # with slight modifications
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        Attention_block=Attention,
        Mlp_block=Mlp,
        init_values=1e-4,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.norm11 = norm_layer(dim)
        self.attn = Attention_block(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.attn1 = Attention_block(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.norm21 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp_block(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )
        self.mlp1 = Mlp_block(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x):
        x = (
            x
            + self.drop_path(self.attn(self.norm1(x)))
            + self.drop_path(self.attn1(self.norm11(x)))
        )
        x = (
            x
            + self.drop_path(self.mlp(self.norm2(x)))
            + self.drop_path(self.mlp1(self.norm21(x)))
        )
        return x

from timm.scheduler.cosine_lr import CosineLRScheduler as cos
from torch.optim.lr_scheduler import LRScheduler
import warnings
import math 
class CosineLRScheduler(LRScheduler):  
    def __init__(self, optimizer, T_max, eta_min=0, last_epoch=-1, verbose="deprecated" ,warmup_lr_init=2e-7,warmup_t=3):
        self.T_max = T_max
        self.eta_min = eta_min 
        

        self.warmup_lr_init= warmup_lr_init
        self.warmup_t= warmup_t
        if self.warmup_t:
            self.warmup_steps = [(group['lr'] - warmup_lr_init) / self.warmup_t for group in optimizer.param_groups]
            

        super().__init__(optimizer, last_epoch, verbose)  

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)
        t=self.last_epoch 
        if t == 0:
            return  [self.warmup_lr_init for group in self.optimizer.param_groups]
        if t<=self.warmup_t:
            lrs = [self.warmup_lr_init + t * s for s in self.warmup_steps]

            return lrs
            # return [self.eta_min for group in self.optimizer.param_groups]

        t=self.last_epoch -self.warmup_t

        if t>self.T_max:
            return [self.eta_min for group in self.optimizer.param_groups]

        if t == 0:
            return [group['lr'] for group in self.optimizer.param_groups]
        elif self._step_count == 1 and t > 0:
            return [self.eta_min + (base_lr - self.eta_min) *
                    (1 + math.cos((t) * math.pi / self.T_max)) / 2
                    for base_lr, group in
                    zip(self.base_lrs, self.optimizer.param_groups)]
        elif (t - 1 - self.T_max) % (2 * self.T_max) == 0:
            return [group['lr'] + (base_lr - self.eta_min) *
                    (1 - math.cos(math.pi / self.T_max)) / 2
                    for base_lr, group in
                    zip(self.base_lrs, self.optimizer.param_groups)]
        return [(1 + math.cos(math.pi * t / self.T_max)) /
                (1 + math.cos(math.pi * (t - 1) / self.T_max)) *
                (group['lr'] - self.eta_min) + self.eta_min
                for group in self.optimizer.param_groups]

    def _get_closed_form_lr(self):

        if t>self.T_max:
            return [self.eta_min for group in self.optimizer.param_groups]

        return [self.eta_min + (base_lr - self.eta_min) *
                (1 + math.cos(math.pi * t / self.T_max)) / 2
                for base_lr in self.base_lrs]

class deiqt_models(nn.Module):
    """Vision Transformer with LayerScale (https://arxiv.org/abs/2103.17239) support"""

    def __init__(
        self,
        patch_size=16,
        in_chans=3,
        num_classes=1,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        global_pool=None,
        block_layers=Block,
        Patch_layer=PatchEmbed,
        act_layer=nn.GELU,
        Attention_block=Attention,
        Mlp_block=Mlp,
        init_scale=1e-4,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim

        self.patch_embed = Patch_layer(
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        num_patches = 196
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))

        dpr = [drop_path_rate for i in range(depth)]
        self.blocks = nn.ModuleList(
            [
                block_layers(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=0.0,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                    Attention_block=Attention_block,
                    Mlp_block=Mlp_block,
                    init_values=init_scale,
                )
                for i in range(depth)
            ]
        )

        self.norm = norm_layer(embed_dim)

        self.feature_info = [dict(num_chs=embed_dim, reduction=0, module="head")]
        # self.head = (
        #     nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        # )
        self.head = Experts_MOS(embed_dim=384, juery_nums=6)

        trunc_normal_(self.pos_embed, std=0.02)
        trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token"}

    def get_classifier(self):
        return self.head

    def get_num_layers(self):
        return len(self.blocks)

    def reset_classifier(self, num_classes, global_pool=""):
        self.num_classes = num_classes
        self.head = (
            nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        )

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)

        x = x + self.pos_embed

        x = torch.cat((cls_tokens, x), dim=1)

        for i, blk in enumerate(self.blocks):
            x = blk(x)

        x = self.norm(x)
        return x[:, 0], x[:, 1:, :]

    def forward(self, x):
        ref, x = self.forward_features(x)
        x = self.head(x, ref)
        return x


def build_deiqt(
    patch_size=16,
    embed_dim=384,
    depth=12,
    num_heads=6,
    mlp_ratio=4,
    qkv_bias=True,
    norm_layer=partial(nn.LayerNorm, eps=1e-6),
    block_layers=Layer_scale_init_Block,
    pretrained=False,
    pretrained_model_path="",
    infer=False,
    infer_model_path="",
):
    model = deiqt_models(
        patch_size=patch_size,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        qkv_bias=qkv_bias,
        norm_layer=norm_layer,
        block_layers=block_layers,
    )
    if pretrained:
        assert pretrained_model_path != ""
        checkpoint = torch.load(pretrained_model_path, map_location="cpu")
        state_dict = checkpoint["model"]
        del state_dict["head.weight"]
        del state_dict["head.bias"]
        model.load_state_dict(state_dict, strict=False)
        del checkpoint
        torch.cuda.empty_cache()
    elif infer:
        assert infer_model_path != ""
        checkpoint = torch.load(infer_model_path, map_location="cpu")
        state_dict = checkpoint["model"]
        model.load_state_dict(state_dict, strict=True)
        del checkpoint
        torch.cuda.empty_cache()
    return model

 


from pyiqa.archs.arch_util import  random_crop
class DEIQT(nn.Module):
    def __init__(self):
        super().__init__() 
        self.net = build_deiqt(
            pretrained=True,
            pretrained_model_path=r"D:\cache\deit_3_small_224_21k.pth",
        )   
 
    def forward(self, batch):

        x, t,index,gt = batch['images'],batch['moss'] ,batch['indexs']  ,batch['mossgt']

        x=T.Resize(320)(x)  
        if not self.training:
            x=T.CenterCrop(320)(x)  
        if self.training:
            x = random_crop(x, 224, 10) 
        else:
            x = random_crop(x, 224, 10) 

        scores=self.net(x).squeeze(1) 
        scores=scores.reshape(-1,10).mean(dim=1)
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








if __name__ == "__main__":

    # model = build_deiqt(
    #     patch_size=16,
    #     embed_dim=384,
    #     depth=12,
    #     num_heads=6,
    #     mlp_ratio=4,
    #     qkv_bias=True,
    #     pretrained=True,
    #     pretrained_model_path=" ",
    #     infer=False,
    #     infer_model_path="",
    # ) 

    model = build_deiqt(
        pretrained=False,
        pretrained_model_path=r"D:\cache\deit_3_small_224_21k.pth",
    )

    input1 = torch.randn(1, 3, 224, 224) 