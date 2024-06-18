'''
Author: ll
LastEditTime: 2024-06-16 16:57:15
LastEditors: ll
无问西东
'''
import torch
import torch.nn as nn
import torch.nn.functional as F    

from torchvision import transforms as T
from pyiqa.archs.unique_arch import UNIQUE as unique
from pyiqa.archs.unique_arch import BCNN 
import torchvision

from lightning import LightningModule

 
# from lnl_sr.losses import GCELoss,SCELoss
# from MODEL.encoder import resnet

from collections import defaultdict
from pyiqa.archs.dbcnn_arch import DBCNN as Dbcnn 
from pyiqa.archs.cnniqa_arch import CNNIQA
from pyiqa.archs.wadiqam_arch import WaDIQaM
from pyiqa.archs.nima_arch import NIMA as Nima
from pyiqa.archs.paq2piq_arch import PAQ2PIQ  as paq2piq
from pyiqa.archs.topiq_arch import CFANet 
# from .util.net.wideresnet_lwta import WideResNet
# from .util.net.layers.layers import LWTA
# from .remodel import Conv2dlwta,BatchNorm2dlwta,softquant
# from .remodel import Conv2dkm, Conv2dr,Conv2dzc,Conv2dst,Conv2dstt,Conv2dstta,Conv2dsttexp,Conv2dsttexpo,Conv2dq ,Conv2dqh,softquant,Conv2dqrelu,Conv2dstable,Conv2dsoftstable,Conv2dpiece
# from .resnet import resnet50 as resnet50o
# from ModEnsNet.util.resnext import ResNet as ResNeXt
# from ModEnsNet.util.mobile_v2 import MobileNetV2
# from torch.nn.modules.utils import _single, _pair, _triple, _reverse_repeat_tuple 
from pyiqa.archs.maniqa_arch import  MANIQA
import yaml
from pyiqa.archs.tres_arch import TReS  as TReSarch
from einops import rearrange
from pyiqa.archs.hypernet_arch import HyperNet as HyperNetarch 
from pyiqa.archs.musiq_arch import MUSIQ as Musiq 
from pyiqa.archs.arch_util import dist_to_mos, load_pretrained_network,random_crop 

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
from functools import partial
torch.optim.lr_scheduler.CosineAnnealingLR
from torch.optim.lr_scheduler import LRScheduler
import math 
import warnings
class CosineAnnealingminLR(LRScheduler): 
    def __init__(self, optimizer, T_max, eta_min=0, last_epoch=-1, verbose="deprecated" ):
        self.T_max = T_max
        self.eta_min = eta_min 
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)
        if self.last_epoch>self.T_max:
            return [self.eta_min for group in self.optimizer.param_groups]

        if self.last_epoch == 0:
            return [group['lr'] for group in self.optimizer.param_groups]
        elif self._step_count == 1 and self.last_epoch > 0:
            return [self.eta_min + (base_lr - self.eta_min) *
                    (1 + math.cos((self.last_epoch) * math.pi / self.T_max)) / 2
                    for base_lr, group in
                    zip(self.base_lrs, self.optimizer.param_groups)]
        elif (self.last_epoch - 1 - self.T_max) % (2 * self.T_max) == 0:
            return [group['lr'] + (base_lr - self.eta_min) *
                    (1 - math.cos(math.pi / self.T_max)) / 2
                    for base_lr, group in
                    zip(self.base_lrs, self.optimizer.param_groups)]
        return [(1 + math.cos(math.pi * self.last_epoch / self.T_max)) /
                (1 + math.cos(math.pi * (self.last_epoch - 1) / self.T_max)) *
                (group['lr'] - self.eta_min) + self.eta_min
                for group in self.optimizer.param_groups]

    def _get_closed_form_lr(self):

        if self.last_epoch>self.T_max:
            return [self.eta_min for group in self.optimizer.param_groups]

        return [self.eta_min + (base_lr - self.eta_min) *
                (1 + math.cos(math.pi * self.last_epoch / self.T_max)) / 2
                for base_lr in self.base_lrs]
class MANIQ(nn.Module):
    def __init__(self):
        super().__init__()      
        self.net = MANIQA(pretrained=False)     
    #     self.net.forward = self.netforward.__get__(self.net, MANIQA)

    # def netforward(self, x):
    #     """
    #     Forward pass of the MANIQA model.

    #     Args:
    #         x (torch.Tensor): Input image tensor.

    #     Returns:
    #         torch.Tensor: Predicted quality score for the input image.
    #     """
    #     x = (x - self.default_mean.to(x)) / self.default_std.to(x)
    #     bsz = x.shape[0]

    #     if self.training:
    #         x = random_crop(x, crop_size=224, crop_num=1)
    #     else:
    #         x = random_crop(x, crop_size=224, crop_num=self.test_sample)

    #     _x = self.vit(x)
    #     x = self.extract_feature(self.save_output)
    #     self.save_output.outputs.clear()

    #     # stage 1
    #     x = rearrange(x, 'b (h w) c -> b c (h w)', h=self.input_size, w=self.input_size)
    #     for tab in self.tablock1:
    #         x = tab(x)
    #     x = rearrange(x, 'b c (h w) -> b c h w', h=self.input_size, w=self.input_size)
    #     x = self.conv1(x)
    #     x = self.swintransformer1(x)

    #     # stage2
    #     x = rearrange(x, 'b c h w -> b c (h w)', h=self.input_size, w=self.input_size)
    #     for tab in self.tablock2:
    #         x = tab(x)
    #     x = rearrange(x, 'b c (h w) -> b c h w', h=self.input_size, w=self.input_size)
    #     x = self.conv2(x)
    #     x = self.swintransformer2(x)

    #     x = rearrange(x, 'b c h w -> b (h w) c', h=self.input_size, w=self.input_size)

    #     per_patch_score = self.fc_score(x)
    #     per_patch_score = per_patch_score.reshape(bsz, -1)
    #     per_patch_weight = self.fc_weight(x)
    #     per_patch_weight = per_patch_weight.reshape(bsz, -1)

    #     score = (per_patch_weight * per_patch_score).sum(dim=-1) / (per_patch_weight.sum(dim=-1) + 1e-8)
    #     return score.unsqueeze(1)


    def forward(self, batch):
        
        x, t,index,gt = batch['images'],batch['moss'] ,batch['indexs']  ,batch['mossgt']

        x=T.Resize(320)(x)  
        if not self.training:
            x=T.CenterCrop(320)(x)  

        scores= self.net(x).squeeze(1)

        loss=F.mse_loss(scores  ,t ) 
        lossfun=partial(F.mse_loss,scores)

        logdict = {'mse_score':[scores,gt] ,  
                   'rmse_score':[scores,gt],  
                   'mean_loss':[ loss.detach()],
                   'srcc':[scores,gt],
                   'plcc':[scores,gt],
                   } 

        out={'scores':scores,'loss':loss,'logdict':logdict,'lossfun':lossfun}
        return out

class MUSIQ(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=Musiq()
    def forward(self, batch):
        
        x, t,index,gt = batch['images'],batch['moss'] ,batch['indexs']  ,batch['mossgt']

        x=T.Resize(320)(x)  
        if not self.training:
            x=T.CenterCrop(320)(x)  

        scores= self.net(x)

        loss=F.mse_loss(scores  ,t ) 

        logdict = {'mse_score':[scores,gt] ,  
                   'rmse_score':[scores,gt],  
                   'mean_loss':[torch.zeros_like(loss),loss.detach()],
                   'srcc':[scores,gt],
                   'plcc':[scores,gt],
                   } 

        out={'scores':scores,'loss':loss,'logdict':logdict}
        return out
class PAQ2PIQ(nn.Module):
    def __init__(self):
        super().__init__() 
        self.net = paq2piq( pretrained=False)
    def forward(self, batch):
        
        x, t,index,gt = batch['images'],batch['moss'] ,batch['indexs']  ,batch['mossgt']

        x=T.Resize(320)(x)  
        if not self.training:
            x=T.CenterCrop(320)(x)  

        scores= self.net(x)

        loss=F.mse_loss(scores  ,t ) 

        logdict = {'mse_score':[scores,gt] ,  
                   'rmse_score':[scores,gt],  
                   'mean_loss':[torch.zeros_like(loss),loss.detach()],
                   'srcc':[scores,gt],
                   'plcc':[scores,gt],
                   } 

        out={'scores':scores,'loss':loss,'logdict':logdict}
        return out

class NIMA(Nima):
    def __init__(self):
        super().__init__(num_classes=1,pretrained=False) 
        # super().__init__(num_classes=10,pretrained=False) 
        # super().__init__(num_classes=10,pretrained=False) 
        # self.net = Nima(num_classes=1,pretrained=False) 
        # self.base_model.default_cfg['input_size']
        # self.net = NIMA(num_classes=10,pretrained=False)

        # self.classifier = [nn.Flatten(),
        #                 #    nn.Dropout(p=dropout_rate),
                        
        #                    nn.Linear(in_features=512, out_features=1),
        #                    ]

        # self.classifier = [nn.Flatten(),
        #                    nn.Dropout(p=0.0),
        #                    nn.Linear(in_features=512, out_features=1), 
        #                    ]

        self.classifier =nn.Sequential(nn.Flatten(),
                           nn.Dropout(p=0.2),
                           nn.Linear(in_features=512, out_features=256),
                           nn.ReLU(), 
                           nn.Linear(in_features=256, out_features=1),
                           )

    def forward(self, batch):
        
        x, t,index,gt = batch['images'],batch['moss'] ,batch['indexs']  ,batch['mossgt']

        x=T.Resize(320)(x)  
        if not self.training:
            x=T.CenterCrop(320)(x)  

        x = (x - self.default_mean.to(x)) / self.default_std.to(x)
        x = self.base_model(x)[-1]
        x = self.global_pool(x)
        dist = self.classifier(x)
        scores = dist_to_mos(dist) .squeeze(1)
        # scores=(scores-1)/10.0
        scores=F.sigmoid (scores)
        # scores=F.sigmoid (self.net(x)).squeeze(1)





        loss=F.mse_loss(scores  ,t ) 
        lossfun=partial(F.mse_loss,scores)

        logdict = {'mse_score':[scores,gt] ,  
                   'rmse_score':[scores,gt],  
                   'mean_loss':[torch.zeros_like(loss),loss.detach()],
                   'srcc':[scores,gt],
                   'plcc':[scores,gt],
                   } 

        out={'scores':scores,'loss':loss,'logdict':logdict,'lossfun':lossfun} 
        return out

# class Wadiqam(nn.Module):
#     def __init__(self):
#         super().__init__() 
#         self.net = WaDIQaM(metric_mode='NR')
#     def forward(self, x):
#         x=self.net(x)
#         return x
from torchvision.models.resnet import resnet50 
class HyperNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=HyperNetarch(pretrained=False)


        class resnetbase_model(nn.Module):
            def __init__(self):
                super().__init__() 
                models = resnet50(pretrained=True)
                models.fc=nn.Sequential(
                    nn.Linear(2048,1),       
                    # nn.ReLU(),            
                    # nn.Linear(1024,256),       
                    # nn.ReLU(),
                    # nn.Linear(256,1),       
                    nn.Sigmoid()          
                )          

                for k,v in models.__dict__['_modules'].items():
                    setattr(self,k,v)   
            def forward(self, x):
                xall=[]
                x = self.conv1(x)
                x = self.bn1(x)  
                x = self.relu(x)
                x = self.maxpool(x)
                xall.append(x)
                x = self.layer1(x)  
                xall.append(x)
                x = self.layer2(x)  
                xall.append(x)
                x = self.layer3(x)  
                xall.append(x)
                x = self.layer4(x) 
                xall.append(x)

                # x = self.avgpool(x)
                # x = torch.flatten(x, 1)
                # x = self.fc(x)
                return xall        
        # models = resnet50(pretrained=True)       
        # self.base_model = nn.Sequential(*list(models.children())[0:8])
        self.net.base_model=resnetbase_model()

    def forward(self, batch):
        
        x, t,index,gt = batch['images'],batch['moss'] ,batch['indexs']  ,batch['mossgt']

        x=T.Resize(320)(x)  
        if not self.training:
            pass 
        else:
            x=T.RandomCrop(224)(x)  

        scores=self.net(x).squeeze(1)

        loss=F.mse_loss(scores,t)
        lossfun=partial(F.mse_loss,scores)

        logdict = {'mse_score':[scores,gt] ,  
                   'rmse_score':[scores,gt],  
                   'mean_loss':[torch.zeros_like(loss),loss.detach()],
                   'srcc':[scores,gt],
                   'plcc':[scores,gt],
                   } 

        out={'scores':scores,'loss':loss,'logdict':logdict,'lossfun':lossfun}
        return out

class TReS(nn.Module):
    def __init__(self):
        super().__init__()          
        self.net = TReSarch(pretrained=False )  
        self.l1_loss=nn.L1Loss()
        # for k,v in models.__dict__ .items():
        #     setattr(self,k,v)   
    #     self.net.forward = self.netforward.__get__(self.net, TReS)

    # def netforward(self, x):
    #     x = (x - self.default_mean.to(x)) / self.default_std.to(x)
    #     bsz = x.shape[0] 
    #     if self.training:
    #         x = random_crop(x, 224, 1)
    #         num_patches = 1
    #     else:
    #         x = random_crop(x, 224, self.test_sample)
    #         num_patches = self.test_sample

    #     self.pos_enc_1 = self.position_embedding(torch.ones(1, self.dim_modelt, 7, 7).to(x))
    #     self.pos_enc = self.pos_enc_1.repeat(x.shape[0], 1, 1, 1).contiguous()

    #     out, layer1, layer2, layer3, layer4 = self.forward_backbone(self.model, x)

    #     layer1_t = self.avg8(self.drop2d(self.L2pooling_l1(F.normalize(layer1, dim=1, p=2))))
    #     layer2_t = self.avg4(self.drop2d(self.L2pooling_l2(F.normalize(layer2, dim=1, p=2))))
    #     layer3_t = self.avg2(self.drop2d(self.L2pooling_l3(F.normalize(layer3, dim=1, p=2))))
    #     layer4_t = self.drop2d(self.L2pooling_l4(F.normalize(layer4, dim=1, p=2)))
    #     layers = torch.cat((layer1_t, layer2_t, layer3_t, layer4_t), dim=1)

    #     out_t_c = self.transformer(layers, self.pos_enc)
    #     out_t_o = torch.flatten(self.avg7(out_t_c), start_dim=1)
    #     out_t_o = self.fc2(out_t_o)
    #     layer4_o = self.avg7(layer4)
    #     layer4_o = torch.flatten(layer4_o, start_dim=1)
    #     predictionQA = self.fc(torch.flatten(torch.cat((out_t_o, layer4_o), dim=1), start_dim=1))

    #     fout, flayer1, flayer2, flayer3, flayer4 = self.forward_backbone(self.model, torch.flip(x, [3]))
    #     flayer1_t = self.avg8(self.L2pooling_l1(F.normalize(flayer1, dim=1, p=2)))
    #     flayer2_t = self.avg4(self.L2pooling_l2(F.normalize(flayer2, dim=1, p=2)))
    #     flayer3_t = self.avg2(self.L2pooling_l3(F.normalize(flayer3, dim=1, p=2)))
    #     flayer4_t = self.L2pooling_l4(F.normalize(flayer4, dim=1, p=2))
    #     flayers = torch.cat((flayer1_t, flayer2_t, flayer3_t, flayer4_t), dim=1)
    #     fout_t_c = self.transformer(flayers, self.pos_enc)
    #     fout_t_o = torch.flatten(self.avg7(fout_t_c), start_dim=1)
    #     fout_t_o = (self.fc2(fout_t_o))
    #     flayer4_o = self.avg7(flayer4)
    #     flayer4_o = torch.flatten(flayer4_o, start_dim=1)
    #     fpredictionQA = (self.fc(torch.flatten(torch.cat((fout_t_o, flayer4_o), dim=1), start_dim=1)))

    #     consistloss1 = self.consistency(out_t_c, fout_t_c.detach())
    #     consistloss2 = self.consistency(layer4, flayer4.detach())
    #     consistloss = 1 * (consistloss1 + consistloss2)

    #     predictionQA = predictionQA.reshape(bsz, num_patches, 1)
    #     predictionQA = predictionQA.mean(dim=1)

    #     if self.training:
    #         return predictionQA, consistloss
    #     else:
    #         return predictionQA
 

    def forward(self, batch): 

        x, t,index,gt = batch['images'],batch['moss'] ,batch['indexs']  ,batch['mossgt']

        x=T.Resize(320)(x)  
        # if not self.training:
        #     x=T.CenterCrop(320)(x)  
        if self.training: 



            img=x;label=t
            pred,closs = self.net(img)
            # loss=F.mse_loss(pred,t)+2*closs 
            # loss=F.mse_loss(pred,t)
            pred2,closs2 = self.net(torch.flip(img, [3]))   

            loss_qa = self.l1_loss(pred.squeeze(), label.float().detach())
            loss_qa2 = self.l1_loss(pred2.squeeze(), label.float().detach())
            # =============================================================================
            # =============================================================================

            indexlabel = torch.argsort(label) # small--> large
            anchor1 = torch.unsqueeze(pred[indexlabel[0],...].contiguous(),dim=0) # d_min
            positive1 = torch.unsqueeze(pred[indexlabel[1],...].contiguous(),dim=0) # d'_min+
            negative1_1 = torch.unsqueeze(pred[indexlabel[-1],...].contiguous(),dim=0) # d_max+

            anchor2 = torch.unsqueeze(pred[indexlabel[-1],...].contiguous(),dim=0)# d_max
            positive2 = torch.unsqueeze(pred[indexlabel[-2],...].contiguous(),dim=0)# d'_max+
            negative2_1 = torch.unsqueeze(pred[indexlabel[0],...].contiguous(),dim=0)# d_min+

            # =============================================================================
            # =============================================================================

            fanchor1 = torch.unsqueeze(pred2[indexlabel[0],...].contiguous(),dim=0)
            fpositive1 = torch.unsqueeze(pred2[indexlabel[1],...].contiguous(),dim=0)
            fnegative1_1 = torch.unsqueeze(pred2[indexlabel[-1],...].contiguous(),dim=0)

            fanchor2 = torch.unsqueeze(pred2[indexlabel[-1],...].contiguous(),dim=0)
            fpositive2 = torch.unsqueeze(pred2[indexlabel[-2],...].contiguous(),dim=0)
            fnegative2_1 = torch.unsqueeze(pred2[indexlabel[0],...].contiguous(),dim=0)



            consistency = nn.L1Loss()
            assert (label[indexlabel[-1]]-label[indexlabel[1]])>=0
            assert (label[indexlabel[-2]]-label[indexlabel[0]])>=0
            triplet_loss1 = nn.TripletMarginLoss(margin=(label[indexlabel[-1]]-label[indexlabel[1]]), p=1) # d_min,d'_min,d_max
            # triplet_loss2 = nn.TripletMarginLoss(margin=label[indexlabel[0]], p=1)
            triplet_loss2 = nn.TripletMarginLoss(margin=(label[indexlabel[-2]]-label[indexlabel[0]]), p=1)
            # triplet_loss1 = nn.TripletMarginLoss(margin=label[indexlabel[-1]], p=1)
            # triplet_loss2 = nn.TripletMarginLoss(margin=label[indexlabel[0]], p=1)
            tripletlosses = triplet_loss1(anchor1, positive1, negative1_1) + \
                triplet_loss2(anchor2, positive2, negative2_1)
            ftripletlosses = triplet_loss1(fanchor1, fpositive1, fnegative1_1) + \
                triplet_loss2(fanchor2, fpositive2, fnegative2_1)
            
            loss = loss_qa + closs + loss_qa2 + closs2 + 0.5*( self.l1_loss(tripletlosses,ftripletlosses.detach())+ self.l1_loss(ftripletlosses,tripletlosses.detach()))+0.05*(tripletlosses+ftripletlosses)
 

        else:
            pred=self.net(x)
            loss=torch.zeros_like(x)
        scores=pred
        scores=scores.squeeze(1)


        logdict = {'mse_score':[scores,gt] ,  
                   'rmse_score':[scores,gt],  
                   'mean_loss':[torch.zeros_like(loss),loss.detach()],
                   'srcc':[scores,gt],
                   'plcc':[scores,gt],
                   } 

        out={'scores':scores,'loss':loss,'logdict':logdict}
        return out

class TOPIQ(nn.Module):
    def __init__(self):
        super().__init__()  
        with open('MODEL/OOD/TOPIQ.yaml') as f:
            options = yaml.safe_load(f)  
        options['network'].pop('type')      
        self.net = CFANet(**options['network'])      
    def forward(self, batch): 

        x, t,index,gt = batch['images'],batch['moss'] ,batch['indexs']  ,batch['mossgt']

        x=T.Resize(320)(x)  
        if not self.training:
            x=T.CenterCrop(320)(x)  

        scores=self.net(x).squeeze(1)
 

        loss=F.mse_loss(scores,t)
        lossfun=partial(F.mse_loss,scores)

        logdict = {'mse_score':[scores,gt] ,  
                   'rmse_score':[scores,gt],  
                   'mean_loss':[ loss.detach()],
                   'srcc':[scores,gt],
                   'plcc':[scores,gt],
                   } 


        out={'scores':scores,'loss':loss,'logdict':logdict,'lossfun':lossfun}
        # out={'scores':scores,'loss':loss,'logdict':logdict}
        return out

class DBCNN(nn.Module):
    def __init__(self):
        super().__init__() 
        opt={'pretrained_scnn_path': 'DBCNN-PyTorch/pretrained_scnn/scnn.pkl', 'pretrained': False,'fc':False}
        self.net = Dbcnn(**opt)      
 
    def forward(self, batch):

        x, t,index,gt = batch['images'],batch['moss'] ,batch['indexs']  ,batch['mossgt']

        x=T.Resize(320)(x)  
        if not self.training:
            x=T.CenterCrop(320)(x)  

 
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



class resnet(nn.Module):
    def __init__(self):
        super().__init__() 
        models = torchvision.models.resnet50( weights='IMAGENET1K_V1')

        models.fc=nn.Sequential(  
            nn.Linear(2048,1),    
            nn.Sigmoid()        
        )               

        # models.fc=nn.Sequential(
        #                     nn.Flatten(),
        #                    nn.Dropout(p=0.2), 
        #     nn.Linear(2048,1024),  
        #     nn.ReLU(),
        #     nn.Dropout(p=0.2), 
        #     nn.Linear(1024,512),  
        #     nn.ReLU(),
        #     nn.Dropout(p=0.2), 
        #     nn.Linear(512,256),  
        #     nn.ReLU(),
        #     nn.Linear(256,1),    
        #     nn.Sigmoid()        
        # )                   

        for k,v in models.__dict__['_modules'].items():
            setattr(self,k,v)   
    def forward(self, batch):

        x, t,index,gt = batch['images'],batch['moss'] ,batch['indexs']  ,batch['mossgt']

        x=T.Resize(320)(x)  
        if not self.training:
            x=T.CenterCrop(320)(x)  


        # x = (x - torch.tensor(IMAGENET_DEFAULT_MEAN).reshape(1,3,1,1).to(x)  ) / torch.tensor(IMAGENET_DEFAULT_STD).reshape(1,3,1,1).to(x)

        x = self.conv1(x)
        x = self.bn1(x)  
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)  
        x = self.layer2(x)  
        x = self.layer3(x)  
        x = self.layer4(x) 

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x) 
        scores=x.squeeze(1)
 

        loss=F.mse_loss(scores,t)
        lossfun=partial(F.mse_loss,scores)

        logdict = {'mse_score':[scores,gt] ,  
                   'rmse_score':[scores,gt],  
                   'mean_loss':[torch.zeros_like(loss),loss.detach()],
                   'srcc':[scores,gt],
                   'plcc':[scores,gt],
                   } 


        out={'scores':scores,'loss':loss,'logdict':logdict,'lossfun':lossfun}
        # out={'scores':scores,'loss':loss,'logdict':logdict}
        return out
 


EPS = 1e-2
esp = 1e-8

# class Fidelity_Loss(torch.nn.Module):

#     def __init__(self):
#         super(Fidelity_Loss, self).__init__()

#     def forward(self, p, g):
#         g = g.view(-1, 1)
#         p = p.view(-1, 1)
#         # loss = 1 - (  torch.sqrt(p * g + esp) + torch.sqrt((1 - p) * (1 - g) + esp))
#         loss = 1 - (  torch.sqrt((p * g).abs() + esp) * torch.sign(p*g) + torch.sqrt(((1 - p) * (1 - g)).abs() + esp)* torch.sign(p*g) )
#         return torch.mean(loss)
    
class Fidelity_Loss(torch.nn.Module):

    def __init__(self):
        super(Fidelity_Loss, self).__init__()

    def forward(self, p, g):
        
        g = g.view(-1, 1)
        p = p.view(-1, 1)
        loss = 1 - (torch.sqrt(p * g + eps) + torch.sqrt((1 - p) * (1 - g) + eps))

        return torch.mean(loss)

class UNIQUE(nn.Module):
    def __init__(self):
        super().__init__()         
        self.net = unique( )   
        self.net.backbone = torchvision.models.resnet34(weights='IMAGENET1K_V1')
        self.net.representation = BCNN()

        self.net.fc = nn.Sequential(
            nn.Linear(512*512,1),            
        )           

        # self.net.fc = nn.Sequential(
        #     nn.Linear(512*512,1024),    
        #     nn.ReLU(),
        #     nn.Linear(1024,512),  
        #     nn.ReLU(),
        #     nn.Linear(512,1),  
        #     nn.Sigmoid()        
        # )            

        self.loss_fn=Fidelity_Loss()
        # self.loss_fn=nn.MSELoss()
        # for k,v in models.__dict__ .items():
        #     setattr(self,k,v)   
    def lossfun(self,scores,target):
        t=target
        t1,t2=t.chunk(2)  
        constant = torch.sqrt(torch.Tensor([1])).to(t.device)
        g = 0.5 * (1 + torch.erf(t1-t2 / constant))

        
        
        # scores=torch.cat(scores,dim=1)
        y1,y2=scores.chunk(2)
        y_diff = y1 - y2   
        p=y_diff 
        p = 0.5 * (1 + torch.erf(p / constant))
        loss = self.loss_fn(p , g.detach())

        # y1,y2=scores.chunk(2)
        # g,yb=t.chunk(2)
        # g = g.view(-1, 1)
        # yb = yb.view(-1, 1)
        # g=g-yb
        # y_diff = y1 - y2
        # p = y_diff

        # # loss = self.loss_fn(p, g.detach())*0.5   +F.mse_loss(scores,t)*0.5  
        # loss =loss*0.1   +F.mse_loss(scores,t)
        loss =loss 
        return loss      
    def forward(self, batch): 

        x, t,index,gt = batch['images'],batch['moss'] ,batch['indexs']  ,batch['mossgt']

        x=T.Resize(320)(x)  
        if not self.training:
            x=T.CenterCrop(320)(x)   

        scores=self.net(x) 
        scores=F.sigmoid(scores)



        if self.training:
            # self.lossfun(scores,t)
            lossfun=partial(self.lossfun,scores)

            
        else: 
            lossfun=partial(F.mse_loss,scores) 

        loss=lossfun(target=t)
        logdict = {'mse_score':[scores,gt] ,  
                   'rmse_score':[scores,gt],  
                   'mean_loss':[torch.zeros_like(loss),loss.detach()],
                   'srcc':[scores,gt],
                   'plcc':[scores,gt]
                   
                   } 


        out={'scores':scores,'loss':loss,'logdict':logdict,"lossfun": lossfun }
        return out
    


import torch.nn as nn
from torchvision import models
import torch
import torch.nn.functional as F
from .SCNN import SCNN
from copy import deepcopy


class DBCNN0(nn.Module):

    def __init__(self,JL=False,n_task=1):
        """Declare all needed layers."""
        nn.Module.__init__(self)
        # Convolution and pooling layers of VGG-16.
        self.n_task = n_task
        self.backbone = models.resnet18(pretrained=True) 

        scnn = SCNN()
        scnn = torch.nn.DataParallel(scnn).cuda()
        scnn.load_state_dict(torch.load(r'D:\doingproject\BIQA_CL\saved_weights\scnn.pkl'))
        self.sfeatures = scnn.module.features
        self.pooling = nn.AdaptiveAvgPool2d((1, 1))

 
        
        self.fc = nn.ModuleList()
        # using deepcopy to insure each fc layer initialized from the same parameters
        # (for fair comparision with sequentail/individual training)
        if JL:
            fc = nn.Linear(512 * 128, 1, bias=True)
        else:
            fc = nn.Linear(512 * 128, 1, bias=False)
        # Initialize the fc layers.
        nn.init.kaiming_normal_(fc.weight.data)
        for i in range(0, self.n_task):
            self.fc.append(deepcopy(fc))



    def forward(self, x):
        """Forward pass of the network.
        """
        N = x.size()[0]

        x1 = self.backbone.conv1(x)
        x1 = self.backbone.bn1(x1)
        x1 = self.backbone.relu(x1)
        x1 = self.backbone.maxpool(x1)

        x1 = self.backbone.layer1(x1)
        x1 = self.backbone.layer2(x1)
        x1 = self.backbone.layer3(x1)
        x1 = self.backbone.layer4(x1)

        H = x1.size()[2]
        W = x1.size()[3]
        assert x1.size()[1] == 512

        x2 = self.sfeatures(x)
        H2 = x2.size()[2]
        W2 = x2.size()[3]
        assert x2.size()[1] == 128

        sfeat = self.pooling(x2)
        sfeat = sfeat.squeeze(3).squeeze(2)
        sfeat = F.normalize(sfeat, p=2)

        if (H != H2) | (W != W2):
            x2 = F.upsample_bilinear(x2, (H, W))

        x1 = x1.view(N, 512, H * W)
        x2 = x2.view(N, 128, H * W)
        x = torch.bmm(x1, torch.transpose(x2, 1, 2)) / (H * W)  # Bilinear
        assert x.size() == (N, 512, 128)
        x = x.view(N, 512 * 128)
        #x = torch.sqrt(x + 1e-8)
        x = F.normalize(x)

        output = []

        for idx, fc in enumerate(self.fc):
            if self.training==False:
                for W in fc.parameters():
                    fc.weight.data = F.normalize(W, p=2, dim=1)
            output.append(fc(x)) 


        # return output, sfeat
        # output=self.fc[0](x)
        return output 
        

eps = 1e-8

class LWF(nn.Module):
    def __init__(self):
        super().__init__()         
        self.net = DBCNN0()
        # self.net.fc = nn.Sequential(
        #     nn.Linear(512*512,1024),    
        #     nn.ReLU(),
        #     nn.Linear(1024,512),  
        #     nn.ReLU(),
        #     nn.Linear(512,1),  
        #     nn.Sigmoid()        
        # )            
        
        self.model_old=deepcopy(self.net)
        self.model_old.eval()  
        # if not hasattr(self, 'model_old'):
        #     setattr(self, 'model_old', deepcopy(self.net))
        #     self.model_old.eval() 
        self.loss_fn=Fidelity_Loss() 
        self.distll=False

        # self.loss_fn=nn.MSELoss()
        # for k,v in models.__dict__ .items():
        #     setattr(self,k,v)   
    # def lossfun(self,scores,t):
    #     y1,y2=scores.chunk(2)
    #     g,yb=t.chunk(2)
    #     g = g.view(-1, 1)
    #     yb = yb.view(-1, 1)
    #     g=g-yb
    #     y_diff = y1 - y2
    #     p = y_diff

    #     output = []
    #     for y11, y22 in zip(y1, y2):
    #         y_diff = y11 - y22
    #         p = y_diff
    #         if self.config.fidelity:
    #             constant = torch.sqrt(torch.Tensor([2])).to(self.device)
    #             p = 0.5 * (1 + torch.erf(p / constant))
    #         output.append(p)
    #     p=output
    #     loss = self.loss_fn(p[self.config.task_id], g.detach())



    #     # loss = self.loss_fn(p, g.detach())*0.5   +F.mse_loss(scores,t)*0.5  
    #     # loss = self.loss_fn(p, g.detach())*0.1   +F.mse_loss(scores,t)
    #     return loss      
    # def do_batch(self, x1, x2):
    #     y1, _ = self.model(x1)
    #     y2, _ = self.model(x2)

    #     output = []
    #     for y11, y22 in zip(y1, y2):
    #         y_diff = y11 - y22
    #         p = y_diff
    #         if self.config.fidelity:
    #             constant = torch.sqrt(torch.Tensor([2])).to(self.device)
    #             p = 0.5 * (1 + torch.erf(p / constant))
    #         output.append(p)

    #     return output
    # def do_batch_old(self, x1, x2):
    #     if self.config.shared_head:
    #         n_old_task = 1
    #     else:
    #         n_old_task = self.config.task_id
    #     ps = []
    #     y1, _ = self.model_old(x1)
    #     y2, _ = self.model_old(x2)
    #     for task_idx in range(n_old_task):
    #         y_diff = y1[task_idx] - y2[task_idx]
    #         p = y_diff

    #         if self.config.fidelity:
    #             constant = torch.sqrt(torch.Tensor([2])).to(self.device)
    #             p = 0.5 * (1 + torch.erf(p / constant))
    #             ps.append(p)
    #             #p = 0.5 * (1 + torch.erf(p / constant))
    #     return ps
    def lossfun(self,scores,target,scores1=None,taskidx=0):
        t=target
        t1,t2=t.chunk(2)  
        constant = torch.sqrt(torch.Tensor([1])).to(t.device)
        g = 0.5 * (1 + torch.erf(t1-t2 / constant))

        
        
        # scores=torch.cat(scores,dim=1)
        y1,y2=scores.chunk(2)
        y_diff = y1 - y2   
        p=y_diff 
        p = 0.5 * (1 + torch.erf(p / constant))
        loss = self.loss_fn(p , g.detach())

        if self.distll==True:
            
            # scores=torch.cat(scores,dim=1)
            y1,y2=scores1.chunk(2)
            y_diff = y1 - y2   
            pold=y_diff 
            pold = 0.5 * (1 + torch.erf(pold / constant))
            loss =loss+10* self.loss_fn(p , pold.detach())
        # loss = self.loss_fn(p, g.detach())*0.5   +F.mse_loss(scores,t)*0.5  
        loss = loss 
        # loss = loss*0.1  +F.mse_loss(scores,t)
        return loss   

    def forward(self, batch): 
        x, t,index,gt,taskidx = batch['images'],batch['moss'] ,batch['indexs']  ,batch['mossgt'],batch['taskidx']
        
        x=T.Resize(320)(x)  
        if not self.training:
            x=T.CenterCrop(320)(x)   


            

        



        # scores=self.net(x) 



        # if  self.training: 
        #     loss=   self.lossfun(scores,t )
        #     # lossfun=partial( self.lossfun,scores)
        #     lossfun=None

            
        # else:
        #     loss=F.mse_loss(scores,t)
        #     lossfun=None
        scores=self.net(x) .squeeze(1)  
        scores=(scores+1)/2
        # scores=F.sigmoid(scores)
        if self.training:
            lossfun=partial(self.lossfun,scores=scores) 

            if self.distll==True:
                scores1=self.model_old(x) .squeeze(1)
                lossfun=partial(self.lossfun,scores=scores,scores1=scores1) 


        else:
            scores=self.net(x) .squeeze(1)
            lossfun=partial(F.mse_loss,scores) 
        loss=lossfun(target=t)
            # scores=torch.cat(scores,dim=1)
            # scores=scores.mean(dim=1)
        # p = self.do_batch(x1, x2)
        # # self.loss = self.loss_fn(p[task_id], g.detach())
        # with torch.no_grad():
        #     p_old = self.do_batch_old(x1, x2)
        # for i, old_pred in enumerate(p_old):
        #     reg_loss +=   10* self.loss_fn(p[i], old_pred.detach())





















        
        # loss=F.mse_loss(scores,t)
        logdict = {'mse_score':[scores,gt] ,  
                   'rmse_score':[scores,gt],  
                   'mean_loss':[torch.zeros_like(loss),loss.detach()],
                   'srcc':[scores,gt],
                   'plcc':[scores,gt],
                   } 


        out={'scores':scores,'loss':loss,'logdict':logdict,'lossfun':lossfun }
        return out
    
 
import glob 
 
class CQA(nn.Module):
    def __init__(self,taskid=0,checkpointpath=None):
        super().__init__()         
        self.net = DBCNN0(n_task=4)
        self.taskid=taskid
        # self.net.fc = nn.Sequential(
        #     nn.Linear(512*512,1024),    
        #     nn.ReLU(),
        #     nn.Linear(1024,512),  
        #     nn.ReLU(),
        #     nn.Linear(512,1),  
        #     nn.Sigmoid()        
        # )            
        
        self.model_old=deepcopy(self.net)
        self.model_old.eval()  
        # if not hasattr(self, 'model_old'):
        #     setattr(self, 'model_old', deepcopy(self.net))
        #     self.model_old.eval() 
        self.loss_fn=Fidelity_Loss()  

        if checkpointpath!=None:

            checkpointpath=glob.glob(checkpointpath)    [0]
            item=torch.load(checkpointpath,map_location='cpu')['state_dict']
            state_dict = self.state_dict()

            for k, v in item.items():  
                if 'model.net.' in k:
                    k = k.replace('model.net.', '')
                    if  'taskid'  in k   :
                        continue 
                    state_dict.update({k: v})
            # self.load_state_dict(state_dict) 
            self.load_state_dict(state_dict,strict=True)  
        self.model_old=deepcopy(self.net)
        self.model_old.eval()  
    def lossfun(self,scores,target,scores1=None,taskidx=0):
        t=target
        t1,t2=t.chunk(2)  
        constant = torch.sqrt(torch.Tensor([1])).to(t.device)
        g = 0.5 * (1 + torch.erf(t1-t2 / constant))

        
        
        # scores=torch.cat(scores,dim=1)
        y1,y2=scores.chunk(2)
        y_diff = y1 - y2   
        p=y_diff 
        p = 0.5 * (1 + torch.erf(p / constant))
        loss = self.loss_fn(p , g.detach())

        if self.taskid>=1:
            
            # scores=torch.cat(scores,dim=1)
            for i in range(0,self.taskid):
                y1,y2=scores1[i].chunk(2)
                y_diff = y1 - y2   
                pold=y_diff 
                pold = 0.5 * (1 + torch.erf(pold / constant))
                loss =loss+5* self.loss_fn(p , pold.detach())
        # loss = self.loss_fn(p, g.detach())*0.5   +F.mse_loss(scores,t)*0.5  
        loss = loss 
        # loss = loss*0.1  +F.mse_loss(scores,t)
        return loss   

    def forward(self, batch): 
        x, t,index,gt,taskidx = batch['images'],batch['moss'] ,batch['indexs']  ,batch['mossgt'],batch['taskidx']
        
        x=T.Resize(320)(x)  
        if not self.training:
            x=T.CenterCrop(320)(x)   


        scores=self.net(x)[self.taskid] .squeeze(1)   
        scores=(scores+1)/2
        # scores=F.sigmoid(scores)
        if self.training:
            lossfun=partial(self.lossfun,scores=scores) 

            if self.taskid>=1:
                scores1=[]
                for i in range(0,self.taskid):
                    scores1.append(self.model_old(x)[self.taskid-1] .squeeze(1))
                lossfun=partial(self.lossfun,scores=scores,scores1=scores1) 


        else:
            scores=self.net(x) [taskidx].squeeze(1)
            lossfun=partial(F.mse_loss,scores) 
        loss=lossfun(target=t) 
        
        # loss=F.mse_loss(scores,t)
        logdict = {'mse_score':[scores,gt] ,  
                   'rmse_score':[scores,gt],  
                   'mean_loss':[torch.zeros_like(loss),loss.detach()],
                   'srcc':[scores,gt],
                   'plcc':[scores,gt],
                   } 


        out={'scores':scores,'loss':loss,'logdict':logdict,'lossfun':lossfun }
        return out
    