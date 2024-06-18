'''
Author: ll
LastEditTime: 2024-06-18 11:04:23
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
# 创建一个默认字典，其默认值是列表


# 动态地添加元素到指定的索引位置
# my_dict[0].append('apple')  # 添加 'apple' 到索引 0
# my_dict[2].append('banana')  # 添加 'banana' 到索引 2
# my_dict[5].append('cherry')  # 添加 'cherry' 到索引 5

# # 打印字典内容
# print(my_dict)




class GDBC(LightningModule):
    def __init__(self,encoder:nn.Module ,datasize,es=20   ,ysigma=0.01,interval=1,snum=3,ema=0.9,sth=0.02,debias=False):
        super().__init__()         
        self.net = encoder
        self.save_hyperparameters(ignore=['encoder'])
        self.momentum = ema 
        self.register_buffer('soft_labels',torch.zeros(datasize))
        
        self.register_buffer('cu_scores',torch.zeros(datasize))
        self.register_buffer('cu_y',torch.zeros(datasize))
        self.register_buffer('gt',torch.zeros(datasize))

        # self.register_buffer('bsigma',torch.zeros(snum,datasize))
        self.register_buffer('bsigma',torch.full((snum,datasize), float('inf')))
# inf_matrix = 
        # self.register_buffer('cu_scores',defaultdict(list)) 
        # self.register_buffer('soft_labels',defaultdict(list)) 
        # self.register_buffer('bsigma',defaultdict(list)) 
        # self.bsigma[0::2]=1         
        self.bananas = []
        # self.gce=GCELoss()
        # self.sce=SCELoss()
    def forward(self, batch): 

        x, t,index,gt = batch['images'],batch['moss'] ,batch['indexs']  ,batch['mossgt']

        

            
        if self.training:
            batch['moss']=t-self.soft_labels[index]


        out=self.net(batch)   
        scores=out['scores'] 
        # loss =out['loss']  
        # if self.training:
        #     self.cu_scores[index].append(scores.detach().cpu().numpy()) 

        if self.training and self.hparams.debias==True:   
            b_0=t-scores
            # self.bsigma[self.current_epoch%self.hparams.snum,index]=b_0.detach()     
            # if self.hparams.savedata==True:
            #     self.cu_scores[index]=scores.detach()
            #     self.cu_y[index]=t.detach()
            

            self.bsigma[0:self.hparams.snum-1,index]=self.bsigma[1:self.hparams.snum,index]
            self.bsigma[self.hparams.snum-1,index]=b_0.detach()     

            # inf_mask = torch.isinf(self.bsigma[:,index])

            sigma=self.bsigma[:,index].norm( p=1,dim=0)/self.hparams.snum
            b=b_0            
            
            momentum=self.momentum

            
            # if self.current_epoch>=self.hparams.snum-1:

                # b=torch.where(sigma>self.hparams.sth*(1-np.exp( (self.current_epoch-self.trainer.max_epochs)/10.0 )), b_0 ,self.soft_labels[index]) 
            b=torch.where((sigma>self.hparams.sth) & ~torch.isinf (sigma) , b_0 ,self.soft_labels[index]) 
                
            # else :
            #     b=self.soft_labels[index]
            self.soft_labels[index] = momentum *  self.soft_labels[index].detach() + (1 - momentum) * b.detach()
            # bias_target=t-self.soft_labels[index]
            
            loss =out['lossfun'] (target=t-self.soft_labels[index] )   

            etarget=t-self.soft_labels[index]

        else:
            loss =out['lossfun'] (target=t  )  
            etarget=t

        if torch.isnan(loss):
            loss=torch.zeros_like(loss)
            print('skip nan loss')
        logdict = {'mse_score':[scores,gt] ,  
                   'rmse_score':[scores,gt],  
                   'mean_loss':[loss.detach() ],
                   'srcc':[scores,gt],
                   'plcc':[scores,gt],
                   'krcc':[scores,gt],
                   'mse_scorecalibrate':[etarget,gt] ,  
                   } 


        return scores,loss,logdict 





    

    # def Gema(self, x,t,index,batch_idx):
    #     scores=self.model(x) 
    #     scores=scores.squeeze(1)  
    #     b_0=t-scores
    #     # self.bsigma[self.current_epoch%self.hparams.snum,index]=b_0.detach()     
    #     if self.hparams.savedata==True:
    #         self.cu_scores[index]=scores.detach()
    #         self.cu_y[index]=t.detach()
        

    #     self.bsigma[0:self.hparams.snum-1,index]=self.bsigma[1:self.hparams.snum,index]
    #     self.bsigma[self.hparams.snum-1,index]=b_0.detach()     
    #     sigma=self.bsigma[:,index].norm( p=1,dim=0)/self.hparams.snum
    #     b=b_0            
        
    #     momentum=self.momentum
    #     if self.current_epoch>=self.hparams.snum-1:

    #         # b=torch.where(sigma>self.hparams.sth*(1-np.exp( (self.current_epoch-self.trainer.max_epochs)/10.0 )), b_0 ,self.soft_labels[index]) 
    #         b=torch.where(sigma>self.hparams.sth, b_0 ,self.soft_labels[index]) 
            
    #     else :
    #         b=self.soft_labels[index]
    #     self.soft_labels[index] = momentum *  self.soft_labels[index].detach() + (1 - momentum) * b.detach()
    #     bias_target=t-self.soft_labels[index]
    #     loss = F.mse_loss(scores,bias_target.detach())        
    #     return scores,loss     