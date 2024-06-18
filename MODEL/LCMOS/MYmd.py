import torch 
from torch.nn import functional as F
from lightning import LightningModule
from torchmetrics import SpearmanCorrCoef ,PearsonCorrCoef,PeakSignalNoiseRatio,MeanMetric,MeanSquaredError,KendallRankCorrCoef
import torch.nn as nn
import re
import os
from collections import defaultdict
from init import setup_seed

import copy 
import numpy as  np
import torch.autograd as autograd
from diffusers import StableDiffusionPipeline
from torchvision.models.resnet import resnet50  

from functools import partial
import time
import glob 
from copy import deepcopy

class MYMD(LightningModule):
    def __init__(self, model:nn.Module ,denoise=1,checkpointpath=None,train_batchsize=0,val_batchsize=0,test_batchsize=0 ):
        
    # def __init__(self, model:nn.Module ,denoise=1,checkpointpath=None,dataname=None,
    # train_batchsize=0,val_batchsize=0,test_batchsize=0,sigma=1.0,beta=None,test_sigma=[1.0,2.0],esp=1.0,predictmode=0,alpha=0.1,trainmode='clean',T=None,stablebeta=None,attack='fgsm',valuelayer='l0',predictsigma=2.0,ood='noood',p=0.2,savedata=False
    # ):
        super().__init__()
        self.save_hyperparameters(ignore=['model'])

        
        self.model=model 
        self.my_device = torch.device("cuda:0")
        self.out={}

    def setup(self,stage): 
        # setup_seed(20)

        if self.hparams.checkpointpath!=None:    
            checkpointpath=glob.glob(self.hparams.checkpointpath)    [0]
            item=torch.load(checkpointpath,map_location='cpu')['state_dict']
            state_dict = self.state_dict()

            for k, v in item.items():  
                if  'soft_labels' in k or 'cu_scores' in k or 'cu_y' in k or 'gt' in k or 'bsigma' in k  :
                    continue
                
                state_dict.update({k: v})
            # self.load_state_dict(state_dict) 
            self.load_state_dict(state_dict,strict=True) 

            if self.model.net._get_name()=="LWF":
                self.model.net.model_old=deepcopy(self.model.net.net)
                self.model.net.model_old.eval()            
                self.model.net.distll=True  

        # a=self.trainer.datamodule.val_dataloader()  
        # self.dataset=a.dataset
        # self.cu_y=torch.tensor(a.dataset.cu_y).type_as(self.cu_y)
        # self.gt=torch.tensor(a.dataset.gt)
        # self.preds=torch.tensor(a.dataset.gt)

    def forward(self, x):        
        scores=self.model(x) 
        # scores=scores.squeeze(1)     
        return scores  

    # def forwardnonoise(self, x,t,index,batch_idx,batch=None):        
    #     scores=self.model(x) 
    #     scores=scores.squeeze(1)    
    #     loss=F.mse_loss(scores,t)

    #     return scores,loss   
    def forwardnonoise(self,  batch_idx,batch=None):    
        # x= [x,batch['imagesref']] 
        start_epoch = time.time()
        scores,loss,logdict=self.model(batch) 

 
        end_epoch = time.time()

        elapsed = end_epoch - start_epoch 
        logdict['mean_time']=torch.ones(1,1).to(self.device)*elapsed
        # if (ori.mean(dim=[1,2,3])==enhance.mean(dim=[1,2,3]) ).all():
        #     loss_enhance=torch.zeros_like(loss_enhance)

        return scores,loss  ,logdict
    def forwarddifboost(self, x,t,index,batch_idx,batch=None):        
        scores=self.model(x) 
        scores=scores.squeeze(1)    
        loss=F.mse_loss(scores,t)

        return scores,loss   
    
    def back(self,loss,opt):
        if torch.is_grad_enabled() : 
            self.manual_backward(loss)
            opt.step() 
            opt.zero_grad(set_to_none=True) 




    def logs(self,logdict,dataloader_idx=''): 
        prefix= 'test_' if self.training==False else 'train_' 

        logdict_idx={ prefix+i+ str(dataloader_idx): logdict[i] for i in  logdict.keys()}

        if not hasattr(self,list(logdict_idx.keys())[0]):
            metric={"srcc":SpearmanCorrCoef,"mse":MeanSquaredError,"mean": MeanMetric,'plcc':PearsonCorrCoef,'rmse': partial(MeanSquaredError,squared=False),'krcc':KendallRankCorrCoef}
            for i in logdict_idx:
                setattr(self,i,metric [ list (filter(lambda x: x in i,metric.keys()))[-1]]().to(self.device) )
                # setattr(self,"test_"+i,metric [i.split('_')[0] ]().to(self.device)  ) 


        # if not hasattr(self,list(logdict.keys())[0]):
        #     metric={"srcc":SpearmanCorrCoef,"mse":MeanSquaredError,"mean": MeanMetric,'plcc':PearsonCorrCoef,'rmse': partial(MeanSquaredError,squared=False),'krcc':KendallRankCorrCoef}
        #     for i in logdict:
        #         setattr(self,"train_"+i,metric [ list (filter(lambda x: x in i,metric.keys()))[0]]().to(self.device) )
        #         setattr(self,"test_"+i,metric [ list (filter(lambda x: x in i,metric.keys()))[0]]().to(self.device) ) 

        for i in logdict_idx: 
            getattr(self,i)(* logdict_idx[i] )
            self.log(i,  getattr(self,i) , on_epoch=True,prog_bar=  any ([j in i for j in ["srcc","mse"] ] )  ,metric_attribute=i)

    def training_step(self, batch, batch_idx,dataloader_idx=''): 
        batch['taskidx']=0 if dataloader_idx=='' else int(dataloader_idx) 
        # opt = self.optimizers()    
        scores,loss ,logdict   = self.forwardnonoise(batch_idx,batch=batch)   
        self.logs(logdict,dataloader_idx)
        return loss  


    def validation_step(self, batch, batch_idx,dataloader_idx='' ):

        batch['taskidx']=0 if dataloader_idx=='' else int(dataloader_idx) 


        scores,loss ,logdict   = self.forwardnonoise(batch_idx,batch=batch)   
        self.logs(logdict,dataloader_idx)
        # if self.current_epoch==49:
        #     self.out.setdefault(('preds',dataloader_idx), [ ])  
        #     self.out[('preds',dataloader_idx)].append(scores)
        #     self.out.setdefault(('targets',dataloader_idx) , [ ])  
        #     self.out[('targets',dataloader_idx)].append(batch['mossgt']) 


            
    # def on_validation_epoch_end(self):
    #     if self.current_epoch==49:
    #         for i in self.out:

    #             self.out[i]=torch.cat(self.out[i]) 
    #         torch.save(self.out,os.path.join(self.logger.root_dir,'out.pt'))

    @torch.inference_mode(mode=True)
    def test_step(self, batch, batch_idx,dataloader_idx=''):    
        batch['taskidx']=0 if dataloader_idx=='' else int(dataloader_idx)  
        self.model.eval() 
        scores,loss ,logdict   = self.forwardnonoise(batch_idx,batch=batch)   
        self.logs(logdict,dataloader_idx)
        # self.out.setdefault(('preds',  metric), {})  
        self.out.setdefault(('preds',dataloader_idx), [ ])  
        self.out[('preds',dataloader_idx)].append(scores)
        self.out.setdefault(('targets',dataloader_idx) , [ ])  
        self.out[('targets',dataloader_idx)].append(batch['mossgt']) 


            
    def on_test_epoch_end(self):
        for i in self.out:

            self.out[i]=torch.cat(self.out[i]) 
        torch.save(self.out,os.path.join(self.logger.root_dir,'out.pt'))


