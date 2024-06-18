'''
Author: lei wang
Date: 2021-07-28 16:07:57
<<<<<<< HEAD
<<<<<<< HEAD
LastEditTime: 2024-06-16 11:01:42
=======
LastEditTime: 2024-05-15 22:56:13
>>>>>>> 05f9dba04cea0db09e215d573b3bf4537ae13cf8
=======
LastEditTime: 2024-05-12 15:37:14
>>>>>>> d2707f4514d84592d9f86d0d6d33384b43bab017
LastEditors: ll
Description:  
无问西东
'''
import lightning.pytorch as pl
from torch.utils.data import  DataLoader
from torch.utils.data import Dataset
from torchvision import transforms
from easydict import EasyDict
import importlib
 
import torch
import numpy 
import random 
from lightning.pytorch.utilities import CombinedLoader
from typing import Dict, Any

from omegaconf import OmegaConf

# 读取 YAML 文件


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    numpy.random.seed(worker_seed)
    random.seed(worker_seed)
class MYDS(pl.LightningDataModule):
    def __init__(self,dataset, datasize,    eta= None,iqatask= 'iqa' ,indexseednum= 0  ,cltestdataset=None,testdataset=None,aug=0,img_size=224,dataroot='/home/ll/datasets/',seednum=0,indexseed=0,train_batchsize=2,val_batchsize=2,test_batchsize=2,num_workers=0):
        super().__init__()
        self.aug=aug
        self.img_size=img_size
        self.dataroot=dataroot
        self.seednum=seednum
        self.indexseed=indexseed
        self.dataset=dataset
        o= importlib.import_module('DATA.'+dataset+'.MYlib') 
        
        self.dataset=o.MYdataset

        if cltestdataset:
            self.cltestdataset= {i:importlib.import_module('DATA.'+i+'.MYlib') .MYdataset for i in cltestdataset}
        else:
            self.cltestdataset=None

        if testdataset:
            self.testdataset=importlib.import_module('DATA.'+testdataset+'.MYlib') .MYdataset
        else:
            self.testdataset=None
        self.save_hyperparameters(ignore=['dataset','testdataset','cltestdataset'])
        self.collate_fn=None 
        # self.testimg_size
    def setup(self, stage=None):
        train_transform = transforms.Compose([ 
            
            transforms.Resize(size=self.img_size,antialias=True),
            transforms.CenterCrop(size=self.img_size ), 
            # transforms.RandomCrop(size=self.img_size ), 
            # transforms.RandomHorizontalFlip(), 
            
        ])
        test_transform = transforms.Compose([ 
            # transforms.Resize(size=self.img_size),
            transforms.Resize(size=self.img_size,antialias=True),
            # transforms.CenterCrop(size=self.img_size ), 
            
        ])
        

        if self.cltestdataset!=None:
            self.mydataset_train = self.dataset(phase="train", transform=train_transform,**self.hparams) 
            self.mydataset_val =   {i: self.cltestdataset[i](phase="val", transform=test_transform,** OmegaConf.load(f'DATA/{i}/{i}.yaml')['data']['init_args']   )  for i in self.cltestdataset.keys()}
            self.mydataset_test = {i: self.cltestdataset[i](phase="test", transform=test_transform,** OmegaConf.load(f'DATA/{i}/{i}.yaml')['data']['init_args'] )  for i in self.cltestdataset.keys()}
        elif self.testdataset!=None:
            self.mydataset_train = self.dataset(phase="crosstrain", transform=train_transform,**self.hparams)
            self.mydataset_val = self.testdataset(phase="crossval", transform=test_transform,**self.hparams)
            self.mydataset_test = self.testdataset(phase="crosstest", transform=test_transform,**self.hparams)
        else:
            self.mydataset_train = self.dataset(phase="train", transform=train_transform,**self.hparams)
            self.mydataset_val = self.dataset(phase="val", transform=test_transform,**self.hparams)
            self.mydataset_test = self.dataset(phase="test", transform=test_transform,**self.hparams)   
                      
    def train_dataloader(self):


 
        generator = torch.Generator()
        generator.manual_seed(self.indexseed)

        # return DataLoader(self.mydataset_train, batch_size=self.hparams.train_batchsize,num_workers=self.hparams.num_workers,shuffle=True,pin_memory=True,persistent_workers=False,collate_fn=self.collate_fn,drop_last=True if str(self.dataset).split('.')[1]=='CSIQ' or str(self.dataset).split('.')[1]=='koniqpsp'  else False) 
    
        return DataLoader(self.mydataset_train, batch_size=self.hparams.train_batchsize,num_workers=self.hparams.num_workers,shuffle=True,pin_memory=True,persistent_workers=False,collate_fn=self.collate_fn,drop_last=True,generator=generator ) 

    def val_dataloader(self):

        if self.cltestdataset!=None:
            iterables={ i: DataLoader(self.mydataset_val[i], batch_size=self.hparams.val_batchsize,num_workers=self.hparams.num_workers,shuffle=False,drop_last=False,pin_memory=True,persistent_workers=False,collate_fn=self.collate_fn)  for i in self.mydataset_val.keys()}
            dataloader=CombinedLoader(iterables, 'sequential')
             
        else:
            dataloader=DataLoader(self.mydataset_val, batch_size=self.hparams.val_batchsize,num_workers=self.hparams.num_workers,shuffle=False,drop_last=False,pin_memory=True,persistent_workers=False,collate_fn=self.collate_fn)

        
        return dataloader
    def test_dataloader(self):
        
        if self.cltestdataset!=None:
            iterables={ i: DataLoader(self.mydataset_val[i], batch_size=self.hparams.val_batchsize,num_workers=self.hparams.num_workers,shuffle=False,drop_last=False,pin_memory=True,persistent_workers=False,collate_fn=self.collate_fn)  for i in self.mydataset_val.keys()}
            dataloader=CombinedLoader(iterables, 'sequential')
             
        else:
            dataloader=DataLoader(self.mydataset_val, batch_size=self.hparams.val_batchsize,num_workers=self.hparams.num_workers,shuffle=False,drop_last=False,pin_memory=True,persistent_workers=False,collate_fn=self.collate_fn)

        
        return dataloader
    def predict_dataloader(self):
        
        return DataLoader(self.mydataset_test, batch_size=self.hparams.test_batchsize,num_workers=self.hparams.num_workers,shuffle=False,drop_last=False,pin_memory=True,persistent_workers=False)
    def collate_fn(self,batch):
        r"""
        Collation function for MinkowskiEngine.SparseTensor that creates batched
        cooordinates given a list of dictionaries.
        """
        # collect items
        batch = EasyDict({k: [u[k] for u in batch] for k in batch[0]}) 
        return batch
    
