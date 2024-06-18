'''
Author: lei wang
Date: 2021-07-28 16:07:57
LastEditTime: 2023-03-16 17:53:13
LastEditors: ll
Description:  
无问西东
'''
import lightning.pytorch as pl
from torch.utils.data import  DataLoader
from .MYlib import MYdataset  
from torchvision import transforms
class MYDS(pl.LightningDataModule):
    def __init__(self, aug=0,img_size=224,dataroot='/home/ll/datasets/',seednum=0,train_batchsize=2,val_batchsize=2,test_batchsize=2,num_workers=0,
    dataname=None,datasize=None,eta=1,pre=None):
        super().__init__()
        self.aug=aug
        self.img_size=img_size
        self.dataroot=dataroot
        self.seednum=seednum
        self.save_hyperparameters()
    def setup(self, stage=None):
        train_transform = transforms.Compose([
            # transforms.RandomHorizontalFlip(),
            # transforms.RandomRotation(10),
            # transforms.RandomCrop(size=self.cfg.img_size ),

            transforms.Resize(size=self.img_size),
            transforms.RandomCrop(size=self.img_size ),
            # transforms.RandomResizedCrop(size=self.cfg.img_size,scale=(1.0, 1.0)   ),
            # transforms.ToTensor(),
            
        ])
        test_transform = transforms.Compose([
            # transforms.RandomRotation(20),
            # transforms.RandomHorizontalFlip(),
            transforms.Resize(size=self.img_size),
            transforms.CenterCrop(size=self.img_size ),
            # transforms.RandomResizedCrop(size=self.cfg.img_size,scale=(1.0, 1.0)   ),
            # transforms.ToTensor(),
            
        ])
        self.mydataset_train = MYdataset(self.dataroot,"train",self.seednum, transform=train_transform,eta=self.hparams.eta,pre=self.hparams.pre)
        self.mydataset_val = MYdataset(self.dataroot,"val",self.seednum, transform=test_transform,eta=self.hparams.eta,pre=self.hparams.pre)
        self.mydataset_test = MYdataset(self.dataroot,"test",self.seednum, transform=test_transform,eta=self.hparams.eta,pre=self.hparams.pre)
    def train_dataloader(self):
        
        return DataLoader(self.mydataset_train, batch_size=self.hparams.train_batchsize,num_workers=self.hparams.num_workers,shuffle=True,drop_last=False,pin_memory=True,persistent_workers=True)
    def val_dataloader(self):
        
        return DataLoader(self.mydataset_val, batch_size=self.hparams.val_batchsize,num_workers=self.hparams.num_workers,shuffle=False,drop_last=False,pin_memory=True,persistent_workers=True)
    def test_dataloader(self):
        
        return DataLoader(self.mydataset_test, batch_size=self.hparams.test_batchsize,num_workers=self.hparams.num_workers,shuffle=False,drop_last=False,pin_memory=True,persistent_workers=True)
    def predict_dataloader(self):
        
        return DataLoader(self.mydataset_test, batch_size=self.hparams.test_batchsize,num_workers=self.hparams.num_workers,shuffle=False,drop_last=False,pin_memory=True,persistent_workers=True)