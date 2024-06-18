'''
Author: lei wang
LastEditTime: 2024-06-18 11:46:20
LastEditors: ll
无问西东
'''

from torch.utils.data import Dataset
import torch 
import numpy as np
import os.path as osp
from io import BytesIO
import random 
from PIL import Image
import time 
import torch.nn.functional as F
import os 
# from util.strlist import delsame
from torchvision import transforms
import scipy.io
import pandas as pd 
# from easydict import EasyDict
import glob
import copy
# from util.P913 import run_alternating_projection as P913
# from util.P910 import run_alternating_projection as P910
# from util.BT500 import run_alternating_projection as BT500
import csv
# from util.mos_to_sureal import convert_file
# from sureal import run_subjective_models ,SubjrejMosModel,P910AnnexEModel,P913124Model,BT500Model
# from sureal.dataset_reader import RawDatasetReader
import scipy.io as scio
from  init import setup_seed
class MYdataset(Dataset):
    def __init__(self, dataroot,phase ,seednum,cfg=None ,transform=None,eta=1,pre=None,iqatask='pre',**kwargs): 

        rate,inten=eta 
        dataroot=os.path.expanduser(dataroot)
        root_dir = os.path.join(dataroot,'koniq10k')  
        data=pd.read_csv(os.path.join(root_dir,'koniq10k_scores_and_distributions.csv') )
        datasplit=pd.read_csv(os.path.join(root_dir,'koniq10k_distributions_sets.csv') )
        split=datasplit['set'].values.tolist()
        mos=((data['MOS'].values-1 )/4.0).tolist();name=data['image_name'].values.tolist();std=data['SD'].values.tolist() 
        rating=list(zip(datasplit['c1'].values.tolist(),datasplit['c2'].values.tolist(),datasplit['c3'].values.tolist(),datasplit['c4'].values.tolist(),datasplit['c5'].values.tolist()   ))
        setup_seed(seednum)           
        self.samples=list(map(lambda x:[torch.tensor( x[0]  ).float() ,    torch.multinomial(torch.tensor(x[4]),int(inten),replacement=True).float()/4.0  if torch.bernoulli(torch.tensor(rate))==1 else  torch.tensor(   x[0] ).float(), x[1],x[2] ,x[3]] ,  zip(mos,name,std,split,rating)   )) 
        self.root_dir=root_dir
        self.transforms = transform
        self.phase=phase         

        if iqatask=='iqadebias' :  
            scores=np.load(r'logs\pspt\0_koniqpsp__1.0_1.0modelsPSPCONTRIQUE_random\lightning_logs\version_0\scores.npy',allow_pickle=True).tolist()['preds'].cpu()  
            for i in range(len(self.samples)):
                self.samples[i][1]=scores[i]    
                 
        if 'iqa' in iqatask:
            if 'cross' not in phase:             
                if phase == 'train':                    
                    self.samples=list(filter(lambda x:x[4]=='training',self.samples))                       
                else :
                    self.samples=list(filter(lambda x:x[4] == 'test'  ,self.samples))


        self.cu_y=[i[1].mean() for i in self.samples ]
        self.gt=[i[0] for i in self.samples ]
    # def __getitem__(self, index): 
    def getit(self,index):
        items=self.samples[index]
        paths =os.path.join(self.root_dir,'imgs',items[2]) 
        images = Image.open(paths).convert('RGB') 
        images=transforms.ToTensor()(images) 
        images = self.transforms(images)
        moss=items[1].mean()
        mossgt=items[0]     
        out={'moss':moss,'images':images,'names':items[0],'indexs':index,'mossgt':mossgt}      
        
        return out 
     

    def __getitem__(self, index):  
        try:
            out=self.getit(index)
        except:
            index=torch.randint(0,self.__len__(),(1,1)).item()
            out=self.__getitem__(index)
        
        return out 

    def __len__(self):
        
        return self.samples.__len__() 