'''
Author: ll
LastEditTime: 2024-06-18 11:01:44
LastEditors: ll
无问西东
''' 
import sys 
import os 
if sys.platform=='win32':
    os.environ['USERPROFILE']='D:/'
import glob 
import pandas as pd
from omegaconf import OmegaConf 
import torch  
import shutil
# os.environ["CUDA_LAUNCH_BLOCKING"]='1'


# os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE" 
  
# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:56327'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:56327' 
# torch.set_float32_matmul_precision('medium')
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
if sys.platform=='linux': 
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'

torch.set_float32_matmul_precision('medium')


from lightning.pytorch import Trainer, seed_everything

seed_everything(42, workers=True)
# sets seeds for numpy, torch and python.random. 

# from pytorch_lightning import LightningDataModule, LightningModule
from lightning.pytorch.core import LightningModule,LightningDataModule
# from pytorch_lightning.utilities.cli import LightningCLI 
from lightning.pytorch.cli import LightningCLI ,SaveConfigCallback
import  lightning.pytorch as pl 
import sys   
import torch
import numpy as np 
import random  
  
import os
 


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':16:8'    
    
    # os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:2'
    # torch.use_deterministic_algorithms(True)

# 设置随机数种子
setup_seed(20) 

class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.link_arguments("data.init_args.datasize", "model.init_args.model.init_args.datasize")  
        # parser.link_arguments("data.init_args.datasize", "model.init_args.param")  
        parser.link_arguments("model.init_args.train_batchsize", "data.init_args.train_batchsize")
        parser.link_arguments("model.init_args.val_batchsize", "data.init_args.val_batchsize")
        parser.link_arguments("model.init_args.test_batchsize", "data.init_args.test_batchsize")
        
def getcli():
    cli = MyLightningCLI(LightningModule, LightningDataModule,  subclass_mode_model=True,subclass_mode_data=True,run=False,parser_kwargs={"parser_mode": "omegaconf"})
    return cli     
def cli_main():
    cli = MyLightningCLI(LightningModule, LightningDataModule,  subclass_mode_model=True,subclass_mode_data=True,run=False,save_config_kwargs={"save_config_overwrite": "True"},parser_kwargs={"parser_mode": "omegaconf"})
    cli.trainer.fit(cli.model, datamodule=cli.datamodule)
    print(0)
def cli_test():
    cli = MyLightningCLI(LightningModule, LightningDataModule,  subclass_mode_model=True,subclass_mode_data=True,run=False,save_config_kwargs={"save_config_overwrite": "True"},parser_kwargs={"parser_mode": "omegaconf"}) 
    cli.trainer.test(cli.model, datamodule=cli.datamodule)  
    out=cli.model.out  
    return out
if __name__ == "__main__": 
    temp=sys.argv[0]
    data='LIVEC'
    model='PSP'

    sys.argv=[temp]+['--config','DATA/'+data+'/'+data+'.yaml']+['--config','DATA/'+data+'/'+model+'.yaml']  
    cli_main()

# git checkout master
# git checkout abc123
# sudo ./cuda_11.8.0_520.61.05_linux.run --silent --driver --override-driver-check
# pip install -r requirements.txt
# import os 
# os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ids 
# huggingface-cli login --token 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
# huggingface-cli login --token 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
# export PATH=/home/ll/doing_project/diffusers/clashcli/clashctl/clashctl-Linux:$PATH

# set http_proxy=http://127.0.0.1:56327
# set https_proxy=http://127.0.0.1:56327
# # 在终端使用代理
# export http_proxy=http://127.0.0.1:7890
# export https_proxy=http://127.0.0.1:7890
# set http_proxy=http://127.0.0.1:7078 
# set https_proxy=http://127.0.0.1:7078 
# # export http_proxy=http://127.0.0.1:7078
# # export https_proxy=http://127.0.0.1:7078

# set http_proxy= 
# set https_proxy= 
# set http_proxy=
# set https_proxy=
# # 测试代理可用
# curl -i google.com
# # 在终端取消代理
# unset http_proxy
# unset https_proxy
# # 更优雅的方式设置/取消终端代理，执行下面两行代码后，可以直接在终端使用 proxy/unproxy 来设置/取消 终端代理
# echo 'alias proxy="export http_proxy=http://127.0.0.1:7890;export https_proxy=http://127.0.0.1:7890' >> ~/.bashrc
# echo 'alias unproxy="unset http_proxy;unset https_proxy"' >> ~/.bashrc


# pip config set global.index-url https://pypi.Python.org/simple/
# pip config set install.trusted-host mirrors.aliyun.com

# set http_proxy=http://127.0.0.1:56327 & set https_proxy=http://127.0.0.1:56327
# sudo dpkg --list | grep nvidia-*
# sudo apt-get --purge remove nvidia*
# sudo apt autoremove
# sudo apt-get --purge remove "cublas" "cuda*"
# sudo apt-get --purge remove "nvidia"
# # 
        # '--model','MODEL.RESNET.MYmd.ImageClassifier',
        # '--data','DATA.MNIST.MYds.MNISTDataModule',
        # '--print_config'



# $ python trainer.py fit --model.help mycode.mymodels.MyModel
# $ python trainer.py fit --model mycode.mymodels.MyModel --print_config    
# python trainer.py fit --print_config > config.yaml        
# sudo chown -R ll anaconda3
# nohup python main.py &
# nohup python main_lr.py >lr_log 2>&1 &
# sudo usermod -d /home/ll -s /bin/bash ll
# git config --global user.name "clearlovewl"
# git config --global user.email "475594230@qq.com"
#wget https://repo.anaconda.com/archive/Anaconda3-2021.11-Linux-x86_64.sh
#  ssh-keyscan -t rsa github.com >> C:\Users\admin/.ssh/known_hosts

# source /usr/local/src/anaconda3/bin/activate
# conda init
# sudo fuser -v /dev/nvidia* |awk '{for(i=1;i<=NF;i++)print "kill -9 " $i;}' | sudo sh