'''
Author: ll
LastEditTime: 2023-05-26 21:37:41
LastEditors: ll
无问西东
'''
import torch 
import numpy as np 
import random 

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
# ssh-keygen -t rsa -C "475594230@qq.com"
#wget https://repo.anaconda.com/archive/Anaconda3-2021.11-Linux-x86_64.sh
#  ssh-keyscan -t rsa github.com >> C:\Users\admin/.ssh/known_hosts\
def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
#      torch.backends.cudnn.benchmark = False

def tensor2numpy():
     x=x.permute(1,2,0).cpu().detach().numpy()