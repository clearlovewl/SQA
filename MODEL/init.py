'''
Author: ll
LastEditTime: 2024-04-12 22:55:00
LastEditors: ll
无问西东
'''
from lightning.pytorch.callbacks import TQDMProgressBar 
from lightning.pytorch.callbacks import *
# from lightning.pytorch.callbacks import RichProgressBar
class LitProgressBar(TQDMProgressBar):
    def __init__(self, refresh_rate  = 1 ):
        super().__init__()

    def init_validation_tqdm(self):
        bar = super().init_validation_tqdm( )
        bar.set_description("running validation...")
        bar.nrows=3
        bar.leave=True 
 

        return bar
    def init_train_tqdm(self)  :
        bar = super().init_train_tqdm( ) 
        bar.leave=False  
        bar.nrows=3
        return bar

