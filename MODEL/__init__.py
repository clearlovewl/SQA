'''
Author: ll
LastEditTime: 2024-04-12 17:24:25
LastEditors: ll
无问西东
'''
from lightning.pytorch.callbacks import TQDMProgressBar 

# class LitProgressBar(TQDMProgressBar):
#     def init_validation_tqdm(self):
#         bar = super().init_validation_tqdm()
#         bar.set_description("running validation...")
#     def init_train_tqdm(self)  :
#         bar = super().init_train_tqdm() 
#         bar.leave=False
#         bar.nrows=2

#         return bar

