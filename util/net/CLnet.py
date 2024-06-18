'''
Author: ll
LastEditTime: 2024-06-15 04:25:08
LastEditors: ll
无问西东
'''
# -*- coding: utf-8 -*-
# @Time    : 2021/8/9 下午3:35
# @Author  : ruima
# @File    : network.py
# @Software: PyCharm

import torch
import torch.nn as nn
from torchvision import models
import torch
import torch.nn as nn
import torch.nn.functional as F    

from torchvision import transforms as T 
import torchvision
from functools import partial
from collections import defaultdict
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
import glob 
from lightning import LightningModule
def weight_init(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
            m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()


class ModifiedResNext101(nn.Module):
    """ModifiedResNext101."""

    def __init__(self):
        super().__init__()
        resnet = models.resnext101_32x8d(pretrained=True)

        # Create the shared feature generator.
        self.shared = nn.Sequential()
        for name, module in resnet.named_children():
            if name != 'fc':
                self.shared.add_module(name, module)

        self.shared.add_module(name='conv_n1', module=nn.Conv2d(2048, 1024, kernel_size=1))
        self.shared.add_module(name='bn_n1', module=nn.BatchNorm2d(1024))
        self.shared.add_module(name='relu', module=nn.ReLU(inplace=True))
        self.shared.add_module(name='conv_n2', module=nn.Conv2d(1024, 1, kernel_size=1))

    def train_nobn(self, mode=True):
        """Override the default module train."""
        super(ModifiedResNext101, self).train(mode)

        # Set the BNs to eval mode so that the running means and averages do not update.
        for module in self.shared.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()

    def forward(self, x):

        x = self.shared(x)
        x = x.view(x.size(0), -1)
        x = nn.Sigmoid()(x)
        x = x.view(-1)
        return x


def init_dump(arch_name):
    """Dumps pretrained model in required format."""
    model = ModifiedResNext101()
    previous_masks = {}
    for module_idx, module in enumerate(model.shared.modules()):
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            mask = torch.ByteTensor(module.weight.data.size()).fill_(1)
            if 'cuda' in module.weight.data.type():
                mask = mask.cuda()
            previous_masks[module_idx] = mask
    torch.save({
        'dataset2idx': {'imagenet': 1},
        'previous_masks': previous_masks,
        'model': model,
    }, './imagenet/{}.pt'.format(arch_name))



class CustomConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super(CustomConv2d, self).__init__(*args, **kwargs) 
        self.prunerate=0.0001
        self.register_buffer('mask',torch.zeros_like(self.weight.data))
    def updatemask(self):
        mask=torch.where(self.weight.data.abs()>self.weight.data.abs().flatten().kthvalue(1+round(self.prunerate*self.weight.data.numel())).values,1.0,0.0).type_as(self.weight.data)
        self.mask=self.mask+mask
        self.weight.data[self.mask==0] = 0.0 

    def forward(self, input):
        

        return super(CustomConv2d, self).forward(input)
    
    def backward(self, grad_output): 
        if self.mask==0:
            pass
        else:
            grad_output[self.mask==0]=0
        return grad_output
class RRNet(LightningModule):
    def __init__(self,checkpointpath=None,prunerate=0.001,finetuned=False,testphase=False): 
        super().__init__()
        resnet = models.resnet50(pretrained=True)

        # Create the shared feature generator.
        self.shared = nn.Sequential()
        for name, module in resnet.named_children():
            if name != 'fc':
                self.shared.add_module(name, module)

        self.shared.add_module(name='conv_n1', module=nn.Conv2d(2048, 1024, kernel_size=1))
        # self.shared.add_module(name='bn_n1', module=nn.BatchNorm2d(1024))
        self.shared.add_module(name='relu', module=nn.ReLU(inplace=True))
        self.shared.add_module(name='conv_n2', module=nn.Conv2d(1024, 1, kernel_size=1)) 
        self.prunerate=prunerate
        self.finetuned=finetuned
        self.testphase=testphase



        def replace_conv2d(module):
            for name, child in module.named_children():
                if isinstance(child, nn.Conv2d):
                    new_conv = CustomConv2d(child.in_channels, child.out_channels,
                                            child.kernel_size, child.stride,
                                            child.padding, child.dilation,
                                            child.groups, child.bias is not None,
                                            child.padding_mode)
                    new_conv.weight = child.weight
                    if child.bias is not None:
                        new_conv.bias = child.bias
                    setattr(module, name, new_conv)
                else:
                    replace_conv2d(child)

        replace_conv2d(self)


        if checkpointpath!=None:

            checkpointpath=glob.glob(checkpointpath)    [0]
            item=torch.load(checkpointpath,map_location='cpu')['state_dict']
            state_dict = self.state_dict()

            for k, v in item.items():  
                if 'model.net.' in k:
                    k = k.replace('model.net.', '')
                    if  'prunerate'  in k or   'finetuned' in k  :
                        continue 
                    state_dict.update({k: v})
            # self.load_state_dict(state_dict) 
            self.load_state_dict(state_dict,strict=True) 
            if self.testphase==False:
                self.updata_conv2d(self)

        for module in self.shared.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()

    def updata_conv2d(self,module):
        for name, child in module.named_children():
            if isinstance(child, CustomConv2d):   
                child.prunerate=self.prunerate
                child.updatemask()
            else:
                self.updata_conv2d(child)


    # def prune(self):
    #     if self.prunerate==0.00001:
    #         self.prunerate=0.75
    #         return 0 
        
    #     for module_idx, module in enumerate(self.shared.modules()):
    #         if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
    #             mask=torch.where(module.weight.data.abs()>module.weight.data.abs().flatten().kthvalue(1+round(self.prunerate*module.weight.data.numel())).values,1.0,0.0) 
    #             self.mask[str(module_idx)]=self.mask[str(module_idx)]+mask 
    #             module.weight.data[self.mask[str(module_idx)]==0] = 0.0
    #             module.weight[self.mask[str(module_idx)]==1].requires_grad=False
        
 
    def forward(self, batch):

        x, t,index,gt = batch['images'],batch['moss'] ,batch['indexs']  ,batch['mossgt']

        x=T.Resize(320)(x)  
        if not self.training:
            x=T.CenterCrop(320)(x)  

        
        # x = (x - torch.tensor(IMAGENET_DEFAULT_MEAN).reshape(1,3,1,1).to(x)  ) / torch.tensor(IMAGENET_DEFAULT_STD).reshape(1,3,1,1).to(x)
        if self.finetuned==False and self.current_epoch==20:
            self.updata_conv2d(self)
            self.finetuned=True
        x = (x - torch.tensor(IMAGENET_DEFAULT_MEAN).reshape(1,3,1,1).to(x)  ) / torch.tensor(IMAGENET_DEFAULT_STD).reshape(1,3,1,1).to(x)
        x = self.shared(x)
        x = x.view(x.size(0), -1)
        x = nn.Sigmoid()(x)
        x = x.view(-1)
        scores=x

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






class GradientFilterFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight):
        ctx.save_for_backward(input, weight)
        return input.clone()

    @staticmethod
    def backward(ctx, grad_output):
        input,mask = ctx.saved_tensors

        grad_output[mask.repeat(grad_output.shape[0],1,1,1)]=0
        return grad_output, None   


# class CustomConv2dFunction(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
#         # 存储用于反向传播的信息
#         ctx.save_for_backward(input, weight, bias)
#         ctx.stride = stride
#         ctx.padding = padding
#         ctx.dilation = dilation
#         ctx.groups = groups

#         # 调用内置的conv2d前向运算
#         output = F.conv2d(input, weight, bias, stride, padding, dilation, groups)
#         return output

#     @staticmethod
#     def backward(ctx, grad_output):
#         input, weight, bias = ctx.saved_tensors
#         grad_input = grad_weight = grad_bias = None

#         # 这里可以添加自定义梯度计算逻辑
#         if ctx.needs_input_grad[0]:
#             grad_input = F.grad.conv2d_input(input.shape, weight, grad_output, ctx.stride, ctx.padding, ctx.dilation, ctx.groups)
#         if ctx.needs_input_grad[1]:
#             grad_weight = F.grad.conv2d_weight(input, weight.shape, grad_output, ctx.stride, ctx.padding, ctx.dilation, ctx.groups)

#         if bias is not None and ctx.needs_input_grad[2]:
#             grad_bias = grad_output.sum(dim=(0, 2, 3))

class SILFConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super(SILFConv2d, self).__init__(*args, **kwargs) 
        self.prunerate=0.0001
        self.register_buffer('mask',torch.zeros_like(self.weight.data))
        self.taskid=0 
        self.weight.register_hook(self.gradient_clipping_hook)
        if self.bias!=None:
            self.bias.register_hook(self.gradient_clipping_hook1)
    def gradient_clipping_hook1( self,grad):
        grad=torch.zeros_like(grad)
        # print(0)
        return grad
    def gradient_clipping_hook( self,grad):
        # grad[self.mask!=self.taskid]=0
        grad=torch.where(self.mask==self.taskid,grad,0.0).type_as(grad)
        # print(1)
        return grad

    def forward(self, input):
        return super(SILFConv2d, self).forward(input) 

    def updatemask(self):
        mask=torch.where(self.weight.data.abs()<self.weight.data.abs().flatten().kthvalue(1+round(self.prunerate*self.weight.data.numel())).values,1.0,0.0).type_as(self.weight.data)
        # index=(mask==1 )& (mask==self.taskid)
        # index=
        self.mask[(mask==1 )& (self.mask==self.taskid)]=self.taskid+1 

        # mask=torch.where(self.weight.data.abs()<self.weight.data.abs().flatten().kthvalue(1+round(self.prunerate/2*self.weight.data.numel())).values,1.0,0.0).type_as(self.weight.data)
        # # index=(mask==1 )& (mask==self.taskid)
        # # index= 

        self.weight.data[(mask==1 )& (self.mask==self.taskid)] = 0.0 

    # def forward(self, x):
    #     # 将权重也传递给自定义的梯度函数
    #     # if self.mask.shape[0]==1:
    #     #     print(0)
    #     # x=CustomConv2dFunction.apply(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    #     x=super(SILFConv2d, self).forward(x) 
    #     # x=GradientFilterFunction.apply(x, self.mask!=self.taskid)
    #     return x
    
    # def backward(self, grad_output): 
    #     # if self.mask==0:
    #     #     pass
    #     # else:
    #     grad_output[self.mask!=self.taskid]=0
    #     return grad_output
     
import torch.nn.utils.prune as prune
class SILF(LightningModule):
    def __init__(self,checkpointpath=None,prunerate=0.001,finetuned=False,testphase=False,taskid=0): 
        super().__init__()
        resnet = models.resnet50(pretrained=True)

        # Create the shared feature generator.
        self.shared = nn.Sequential()
        for name, module in resnet.named_children():
            if name != 'fc':
                self.shared.add_module(name, module)

        self.shared.add_module(name='conv_n1', module=nn.Conv2d(2048, 1, kernel_size=1)) 
        self.prunerate={0:0.3,1:0.05,2:0.01,3:0.0}[taskid]
        self.finetuned=finetuned
        self.testphase=testphase
        self.taskid=taskid



        def replace_conv2d(module):
            for name, child in module.named_children():
                if isinstance(child, nn.Conv2d):
                    new_conv = SILFConv2d(child.in_channels, child.out_channels,
                                            child.kernel_size, child.stride,
                                            child.padding, child.dilation,
                                            child.groups, child.bias is not None,
                                            child.padding_mode)
                    new_conv.weight.data = child.weight.data
                    new_conv.taskid=taskid
                    # new_conv.weight.register_hook(gradient_clipping_hook)
                    if child.bias is not None:
                        new_conv.bias.data = child.bias.data
                    setattr(module, name, new_conv) 
                else:
                    replace_conv2d(child)

        replace_conv2d(self)


        if checkpointpath!=None:

            checkpointpath=glob.glob(checkpointpath)    [0]
            item=torch.load(checkpointpath,map_location='cpu')['state_dict']
            state_dict = self.state_dict()

            for k, v in item.items():  
                if 'model.net.' in k:
                    k = k.replace('model.net.', '')
                    if  'taskid'  in k or   'finetuned' in k  or   'prunerate' in k :
                        continue 
                    state_dict.update({k: v})
            # self.load_state_dict(state_dict) 
            self.load_state_dict(state_dict,strict=True)  


        for module in self.shared.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
                module.weight.requires_grad_(False)
                module.bias.requires_grad_(False)
    def updata_conv2d(self,module):
        for name, child in module.named_children():
            if isinstance(child, SILFConv2d):   
                child.prunerate=self.prunerate
                child.taskid=self.taskid
                child.updatemask()
            else:
                self.updata_conv2d(child)


    # def prune(self):
    #     if self.prunerate==0.00001:
    #         self.prunerate=0.75
    #         return 0 
        
    #     for module_idx, module in enumerate(self.shared.modules()):
    #         if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
    #             mask=torch.where(module.weight.data.abs()>module.weight.data.abs().flatten().kthvalue(1+round(self.prunerate*module.weight.data.numel())).values,1.0,0.0) 
    #             self.mask[str(module_idx)]=self.mask[str(module_idx)]+mask 
    #             module.weight.data[self.mask[str(module_idx)]==0] = 0.0
    #             module.weight[self.mask[str(module_idx)]==1].requires_grad=False
        
 
    def forward(self, batch):

        x, t,index,gt = batch['images'],batch['moss'] ,batch['indexs']  ,batch['mossgt']

        x=T.Resize(320)(x)  
        if not self.training:
            x=T.CenterCrop(320)(x)  

        
        # x = (x - torch.tensor(IMAGENET_DEFAULT_MEAN).reshape(1,3,1,1).to(x)  ) / torch.tensor(IMAGENET_DEFAULT_STD).reshape(1,3,1,1).to(x)
        if self.finetuned==False and self.current_epoch==20:
            self.updata_conv2d(self)
            self.finetuned=True
        x = (x - torch.tensor(IMAGENET_DEFAULT_MEAN).reshape(1,3,1,1).to(x)  ) / torch.tensor(IMAGENET_DEFAULT_STD).reshape(1,3,1,1).to(x)
        x = self.shared(x)
        x = x.view(x.size(0), -1)
        x = nn.Sigmoid()(x)
        x = x.view(-1)
        scores=x

        loss=F.l1_loss(scores,t)
        lossfun=partial(F.l1_loss,scores)

        logdict = {'mse_score':[scores,gt] ,  
                   'rmse_score':[scores,gt],  
                   'mean_loss':[torch.zeros_like(loss),loss.detach()],
                   'srcc':[scores,gt],
                   'plcc':[scores,gt],
                   } 


        out={'scores':scores,'loss':loss,'logdict':logdict,'lossfun':lossfun}
        # out={'scores':scores,'loss':loss,'logdict':logdict}
        return out



if __name__ == '__main__':
    arch_name = 'ModifiedResNext101'
    init_dump(arch_name)