import torch  
# from tensorly.tucker_tensor import tucker_to_tensor 
# from tensorly.decomposition import tucker
# import tensorly as tl
import torch.nn as nn
from utilplot.plottensor import plottensor
# torch.fft.fft
# torch.autograd.grad(weight.mean(), self.core,
#                                 retain_graph=False, create_graph=False)[0]
import torch.nn.functional as F
import random
import torchvision 
import glo as glo
from torch.nn import  init 
from .kaiming import denoising_block
from .util.net.layers.layers import LWTA

import math 



glo._init()
glo.set_value('adbeta', False)
glo.set_value('valori', False)

# tl.set_backend('pytorch')




class Conv2dpiece(torch.nn.Conv2d): 
    def __init__( self, conv ,beta ,T ) :
        for k,v in conv.__dict__.items():
            setattr(self,k,v)         
        self.betastweight=nn.Parameter(torch.tensor([0.0500,0.100,0.900,1.00,1.8])*0.1)  
        # self.betastweight=nn.Parameter(torch.arange(0.05,2,0.3)*0.1) 
        # self.betastweight=nn.Parameter(torch.tensor([0.0500,0.100,0.900])*0.1)  
        # self.betastweight=nn.Parameter(torch.tensor([0.200,0.400,0.600,0.80,1.00])*0.1)  
        self.num_tokens=self.betastweight.shape[0]
        self.betastweight0=beta
        self.T1=T  
        self.loss=0 
        self.decay=0.99    
    def forward(self, input):
        weight=self.weight.clone()  
        o=self._conv_forward( input, weight, self.bias) 
        b=self.betastweight*10.0 
        T=self.T1
        
        # out=F.sigmoid(T*(o-b[0]))+F.sigmoid(T*(o-b[1]))+F.sigmoid(T*(o-b[2]))+F.sigmoid(T*(o-b[3]))+F.sigmoid(T*(o-b[4]))
        # out=out.detach()+o-o.detach()
        quant=-torch.cos(50*(o-torch.pi/4))/40+o   
        out=quant+o-o.detach()
        
        return out


class Conv2dsoftstable(torch.nn.Conv2d): 
    def __init__( self, conv ,beta ,T ) :
        for k,v in conv.__dict__.items():
            setattr(self,k,v)         
        self.betastweight=nn.Parameter(torch.tensor([0.0500,0.100,0.900,1.00,1.8])*0.1)  
        # self.betastweight=nn.Parameter(torch.arange(0.05,2,0.3)*0.1) 
        # self.betastweight=nn.Parameter(torch.tensor([0.0500,0.100,0.900])*0.1)  
        # self.betastweight=nn.Parameter(torch.tensor([0.200,0.400,0.600,0.80,1.00])*0.1)  
        self.num_tokens=self.betastweight.shape[0]
        self.betastweight0=beta
        self.T1=T  
        self.loss=0 
        self.decay=0.99    
    def forward(self, input):
        weight=self.weight.clone()  
        o=self._conv_forward( input, weight, self.bias) 
        b=self.betastweight*10.0 
        T=self.T1
        # with torch.no_grad():
        #     quant=softquant(o,b,T).detach()
        w=4*torch.pi
        x=o
        # x=(-2*torch.sin(10*w*x)+25*torch.sin(8*w*x)-150*torch.sin(6*w*x)+600*torch.sin(4*w*x)-2100*torch.sin(2*w*x)+2520*w*x)/(10240*w)
        # quant=softquant(o,b,T)
        quant=-torch.cos(50*(o-torch.pi/4))/40+o   
        # quant=(-2*torch.sin(10*w*x)+25*torch.sin(8*w*x)-150*torch.sin(6*w*x)+600*torch.sin(4*w*x)-2100*torch.sin(2*w*x)+2520*w*x)/(10240*w)
        
        if self.training==False:  
            
            # out=quant.detach()+o-o.detach()
            # out=quant+o-o.detach()

            # o=-torch.cos(10*(o-torch.pi/4))/10+o 
            out=quant
        else:  
            # out=quant.detach()+o-o.detach()
            # out=quant+o-o.detach()

            # o=-torch.cos(10*(o-torch.pi/4))/10+o 

            out=quant
            # weight=weight.reshape(weight.shape[0],-1)
            # self.loss=F.mse_loss(o,out) 
            # self.loss=(weight.t()@weight-1).mean() 
        return out


# class Conv2dsoftstable(torch.nn.Conv2d): 
#     def __init__( self, conv ,beta ,T ) :
#         for k,v in conv.__dict__.items():
#             setattr(self,k,v)         
#         self.betastweight=nn.Parameter(torch.tensor([0.0500,0.100,0.900,1.00,1.800])*0.1)  
#         # self.betastweight=nn.Parameter(torch.tensor([0.500,1.00,1.50,2.0,2.50])*0.1)  
#         # self.betastweight=nn.Parameter(torch.tensor([0.200,0.400,0.600,0.80,1.00])*0.1)  
#         self.num_tokens=self.betastweight.shape[0]
#         self.betastweight0=beta
#         self.T1=T  
#         self.loss=0 
#         self.decay=0.99    
#         self.weight=nn.Parameter(-torch.log((1 / (((self.weight*self.weight.shape[0]).clamp(-1,1)+1)/2 + 1e-8)) - 1))
#     def forward(self, input): 
#         weight=(F.sigmoid(self.weight)*2-1)/self.weight.shape[0]
#         o=self._conv_forward( input, weight, self.bias) 
#         b=self.betastweight*10.0  
#         if self.training==False:  
#             T=self.T1
#             out=softquant(o,b,T)
#         else: 
#             T=self.T1
#             out=softquant(o,b,T)   
#         weight=weight.reshape(weight.shape[0],-1)
#         a=weight.t()@weight
#         self.loss=0.0

#         return out




# class Conv2dsoftstable(torch.nn.Conv2d): 
#     def __init__( self, conv ,beta ,T ) :
#         for k,v in conv.__dict__.items():
#             setattr(self,k,v)         
#         self.betastweight=nn.Parameter(torch.tensor([0.0500,0.100,0.900,1.00,1.800])*0.1)  
#         # self.betastweight=nn.Parameter(torch.tensor([0.200,0.400,0.600,0.80,1.00])*0.1)  
#         self.num_tokens=self.betastweight.shape[0]
#         self.betastweight0=beta
#         self.T1=T  
#         self.loss=0 
#         self.decay=0.99    
#     def forward(self, input):
#         if self.weight.shape[1]>3  :  
#             weight=self.weight.clone()  
#             o=self._conv_forward( input, weight, self.bias) 
#             b=self.betastweight*10.0 
#             if self.training==False:  
#                 T=self.T1
#                 out=softquant(o,b,T)
#             else: 
#                 T=self.T1
#                 out=softquant(o,b,T) 
#                 weight=weight.reshape(weight.shape[0],-1)
#                 self.loss=F.mse_loss(o,out)
#                 # self.loss=(weight.t()@weight-1).mean()







#         else:
#             weight=self.weight 
#             out=self._conv_forward(input, weight, self.bias) 
#         return out
 

class Conv2dstable(torch.nn.Conv2d): 
    def __init__( self, conv ,beta ,T ) :
        for k,v in conv.__dict__.items():
            setattr(self,k,v)        

        # self.betastweight=nn.Parameter(torch.tensor([-0.07,-0.45,-1.00,-1.400,0.00,0.07,0.45,1.00,1.400]),requires_grad = False) 
        self.betastweight=nn.Parameter(torch.tensor([-0.4,-0.8,-1.2,-1.6,0.00,0.4,0.8,1.2,1.6]),requires_grad = False) 
        self.num_tokens=self.betastweight.shape[0]
        # self.betastweight=nn.Parameter(torch.arange(-2,2,0.1)) 
        self.betastweight0=beta
        self.T1=T
        self.decay=0.99
        self.loss=0 

    def forward(self, input):
        if self.weight.shape[1]>3  :  
            weight=self.weight.clone()  

            o=self._conv_forward( input, weight, self.bias) 
            b=self.betastweight
            if self.training==False:   
                a=o.unsqueeze(0)-b.reshape(-1,1,1,1,1) 
                a=a.detach()
                min_encoding_indices =torch.argmin(a.abs(), dim=0)
                out = b[min_encoding_indices].detach()+o-o.detach() 
            else: 
                a=o.unsqueeze(0)-b.reshape(-1,1,1,1,1) 
                a=a.detach()
                min_encoding_indices = torch.argmin(a.abs(), dim=0)
                out = b[min_encoding_indices].detach()+o-o.detach()   


                encodings = F.one_hot(min_encoding_indices, self.num_tokens)  .reshape(-1,self.num_tokens)
                embed_sum = encodings.transpose(0,1).type_as(o) @ o.reshape(-1,1).clamp(0.01)
                meanb=embed_sum.squeeze(1)/(encodings.sum(dim=0).type_as(o)+1)
                avg=meanb.detach() -b.detach()  

                b.mul_(self.decay).add_(avg, alpha=1 - self.decay)

                
                self.loss=F.mse_loss(o,b[min_encoding_indices].type_as(o).detach())

        else:
            weight=self.weight 


            out=self._conv_forward(input, weight, self.bias) 
        return out


def reluquant(o,b,T):
    # T=1.0
    b=torch.sort(b.abs())[0]
    a=torch.cat([-b[0:1],b])
    n=b.shape[0]-1
    # itemp=Sigmoidback.apply(100*(o.unsqueeze(0)-b[0:-1].reshape(n,1,1,1,1) ) )
    # itemd=Sigmoidback.apply(100*(o.unsqueeze(0)+b[0:-1].reshape(n,1,1,1,1)) )
    itemp=torch.relu(T*(o.unsqueeze(0)-b[0:-1].reshape(n,1,1,1,1) ) )
    itemd=torch.relu(T*(o.unsqueeze(0)+b[0:-1].reshape(n,1,1,1,1)) )

    m=(a[2::]-a[0:-2])/2 
    m=m.reshape(n,1,1,1,1)
    bias= (a[-1]+a[-2])/2 
    c=  torch.clamp(itemp,min=torch.zeros_like(m),max=m)+torch.clamp(itemd,min=torch.zeros_like(m),max=m)
    out=c.sum(dim=0) -bias
    return out 

class Conv2dqrelu(torch.nn.Conv2d): 
    def __init__( self, conv ,beta ,T ) : 


        for k,v in conv.__dict__.items():
            setattr(self,k,v)         

        self.betastweight=nn.Parameter(torch.tensor([0.0500,0.100,0.900,1.00,1.800])*0.1)  
        self.betastweight0=beta
        self.T1=T 

    def forward(self, input):
        if self.weight.shape[1]>3  :  
            weight=self.weight.clone()  

            o=self._conv_forward( input, weight, self.bias) 
            b=self.betastweight*10.0 
            if self.training==False:  
 
                T=self.T1
                out=reluquant(o,b,T)



            else: 
                T=self.T1
                out=reluquant(o,b,T) 
 

        else:
            weight=self.weight 


            out=self._conv_forward(input, weight, self.bias)

        # if self.weight.shape[1]==512 and self.weight.shape[2]==3:
        #     a=input.clone().detach()
            # with torch.no_grad():
            #     grid = torchvision.utils.make_grid(a[0].unsqueeze(1).clamp_(0., 1.))
            #     plottensor(grid   ,'./1.jpg')
        return out

class Conv2dqh(torch.nn.Conv2d): 
    def __init__( self, conv ,beta ) : 
        for k,v in conv.__dict__.items():
            setattr(self,k,v)         
        self.betastweight0=0.1 

    def forward(self, input):
        if self.weight.shape[1]>3  :  
            weight=self.weight.clone()  

            o=self._conv_forward( input, weight, self.bias)  
            beta=self.betastweight0
            if self.training==False:  


                # out=o+(F.relu(o.abs().detach() -beta*torch.exp(-o.abs().detach() )  ) *o.sign().detach() )-o.detach()

                # out=o+(F.relu(o.abs().detach() -beta ) *o.sign().detach() )-o.detach()
                out=o+(torch.round (o.abs()/beta)*beta *o.sign()).detach() -o.detach()  



            else:
                # if torch.randn(1)>self.betastalpha: 
                # out=o+(F.relu(o.abs()-beta*torch.exp(-o.abs()  )  ) *o.sign()).detach() -o.detach()

                # out=o+(F.relu(o.abs()-beta ) *o.sign()).detach() -o.detach()
                out=o+(torch.round (o.abs()/beta)*beta *o.sign()).detach() -o.detach()  
 

        else:
            weight=self.weight 


            out=self._conv_forward(input, weight, self.bias)

        # if self.weight.shape[1]==512 and self.weight.shape[2]==3:
        #     a=input.clone().detach()
            # with torch.no_grad():
            #     grid = torchvision.utils.make_grid(a[0].unsqueeze(1).clamp_(0., 1.))
            #     plottensor(grid   ,'./1.jpg')
        return out
 



class BatchNorm2dlwta(torch.nn.Module): 
    def __init__( self, inp ) :
        super(BatchNorm2dlwta, self).__init__()
        for k,v in inp.__dict__.items():
            setattr(self,k,v)         
        self.lwta=LWTA()
    def forward(self,input) :  

        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            # TODO: if statement only here to tell the jit to skip emitting this when it is None
            if self.num_batches_tracked is not None:  # type: ignore[has-type]
                self.num_batches_tracked = self.num_batches_tracked + 1  # type: ignore[has-type]
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        r"""
        Decide whether the mini-batch stats should be used for normalization rather than the buffers.
        Mini-batch stats are used in training mode, and in eval mode when buffers are None.
        """
        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (self.running_var is None)

        r"""
        Buffers are only updated if they are to be tracked and we are in training mode. Thus they only need to be
        passed when the update should occur (i.e. in training mode when they are tracked), or when buffer stats are
        used for normalization (i.e. in eval mode when buffers are not None).
        """
        out=F.batch_norm(
            input,
            # If buffers are not to be tracked, ensure that they won't be updated
            self.running_mean
            if not self.training or self.track_running_stats
            else None,
            self.running_var if not self.training or self.track_running_stats else None,
            self.weight,
            self.bias,
            bn_training,
            exponential_average_factor,
            self.eps,
        ) 
        out=self.lwta(out)
        return out

class Conv2dlwta(torch.nn.Conv2d): 
    def __init__( self, conv ,beta,alpha=0.0) : 
        for k,v in conv.__dict__.items():
            setattr(self,k,v)       
        self.lwta=LWTA()
    def forward(self, input):
        weight=self.weight 
        if self.weight.shape[1]>3  :   
            out=self._conv_forward(input, weight, self.bias) 
            out=self.lwta(out)
        else:
        
            out=self._conv_forward(input, weight, self.bias) 
        return out



class Sigmoidback(torch.autograd.Function): 

    @staticmethod
    def forward(ctx, o, ): 
        out=torch.sigmoid(o)
        return out

    @staticmethod
    def backward(ctx, grad_outputImg): 
        grad_input  = None
 
        if ctx.needs_input_grad[0]:
            grad_input = grad_outputImg.sign()
        return grad_input

def softquant(o,b,T):
    
    b=torch.sort(b.abs())[0]
    a=torch.cat([-b[0:1],b])
    n=b.shape[0]-1
    # itemp=Sigmoidback.apply(100*(o.unsqueeze(0)-b[0:-1].reshape(n,1,1,1,1) ) )
    # itemd=Sigmoidback.apply(100*(o.unsqueeze(0)+b[0:-1].reshape(n,1,1,1,1)) )
    itemp=torch.sigmoid(T*(o.unsqueeze(0)-b[0:-1].reshape(n,1,1,1,1) ) )
    itemd=torch.sigmoid(T*(o.unsqueeze(0)+b[0:-1].reshape(n,1,1,1,1)) )

    m=(a[2::]-a[0:-2])/2 
    m=m.reshape(n,1,1,1,1)
    bias= (a[-1]+a[-2])/2 
    c=  m*itemp+m*itemd
    out=c.sum(dim=0) -bias
    return out 

class Roundback(torch.autograd.Function): 

    @staticmethod
    def forward(ctx, o, beta):
        ctx.save_for_backward(o,beta)
        out=torch.round(o.abs().detach() /beta.detach()) *beta.detach() *o.sign().detach() 
        
        # b=torch.sort(b.abs())[0]
        # a=torch.cat([-b[0],b])
        # c=( ((a[2::]-a[0:-2])/2 )*F.sigmoid(10*(o-b[0:-1]))+F.sigmoid(10*(o+b[0:-1])) -(b[-1]+b[-2]) /2 ).sum()
        return out

    @staticmethod
    def backward(ctx, grad_outputImg):
        o,beta = ctx.saved_tensors
        grad_input = grad_weight = None
 
        if ctx.needs_input_grad[0]:
            grad_input = grad_outputImg
        if ctx.needs_input_grad[1]:
            # beta=beta+0.01
            # out0=torch.round(o.abs().detach() /beta.detach()) *beta.detach() *o.sign().detach() 
            # beta=beta-0.02
            # out1=torch.round(o.abs().detach() /beta.detach()) *beta.detach() *o.sign().detach() 
            grad_weight = grad_outputImg 
        return grad_input,grad_weight

class Conv2dq(torch.nn.Conv2d): 
    def __init__( self, conv ,beta ,T ) :
        # super(Conv2dsttexpo, self).__init__()


        for k,v in conv.__dict__.items():
            setattr(self,k,v)        
        # self._conv_forward=conv._conv_forward
 
        # self.betastweight=nn.Parameter(init.normal_(torch.empty(self.weight.shape[0]),0,2)) 
        # self.betastweight=nn.Parameter( beta*torch.ones(self.weight.shape[0]) *0.01) 

        # self.betastweight=nn.Parameter(init.normal_(torch.empty(10),0,1)) 
        # self.betastweight=nn.Parameter(torch.arange(0,2,0.4)) 
        # self.betastweight=nn.Parameter(torch.tensor([0.0500,0.100,0.900,1.300,1.700])*0.1) 
        # self.betastweight=nn.Parameter(init.normal_(torch.empty(5),0,1)*0.1) 

        # self.betastweight=nn.Parameter(torch.tensor([0.0500,0.100,0.900,1.00,1.800])*0.1) 
        # self.betastweight=nn.Parameter(torch.tensor([0.2,0.4,0.6,0.8,1.0])*0.1,requires_grad = True)  
        # self.betastweight=nn.Parameter(torch.tensor([0.00,10.00])*0.1) 

        # self.betastweight=nn.Parameter(torch.rand(1)*6-3) 
        # self.betastweight=nn.Parameter(init.normal_(torch.empty(1),0,1)) 
        self.betastweight=beta
        self.T1=T
        # self.T0=1.0
        # self.n=0.0
        # self.betastalpha=alpha  

    def forward(self, input):
        if self.weight.shape[1]>3  :  
            weight=self.weight.clone() 
            # a=self.betastweight *100 
            # beta=a.abs()+1e-3

            o=self._conv_forward( input, weight, self.bias) 
            # b=self.betastweight*10.0
            beta=0.1
            # beta=beta.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(o)   
            if self.training==False:  


                # out=o+(F.relu(o.abs().detach() -beta*torch.exp(-o.abs().detach() )  ) *o.sign().detach() )-o.detach()

                # out=o+(F.relu(o.abs().detach() -beta ) *o.sign().detach() )-o.detach()
                out=o+(torch.round (o.abs().detach() /beta)*beta *o.sign().detach() )-o.detach() 
                # out=Roundback.apply(o,beta)
                # indices = argsort(x,dim)
                # y = gather(x,indices,dim)
                # indices=torch.sort(b.abs())[1]
                # T=(1-math.exp(-self.n*0.002))*self.T1+math.exp(-self.n*0.002)*self.T0
                # T=self.T1
                # out=softquant(o,b,T)



            else:
                # if torch.randn(1)>self.betastalpha: 
                # out=o+(F.relu(o.abs()-beta*torch.exp(-o.abs()  )  ) *o.sign()).detach() -o.detach()

                # out=o+(F.relu(o.abs()-beta ) *o.sign()).detach() -o.detach()
                out=o+(torch.round (o.abs()/beta)*beta *o.sign()).detach() -o.detach()
                # out=Roundback.apply(o,beta)
                # self.n=self.n+1
                # T=(1-math.exp(-self.n*0.002))*self.T1+math.exp(-self.n*0.002)*self.T0
                # T=self.T1
                # out=softquant(o,b,T)
                # else:
                #     out=o
                
                # out=self._conv_forward( input, weight, self.bias) 

            # corenew=self.core-a.detach()
            # weight=tl.tucker_to_tensor((  corenew, list(self.tucker_factors))) 
            # with torch.no_grad():
            #     core, tucker_factors = tucker(input, rank=[input.shape[0],input.shape[1]//2,input.shape[2],input.shape[3]], init='random', tol=10e-6, random_state=0)
            #     inputnew=tl.tucker_to_tensor((  core, list(tucker_factors))) 
 

        else:
            weight=self.weight 


            out=self._conv_forward(input, weight, self.bias)

        # if self.weight.shape[1]==512 and self.weight.shape[2]==3:
        #     a=input.clone().detach()
            # with torch.no_grad():
            #     grid = torchvision.utils.make_grid(a[0].unsqueeze(1).clamp_(0., 1.))
            #     plottensor(grid   ,'./1.jpg')
        return out
 

class Conv2dzc(torch.nn.Module): 
    def __init__( self, conv ) :
        super(Conv2dzc, self).__init__()
        for k,v in conv.__dict__.items():
            setattr(self,k,v)        
        

    def forward(self, input): 

        weight=self.weight.clone() 
        out=self._conv_forward(input, weight, self.bias)
         
        return out

class Conv2dsttexpo(torch.nn.Conv2d): 
    def __init__( self, conv ,beta,alpha=0.0) :
        # super(Conv2dsttexpo, self).__init__()


        for k,v in conv.__dict__.items():
            setattr(self,k,v)        
        # self._conv_forward=conv._conv_forward
 
        # self.betastweight=nn.Parameter(init.normal_(torch.empty(self.weight.shape[0]),0,1)) 
        self.betastweight=nn.Parameter(init.normal_(torch.empty(1),0,1)) 
        # self.betastweight=nn.Parameter(torch.rand(1)*6-3) 
        self.betastweight0=beta
        self.betastalpha=alpha
    def forward(self, input):
        if self.weight.shape[1]>3  :  
            weight=self.weight.clone() 
            # beta=self.betastweight0
            o=self._conv_forward( input, weight, self.bias)  
            a=self.betastweight
            beta=(F.tanh(a)+1)*self.betastweight0/2
            beta=beta.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(o)  
            # out=o+(F.relu(o.abs()-beta*torch.exp(-o.abs())  ) *o.sign()).detach() -o.detach()
            # out=o+(F.relu(o.abs()-beta  ) *o.sign()).detach() -o.detach() 
            if self.training==False: 
                out=o+(F.relu(o.abs()-beta  ) *o.sign()).detach() -o.detach()
                # out=o
            else:
                if torch.randn(1)>self.betastalpha:
                # out= F.relu(o.abs().detach()-beta ) *o.sign().detach()-o.detach()+o 
                    out=(F.relu(o.abs().detach()-beta ) *o.sign().detach()-o.detach())+o 
                else:
                    out=o
            # out=o+(torch.round (o.abs()/beta)*beta *o.sign()).detach() -o.detach() 
            # out=o+(F.relu(o.abs()-beta*torch.exp(-o.abs())   ) *o.sign()).detach() -o.detach()
        else:
            weight=self.weight 
            out=self._conv_forward(input, weight, self.bias)
        return out
class Conv2dsttexpoo(torch.nn.Module): 
    def __init__( self, conv ,beta) :
        super(Conv2dsttexpoo, self).__init__()
        self=conv  
        # self.betastweight=nn.Parameter(init.normal_(torch.empty(self.weight.shape[0]),0,1)) 
        self.betastweight=nn.Parameter(init.normal_(torch.empty(1),0,100)) 
        # self.betastweight=nn.Parameter(torch.rand(1)*6-3) 
        self.betastweight0=beta
    def forward(self, input):
        if self.weight.shape[1]>3  :  
            weight=self.weight.clone() 
            # beta=self.betastweight0
            o=self._conv_forward( input, weight, self.bias)  
            a=self.betastweight
            beta=(F.tanh(a)+1)*self.betastweight0/2
            beta=beta.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(o)  
            # out=o+(F.relu(o.abs()-beta*torch.exp(-o.abs())  ) *o.sign()).detach() -o.detach()
            # out=o+(F.relu(o.abs()-beta  ) *o.sign()).detach() -o.detach() 
            if self.training==False: 
                out=o+(F.relu(o.abs()-beta  ) *o.sign()).detach() -o.detach()
            else:
                out=o+F.relu(o.abs().detach()-beta ) *o.sign().detach() -o.detach()
            # out=o+(torch.round (o.abs()/beta)*beta *o.sign()).detach() -o.detach() 
            # out=o+(F.relu(o.abs()-beta*torch.exp(-o.abs())   ) *o.sign()).detach() -o.detach()
        else:
            weight=self.weight 
            out=self._conv_forward(input, weight, self.bias)
        return out


class Conv2dsttexp(torch.nn.Module): 
    def __init__( self, conv ,beta) :
        super(Conv2dsttexp, self).__init__()
        self=conv  

        self.betastweight0=beta
        self.betastweight=nn.Parameter(init.normal_(torch.empty(1),0,1)) 
    def forward(self, input):
        if self.weight.shape[1]>3  :  
            weight=self.weight.clone()
            if self.training==False: 

                valori = glo.get_value('valori')
                if valori==False:
                
                    o=self._conv_forward( input, weight, self.bias)  
                    a=self.betastweight
                    beta=(F.tanh(a)+1)*self.betastweight0/2
                    beta=beta.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(o)  
                    # out=o+(F.relu(o.abs()-beta*torch.exp(-o.abs())  ) *o.sign()).detach() -o.detach()
                    out=o+(F.relu(o.abs()-beta  ) *o.sign()).detach() -o.detach()
                else: 
                
                    out=self._conv_forward( input, weight, self.bias) 
            else:
                # beta=self.betastweight
                adbeta = glo.get_value('adbeta')
                if adbeta==False:
                    out=self._conv_forward( input, weight, self.bias)  

                else: 
                    o=self._conv_forward( input, weight.detach(), self.bias)  
                    a=self.betastweight
                    beta=(F.tanh(a)+1)*self.betastweight0/2
                    beta=beta.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(o)  
                    # out=o+F.relu(o.abs().detach()-beta*torch.exp(-o.abs())  ) *o.sign().detach() -o.detach()
                    out=o+F.relu(o.abs().detach()-beta ) *o.sign().detach() -o.detach()
        else:
            weight=self.weight 
            out=self._conv_forward(input, weight, self.bias)
        return out

class Conv2dstta(torch.nn.Module): 
    def __init__( self, conv ,beta) :
        super(Conv2dstta, self).__init__()
        self=conv  

        self.betastweight0=beta
        self.betastweight=nn.Parameter(init.normal_(torch.empty(1),0,1)) 
        # self.betastbias=nn.Parameter(init.normal_(torch.empty(1),0,1)) 
        # self.betastweight=nn.Parameter(torch.ones(1)*0.001) 

        # selffth=nn.Linear(self.weight.shape[0],self.weight.shape[0])
        # self.betastweight=nn.Parameter( torch.ones(1)*0.015) 
    def forward(self, input):
        if self.weight.shape[1]>3  : 
            
            # beta=F.relu(self.betastweight)
            weight=self.weight.clone()
            if self.training==False: 

                valori = glo.get_value('valori')
                if valori==False:
                
                    o=self._conv_forward( input, weight, self.bias) 
                    # a,_=torch.topk(o.flatten(1).abs(),(o.shape[1]*o.shape[2]*o.shape[3])//2,dim=1,largest=False)
                    # a=a.mean(dim=1)
                    # a=a*self.betastweight
                    a=self.betastweight
                    beta=(F.tanh(a)+1)*self.betastweight0/2
                    beta=beta.unsqueeze(1).unsqueeze(2).unsqueeze(3).expand_as(o) 
                    # out=o+(F.relu(o.abs()-beta) *o.sign()).detach() -o.detach()
                    out=o+(F.relu(o.abs()-beta   ) *o.sign()).detach() -o.detach()
                else: 
                
                    out=self._conv_forward( input, weight, self.bias) 
            else:
                # beta=self.betastweight
                adbeta = glo.get_value('adbeta')
                if adbeta==False:
                    out=self._conv_forward( input, weight, self.bias) 
                    # o=self._conv_forward( input, weight.detach(), self.bias) 
                    # out=o+F.relu(o.abs().detach()-beta) *o.sign().detach() -o.detach()

                else: 
                    o=self._conv_forward( input, weight.detach(), self.bias) 

                    # a,_=torch.topk(o.flatten(1).abs(),(o.shape[1]*o.shape[2]*o.shape[3])//2,dim=1,largest=False)
                    # a=a.mean(dim=1)
                    # a=a*self.betastweight
                    a=self.betastweight
                    beta=(F.tanh(a)+1)*self.betastweight0/2
                    beta=beta.unsqueeze(1).unsqueeze(2).unsqueeze(3).expand_as(o) 
                    # out=o+F.relu(o.abs().detach()-beta) *o.sign().detach() -o.detach()
                    out=o+F.relu(o.abs().detach()-beta  ) *o.sign().detach() -o.detach()
        else:
            weight=self.weight 
            out=self._conv_forward(input, weight, self.bias)
        return out

class Conv2dstt(torch.nn.Module): 
    def __init__( self, conv ,beta) :
        super(Conv2dstt, self).__init__()
        self=conv  

        self.betastweight0=beta
        self.betastweight=nn.Parameter(init.normal_(torch.empty(1),0,1)) 
        # self.betastweight=nn.Parameter( torch.ones(1)*0.015) 
    def forward(self, input):
        if self.weight.shape[1]>3  : 
            beta=F.sigmoid(self.betastweight)*self.betastweight0
            # beta=F.relu(self.betastweight)
            weight=self.weight.clone()
            if self.training==False: 
                
                o=self._conv_forward( input, weight, self.bias) 
                out=o+(F.relu(o.abs()-beta) *o.sign()).detach() -o.detach()
            else:
                # beta=self.betastweight
                adbeta = glo.get_value('adbeta')
                if adbeta==False:
                    out=self._conv_forward( input, weight, self.bias) 
                    # o=self._conv_forward( input, weight.detach(), self.bias) 
                    # out=o+F.relu(o.abs().detach()-beta) *o.sign().detach() -o.detach()

                else: 
                    o=self._conv_forward( input, weight.detach(), self.bias) 
                    out=o+F.relu(o.abs().detach()-beta) *o.sign().detach() -o.detach()
        else:
            weight=self.weight 
            out=self._conv_forward(input, weight, self.bias)
        return out

class Conv2dst(torch.nn.Module): 
    def __init__( self, conv ,beta) :
        super(Conv2dst, self).__init__()
        self=conv  
        self.betastweight=beta 
    def forward(self, input):
        if self.weight.shape[1]>3  : 
            weight=self.weight.clone()
            if self.training==False: 
                beta=self.betastweight
                o=self._conv_forward( input, weight, self.bias) 
                out=o+(F.relu(o.abs()-beta) *o.sign()).detach() -o.detach()
            else:
                out=self._conv_forward(input, weight, self.bias)
        else:
            weight=self.weight 
            out=self._conv_forward(input, weight, self.bias)
        return out

class Conv2dkm(torch.nn.Module): 
    def __init__( self, conv ,beta) :
        super(Conv2dkm, self).__init__()
        self=conv  
        self.betastweight=beta 
        self.denoising_block1 = denoising_block(in_planes=self.weight.shape[1], ksize=3, filter_type='Gaussian_Filter')
    def forward(self, input):
        weight=self.weight
        if self.weight.shape[1]>3  : 
            out = self.denoising_block1(input)
            out=self._conv_forward(out, weight, self.bias)
        else:
            weight=self.weight 
            out=self._conv_forward(input, weight, self.bias)
        return out

class Conv2dr(torch.nn.Module): 
    def __init__( self, conv ,beta) :
        super(Conv2dr, self).__init__()
        self=conv  
        self.betastweight=beta
        # self.betastweight= nn.Parameter(beta)
        # core=self.weight
        # self.core=nn.Parameter(core)
        # self.embed = nn.Embedding(self.weight.shape[0],self.weight.shape[0])

        # tucker_rank=[self.weight.shape[0]//2+1,self.weight.shape[0]//2+1 if self.weight.shape[0]//2+1>2 else 2,self.weight.shape[-1],self.weight.shape[-1]  ]

        # if self.weight.shape[-1]>1:
        #     with torch.no_grad():
        #         core, tucker_factors = tucker(self.weight, rank=tucker_rank, init='random', tol=10e-6, random_state=0)
        #     self.tucker_factors=nn.ParameterList( [nn.Parameter(i) for i in tucker_factors])
        #     self.core=nn.Parameter(core)

    def forward(self, input):
        if self.weight.shape[1]>3  : 
            # weight=tl.tucker_to_tensor((  self.core, list(self.tucker_factors))) 
            weight=self.weight.clone()
            # a=self.core.clone().detach()
            # index=random.sample(range(weight.shape[1]),int(weight.shape[1]*0.6))
            # index=random.sample(range(weight.shape[1]),weight.shape[1]//2)
            # index=random.sample(range(weight.shape[1]),weight.shape[1]//2)
            
            if self.training==False: 
                beta=self.betastweight
                # out=self._conv_forward( input+(torch.round (input.abs()/beta)*beta *input.sign()-input).detach(), weight, self.bias) 

                # input=input+(torch.round (input.abs()/beta)*beta *input.sign()).detach() -input.detach()
                # out=self._conv_forward( input, weight, self.bias) 


                o=self._conv_forward( input, weight, self.bias) 
                out=o+(torch.round (o.abs()/beta)*beta *o.sign()).detach() -o.detach()

                # out=o+(torch.where(o.abs()<beta,torch.zeros_like(o).type_as(o),o.abs()) *o.sign() ).detach()-o.detach()

                # (o.abs()).histc(min=1e-7,max=0.5,bins=400)
                # if self.weight.shape[1]==1024:
                #     print(' ')

                # w=self.weight.clone()
                # a=w.sum(dim=0,keepdim=True)
                # an=a/torch.norm(a,p=2,dim=1,keepdim=True)

                # wo=((weight*an).sum(dim=1,keepdim=True) * an )
                # wp=weight- ((weight*an).sum(dim=1,keepdim=True) * an ) 

                # op=self._conv_forward( input, wp, self.bias) 
                # op_q=torch.round (op.abs()/beta)*beta *op.sign()  
                # oth=self._conv_forward( input,  wo, self.bias) 
                # oth_q=torch.round (oth.abs()/beta)*beta *oth.sign()   

                # o=self._conv_forward( input, weight, self.bias) 
                # out=o-oth.detach()+ oth_q-op.detach()+op_q



            else:
                # weight[:,index,:,:]=0
                beta=self.betastweight
                o=self._conv_forward( input, weight, self.bias) 
                out=o+(torch.round (o.abs()/beta)*beta *o.sign()).detach() -o.detach()
                
                # out=self._conv_forward( input, weight, self.bias) 

            # corenew=self.core-a.detach()
            # weight=tl.tucker_to_tensor((  corenew, list(self.tucker_factors))) 
            # with torch.no_grad():
            #     core, tucker_factors = tucker(input, rank=[input.shape[0],input.shape[1]//2,input.shape[2],input.shape[3]], init='random', tol=10e-6, random_state=0)
            #     inputnew=tl.tucker_to_tensor((  core, list(tucker_factors))) 
 

        else:
            weight=self.weight 


            out=self._conv_forward(input, weight, self.bias)

        # if self.weight.shape[1]==512 and self.weight.shape[2]==3:
        #     a=input.clone().detach()
            # with torch.no_grad():
            #     grid = torchvision.utils.make_grid(a[0].unsqueeze(1).clamp_(0., 1.))
            #     plottensor(grid   ,'./1.jpg')
        return out


# class Conv2dr(torch.nn.Module): 
#     def __init__( self, conv ) :
#         super(Conv2dr, self).__init__()
#         self=conv  
#         core=self.weight
#         self.core=nn.Parameter(core)

#         # tucker_rank=[self.weight.shape[0]//2+1,self.weight.shape[0]//2+1 if self.weight.shape[0]//2+1>2 else 2,self.weight.shape[-1],self.weight.shape[-1]  ]

#         # if self.weight.shape[-1]>1:
#         #     with torch.no_grad():
#         #         core, tucker_factors = tucker(self.weight, rank=tucker_rank, init='random', tol=10e-6, random_state=0)
#         #     self.tucker_factors=nn.ParameterList( [nn.Parameter(i) for i in tucker_factors])
#         #     self.core=nn.Parameter(core)

#     def forward(self, input):
#         if self.weight.shape[1]>3 and self.weight.shape[-1]>1 : 
#             # weight=tl.tucker_to_tensor((  self.core, list(self.tucker_factors))) 
#             weight=self.weight.clone()
#             # a=self.core.clone().detach()
#             index=random.sample(range(weight.shape[1]),int(weight.shape[1]*0.6))
#             # index=random.sample(range(weight.shape[1]),weight.shape[1]//2)
#             # index=random.sample(range(weight.shape[1]),weight.shape[1]//2)
            
#             if self.training==False:
#                 # weight=weight*index.__len__() /weight.shape[1]
#                 weight=weight*1.5
#                 # weight[:,index,:,:]=0
#             else:
#                 # weight[:,index,:,:]=0
#                 pass
#             # corenew=self.core-a.detach()
#             # weight=tl.tucker_to_tensor((  corenew, list(self.tucker_factors))) 
#             # with torch.no_grad():
#             #     core, tucker_factors = tucker(input, rank=[input.shape[0],input.shape[1]//2,input.shape[2],input.shape[3]], init='random', tol=10e-6, random_state=0)
#             #     inputnew=tl.tucker_to_tensor((  core, list(tucker_factors))) 

#             inputnew=input

#         else:
#             weight=self.weight
#             inputnew=input


#         out=self._conv_forward(inputnew, weight, self.bias)

#         if self.weight.shape[1]==512 and self.weight.shape[2]==3:
#             a=input.clone().detach()
#             # with torch.no_grad():
#             #     grid = torchvision.utils.make_grid(a[0].unsqueeze(1).clamp_(0., 1.))
#             #     plottensor(grid   ,'./1.jpg')
#         return out

# class Conv2dr0(torch.nn.Module): 
#     def __init__( self, conv ) :
#         super(Conv2dr0, self).__init__()
#         self=conv  
#         # tucker_rank=[self.weight.shape[0]//2+1,self.weight.shape[0]//2+1 if self.weight.shape[0]//2+1>2 else 2,self.weight.shape[-1],self.weight.shape[-1]  ]
#         # if self.weight.shape[-1]>1:
#         #     with torch.no_grad():
#         #         core, tucker_factors = tucker(self.weight, rank=tucker_rank, init='random', tol=10e-6, random_state=0)
#         #     self.tucker_factors=nn.ParameterList( [nn.Parameter(i) for i in tucker_factors])
#         #     self.core=nn.Parameter(core)
#     def forward(self, input):
#         if self.weight.shape[1]>3 and self.weight.shape[-1]>1 : 
#             # weight=tl.tucker_to_tensor((  self.core, list(self.tucker_factors))) 
#             weight=self.weight.clone()
#             # a=self.core.clone().detach()
#             index=random.sample(range(weight.shape[1]),weight.shape[1]//2)
#             # index=random.sample(range(weight.shape[1]),weight.shape[1]//2)
#             # index=random.sample(range(weight.shape[1]),weight.shape[1]//2)
            
#             if self.training==False:
#                 # weight=weight*index.__len__() /weight.shape[1]
#                 weight[:,index,:,:]=0
#             else:
#                 weight[:,index,:,:]=0
#             # corenew=self.core-a.detach()
#             # weight=tl.tucker_to_tensor((  corenew, list(self.tucker_factors))) 
#             # with torch.no_grad():
#             #     core, tucker_factors = tucker(input, rank=[input.shape[0],input.shape[1]//2,input.shape[2],input.shape[3]], init='random', tol=10e-6, random_state=0)
#             #     inputnew=tl.tucker_to_tensor((  core, list(tucker_factors))) 

#             inputnew=input

#         else:
#             weight=self.weight
#             inputnew=input


#         out=self._conv_forward(inputnew, weight, self.bias)

#         if self.weight.shape[1]==512 and self.weight.shape[2]==3:
#             a=input.clone().detach()
#             with torch.no_grad():
#                 grid = torchvision.utils.make_grid(a[0].unsqueeze(1).clamp_(0., 1.))
#                 plottensor(grid   ,'./1.jpg')
#         return out




# class BatchNorm2dr(torch.nn.Module): 
#     def __init__( self, inp ) :
#         super(BatchNorm2dr, self).__init__()
#         for k,v in inp.__dict__.items():
#             setattr(self,k,v)        
#         self.track_running_stats=False
#         self.affine=False
#     def forward(self,input) :  

#         if self.momentum is None:
#             exponential_average_factor = 0.0
#         else:
#             exponential_average_factor = self.momentum

#         if self.training and self.track_running_stats:
#             # TODO: if statement only here to tell the jit to skip emitting this when it is None
#             if self.num_batches_tracked is not None:  # type: ignore[has-type]
#                 self.num_batches_tracked = self.num_batches_tracked + 1  # type: ignore[has-type]
#                 if self.momentum is None:  # use cumulative moving average
#                     exponential_average_factor = 1.0 / float(self.num_batches_tracked)
#                 else:  # use exponential moving average
#                     exponential_average_factor = self.momentum

#         r"""
#         Decide whether the mini-batch stats should be used for normalization rather than the buffers.
#         Mini-batch stats are used in training mode, and in eval mode when buffers are None.
#         """
#         if self.training:
#             bn_training = True
#         else:
#             bn_training = (self.running_mean is None) and (self.running_var is None)

#         r"""
#         Buffers are only updated if they are to be tracked and we are in training mode. Thus they only need to be
#         passed when the update should occur (i.e. in training mode when they are tracked), or when buffer stats are
#         used for normalization (i.e. in eval mode when buffers are not None).
#         """
#         return F.batch_norm(
#             input,
#             # If buffers are not to be tracked, ensure that they won't be updated
#             self.running_mean
#             if not self.training or self.track_running_stats
#             else None,
#             self.running_var if not self.training or self.track_running_stats else None,
#             self.weight,
#             self.bias,
#             bn_training,
#             exponential_average_factor,
#             self.eps,
#         ) 