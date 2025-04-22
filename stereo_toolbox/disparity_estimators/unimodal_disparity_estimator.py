import torch


def unimodal_disparity_estimator(x, maxdisp=192):
    disp = torch.arange(maxdisp,dtype=x.dtype,device=x.device).reshape(1,maxdisp,1,1)
    index = torch.argmax(x,1,keepdim=True)
    mask = disp.repeat(x.size(0),1,x.size(2),x.size(3))
    mask2 = torch.arange(maxdisp+1,dtype=x.dtype,device=x.device).reshape([1,maxdisp+1,1,1]).repeat(x.size(0),1,x.size(2),x.size(3))

    x_diff_r = torch.diff(x,dim=1,prepend=torch.ones(x.size(0),1,x.size(2),x.size(3),dtype=x.dtype,device=x.device),\
                        append=torch.ones(x.size(0),1,x.size(2),x.size(3),dtype=x.dtype,device=x.device))
    x_diff_l = torch.diff(x,dim=1,prepend=torch.ones(x.size(0),1,x.size(2),x.size(3),dtype=x.dtype,device=x.device))
    
    index_r = torch.gt(x_diff_r * torch.gt(mask2,index),0).int()
    index_r = torch.argmax(index_r,1,keepdim=True)-1
    
    index_l = torch.lt(x_diff_l * torch.le(mask,index),0).int()
    index_l = (maxdisp-1) - torch.argmax(torch.flip(index_l,[1]),1,keepdim=True)
    
    mask = torch.ge(mask,index_l) * torch.le(mask,index_r)
    x = x * mask.data
    x = x / torch.sum(x,1,keepdim=True)
    
    out = torch.sum(x*disp,1, keepdim=True)
    return out
