import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

n_branch = 3  

# Performing PGD on main branch loss
def pgd_main(x, preds, loss_fn, y=None, eps=None, model=None, steps=3, gamma=None, randinit=False, **kwargs):
    x_adv = x.clone()
    if randinit:
        x_adv += (2.0 * torch.rand(x_adv.shape) - 1.0) * eps

    x_adv = Variable(x_adv.cuda(), requires_grad=True)
    x = x.cuda()

    for t in range(steps):
        out_adv_branch = model(x_adv) # take main branch loss out
        loss_adv0 = loss_fn(out_adv_branch[-1], y)
        grad0 = torch.autograd.grad(loss_adv0, x_adv, only_inputs=True)[0]
        x_adv.data.add_(gamma * torch.sign(grad0.data))
        linfball_proj(x, eps, x_adv, in_place=True)
        x_adv = torch.clamp(x_adv, 0, 1)

    return x_adv


# Performing PGD on k-th branch loss
def pgd_k(x, preds, loss_fn, y=None, eps=None, model=None, steps=3, gamma=None, randinit=False, **kwargs):
    k = int(kwargs['branch_num'])

    x_adv = x.clone()
    if randinit:
        x_adv += (2.0 * torch.rand(x_adv.shape) - 1.0) * eps
    x_adv = Variable(x_adv.cuda(), requires_grad=True)
    x = x.cuda()
    for t in range(steps):
        out_adv_branch = model(x_adv)
        out = out_adv_branch[k] # take k-th branch loss out
        loss_adv0 = loss_fn(out, y)
        grad0 = torch.autograd.grad(loss_adv0, x_adv, only_inputs=True)[0]
        x_adv.data.add_(gamma * torch.sign(grad0.data))
        linfball_proj(x, eps, x_adv, in_place=True)
        x_adv = torch.clamp(x_adv, 0, 1)
        
    return x_adv


# Performing PGD on k-th branch loss
def pgd_avg(x, preds, loss_fn, y=None, eps=None, model=None, steps=3, gamma=None, randinit=False, **kwargs):
    x_adv = x.clone()
    if randinit:
        x_adv += (2.0 * torch.rand(x_adv.shape) - 1.0) * eps
    x_adv = Variable(x_adv.cuda(), requires_grad=True)
    x = x.cuda()

    for t in range(steps):
        out_adv_branch = model(x_adv)
        loss_adv0 = 0
        for i in range(n_branch): # average the loss of each branch
             loss_adv0 += loss_fn(out_adv_branch[i], y) * (1.0/len(out_adv_branch))
        grad0 = torch.autograd.grad(loss_adv0, x_adv, only_inputs=True)[0]
        x_adv.data.add_(gamma * torch.sign(grad0.data))
        linfball_proj(x, eps, x_adv, in_place=True)
        x_adv = torch.clamp(x_adv, 0, 1)
    return x_adv



# Performing PGD on max-avg loss
def pgd_max(x, preds, loss_fn, y=None, eps=None, model=None, steps=3, gamma=None, randinit=False, **kwargs):

    x_advs = [x.clone() for _ in range(n_branch)] 
    if randinit:
        x_advs = [(x_adv+(2.0 * torch.rand(x_adv.shape) - 1.0) * eps) for x_adv in x_advs]        
        x_advs = [Variable(x_adv.cuda(), requires_grad=True) for x_adv in x_advs]        

    # Generate Adv samples for each branch
    x = x.cuda()
    for t in range(steps):
        for i in range(n_branch):
            x_adv = x_advs[i]
            out_adv_branch = model(x_adv)
            out = out_adv_branch[i]
            loss_adv0 = loss_fn(out, y)
            grad0 = torch.autograd.grad(loss_adv0, x_adv, only_inputs=True)[0]
            x_adv.data.add_(gamma * torch.sign(grad0.data))
            linfball_proj(x, eps, x_adv, in_place=True)
            x_adv = torch.clamp(x_adv, 0, 1)
            x_advs[i] = x_adv

    # Record average losses for each adv samples
    losses = []
    for i in range(n_branch):
        x_adv = x_advs[i]
        out_adv_branch = model(x_adv)

        for j in range(n_branch):  # calculate average loss
            out = out_adv_branch[j]
            if j == 0:
                loss_adv0 = torch.nn.functional.cross_entropy(input=out, target=y, reduce=False)
            else:
                loss_adv0 += torch.nn.functional.cross_entropy(input=out, target=y, reduce=False)
        losses.append(loss_adv0)


    ## select the adv sample by referencing average losses
    losses =  torch.stack(losses, dim=-1)
    x_advs =  torch.stack(x_advs, dim=1)
    _, idxs = losses.topk(1,dim=-1)
    idxs = idxs.long().view(-1,1)
    idxs =  idxs.unsqueeze(2).unsqueeze(3).unsqueeze(4).repeat(1, 1, 1, 28, 28) # flatten index based on mnist image
    x_adv = torch.gather(x_advs, 1, idxs).squeeze(1)
    return x_adv



def tensor_clamp(t, min, max, in_place=True):
    if not in_place:
        res = t.clone()
    else:
        res = t
    idx = res.data < min
    res.data[idx] = min[idx]
    idx = res.data > max
    res.data[idx] = max[idx]

    return res

def linfball_proj(center, radius, t, in_place=True):
    return tensor_clamp(t, min=center - radius, max=center + radius, in_place=in_place)

