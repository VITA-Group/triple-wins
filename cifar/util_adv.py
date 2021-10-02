import numpy as np
import torch
import torch.nn as nn


from torch.autograd import Variable

def cross_entropy(input, target, label_smoothing=0.0, size_average=True):
    """ Cross entropy that accepts soft targets
    Args:
         pred: predictions for neural network
         targets: targets, can be soft
         size_average: if false, sum is returned instead of mean

    Examples::

        input = torch.FloatTensor([[1.1, 2.8, 1.3], [1.1, 2.1, 4.8]])
        input = torch.autograd.Variable(out, requires_grad=True)

        target = torch.FloatTensor([[0.05, 0.9, 0.05], [0.05, 0.05, 0.9]])
        target = torch.autograd.Variable(y1)
        loss = cross_entropy(input, target)
        loss.backward()
    """
    if label_smoothing > 0:
        target = torch.clamp(target, max=1-label_smoothing, min=label_smoothing/9.0)

    logsoftmax = nn.LogSoftmax(dim=1)
    if size_average:
        return torch.mean(torch.sum(-target * logsoftmax(input), dim=1))
    else:
        return torch.sum(torch.sum(-target * logsoftmax(input), dim=1))


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


def l2ball_proj(center, radius, t, in_place=True):
    if not in_place:
        res = t.clone()
    else:
        res = t

    direction = t - center
    dist = direction.view(direction.size(0), -1).norm(p=2, dim=1, keepdim=True)
    direction.view(direction.size(0), -1).div_(dist)
    dist[dist > radius] = radius
    direction.view(direction.size(0), -1).mul_(dist)
    res.data.copy_(center + direction)
    return res


def linfball_proj(center, radius, t, in_place=True):
    return tensor_clamp(t, min=center - radius, max=center + radius, in_place=in_place)


_extra_args = {'alpha', 'steps', 'randinit', 'gamma'}


def ifgm(x, preds, loss_fn, y=None, eps=None, model=None, steps=3, gamma=None, randinit=False, **kwargs):
    if len(kwargs) > 0:
        assert set(kwargs.keys()).issubset(_extra_args)
    if eps is None:
        # eps = 0.07972772183418274
        # eps = 0.45474205
        eps = 0.062
    if gamma is None:
        gamma = (eps * 1.25) / steps
    if y is None:
        # Using model predictions as ground truth to avoid label leaking
        preds_max = preds.data.max(1)[1]
        y = torch.equal(preds, preds_max).float()

    # Compute loss

    x_adv = x.clone()
    if randinit:
        x_adv += (2.0 * torch.rand(x_adv.shape) - 1.0) * eps
    x_adv = Variable(x_adv.cuda(), requires_grad=True)
    x = x.cuda()
    for t in range(steps):
        out_adv = model(x_adv)
        loss_adv0 = loss_fn(out_adv, y)
        grad0 = torch.autograd.grad(loss_adv0, x_adv, only_inputs=True)[0]
        x_adv.data.add_(gamma * torch.sign(grad0.data))
        linfball_proj(x, eps, x_adv, in_place=True)
        x_adv = torch.clamp(x_adv, 0, 1)
        
    return x_adv


def ifgm_k(x, preds, loss_fn, y=None, eps=None, model=None, steps=3, gamma=None, randinit=False, **kwargs):
    k = int(kwargs['branch_num'])
    if eps is None:
        eps = 0.062
    if gamma is None:
        gamma = (eps * 1.25) / steps
    if y is None:
        # Using model predictions as ground truth to avoid label leaking
        preds_max = preds.data.max(1)[1]
        y = torch.equal(preds, preds_max).float()

    # Compute loss

    x_adv = x.clone()
    if randinit:
        x_adv += (2.0 * torch.rand(x_adv.shape) - 1.0) * eps
    x_adv = Variable(x_adv.cuda(), requires_grad=True)
    x = x.cuda()
    for t in range(steps):
        out_adv_branch = model(x_adv)
        out = out_adv_branch[k]
        loss_adv0 = loss_fn(out, y)
        grad0 = torch.autograd.grad(loss_adv0, x_adv, only_inputs=True)[0]
        x_adv.data.add_(gamma * torch.sign(grad0.data))
        linfball_proj(x, eps, x_adv, in_place=True)
        x_adv = torch.clamp(x_adv, 0, 1)

        
    return x_adv


def ifgm_main(x, preds, loss_fn, y=None, eps=None, model=None, steps=3, gamma=None, randinit=False, **kwargs):
    #if len(kwargs) > 0:
    #    assert set(kwargs.keys()).issubset(_extra_args)
    if eps is None:
        eps = 0.062
    if gamma is None:
        gamma = (eps * 1.25) / steps
    if y is None:
        # Using model predictions as ground truth to avoid label leaking
        preds_max = preds.data.max(1)[1]
        y = torch.equal(preds, preds_max).float()

    # Compute loss

    x_adv = x.clone()
    if randinit:
        x_adv += (2.0 * torch.rand(x_adv.shape) - 1.0) * eps
    x_adv = Variable(x_adv.cuda(), requires_grad=True)
    x = x.cuda()
    for t in range(steps):
        out_adv_branch = model(x_adv)
        loss_adv0 = loss_fn(out_adv_branch[-1], y)
        grad0 = torch.autograd.grad(loss_adv0, x_adv, only_inputs=True)[0]
        x_adv.data.add_(gamma * torch.sign(grad0.data))
        linfball_proj(x, eps, x_adv, in_place=True)
        x_adv = torch.clamp(x_adv, 0, 1)
    return x_adv


def ifgm_branchy(x, preds, loss_fn, y=None, eps=None, model=None, steps=3, gamma=None, randinit=False, **kwargs):
    if eps is None:
        # eps = 0.07972772183418274
        # eps = 0.45474205
        eps = 0.062
    if gamma is None:
        gamma = (eps * 1.25) / steps
    if y is None:
        # Using model predictions as ground truth to avoid label leaking
        preds_max = preds.data.max(1)[1]
        y = torch.equal(preds, preds_max).float()

    # Compute loss

    x_adv = x.clone()
    if randinit:
        x_adv += (2.0 * torch.rand(x_adv.shape) - 1.0) * eps
    x_adv = Variable(x_adv.cuda(), requires_grad=True)
    x = x.cuda()
    for t in range(steps):
        out_adv_branch = model(x_adv)
        loss_adv0 = 0
        for i in range(len(out_adv_branch)):
             loss_adv0 += loss_fn(out_adv_branch[i], y) * (1.0/len(out_adv_branch))
        grad0 = torch.autograd.grad(loss_adv0, x_adv, only_inputs=True)[0]
        x_adv.data.add_(gamma * torch.sign(grad0.data))
        linfball_proj(x, eps, x_adv, in_place=True)
        x_adv = torch.clamp(x_adv, 0, 1)
    return x_adv


def ifgm_random(x, preds, loss_fn, y=None, eps=None, model=None, steps=3, gamma=None, randinit=False, **kwargs):
    if eps is None:
        # eps = 0.07972772183418274
        # eps = 0.45474205
        eps = 0.062
    if gamma is None:
        gamma = (eps * 1.25) / steps
    if y is None:
        # Using model predictions as ground truth to avoid label leaking
        preds_max = preds.data.max(1)[1]
        y = torch.equal(preds, preds_max).float()

    # Compute loss

    x_adv = x.clone()
    if randinit:
        x_adv += (2.0 * torch.rand(x_adv.shape) - 1.0) * eps
    x_adv = Variable(x_adv.cuda(), requires_grad=True)
    x = x.cuda()
  
    random_weight = None
    for t in range(steps):
        out_adv_branch = model(x_adv)
        loss_adv0 = 0
        if not isinstance(random_weight, list):
            random_weight = np.random.random(len(out_adv_branch))
            random_weight = list(random_weight / sum(random_weight))
             
        for i in range(len(out_adv_branch)):
             loss_adv0 += random_weight[i] * loss_fn(out_adv_branch[i], y) * (1.0/len(out_adv_branch))
        grad0 = torch.autograd.grad(loss_adv0, x_adv, only_inputs=True)[0]
        x_adv.data.add_(gamma * torch.sign(grad0.data))
        linfball_proj(x, eps, x_adv, in_place=True)
        x_adv = torch.clamp(x_adv, 0, 1)
    return x_adv


def ifgm_max(x, preds, loss_fn, y=None, eps=None, model=None, steps=3, gamma=None, randinit=False, **kwargs):
    if eps is None:
        # eps = 0.07972772183418274
        # eps = 0.45474205
        eps = 0.062
    if gamma is None:
        gamma = (eps * 1.25) / steps
    if y is None:
        # Using model predictions as ground truth to avoid label leaking
        preds_max = preds.data.max(1)[1]
        y = torch.equal(preds, preds_max).float()

    
    # Compute loss
    x_advs = [x.clone() for _ in range(7)] 
    if randinit:
        x_advs = [(x_adv+(2.0 * torch.rand(x_adv.shape) - 1.0) * eps) for x_adv in x_advs]        
        x_advs = [Variable(x_adv.cuda(), requires_grad=True) for x_adv in x_advs]        

    x = x.cuda()
    for t in range(steps):
        for i in range(7):
            x_adv = x_advs[i]
            out_adv_branch = model(x_adv)
            out = out_adv_branch[i]
            loss_adv0 = loss_fn(out, y)
            grad0 = torch.autograd.grad(loss_adv0, x_adv, only_inputs=True)[0]
            x_adv.data.add_(gamma * torch.sign(grad0.data))
            linfball_proj(x, eps, x_adv, in_place=True)
            x_adv = torch.clamp(x_adv, 0, 1)
            x_advs[i] = x_adv


    # determine adv images
    losses = []
    for i in range(7):
        x_adv = x_advs[i]
        out_adv_branch = model(x_adv)
        out = out_adv_branch[i]
        loss_adv0 = torch.nn.functional.cross_entropy(input=out, target=y, reduce=False)
        losses.append(loss_adv0)
    losses =  torch.stack(losses, dim=-1)
    x_advs =  torch.stack(x_advs, dim=1)
    _, idxs = losses.topk(1,dim=-1)
    idxs = idxs.long().view(-1,1)
    idxs =  idxs.unsqueeze(2).unsqueeze(3).unsqueeze(4).repeat(1, 1, 3, 32, 32)
    x_adv = torch.gather(x_advs, 1, idxs).squeeze(1)
    return x_adv

def ifgm_max_v2(x, preds, loss_fn, y=None, eps=None, model=None, steps=3, gamma=None, randinit=False, **kwargs):
    #if len(kwargs) > 0:
    #    assert set(kwargs.keys()).issubset(_extra_args)
    if eps is None:
        # eps = 0.07972772183418274
        # eps = 0.45474205
        eps = 0.062
    if gamma is None:
        gamma = (eps * 1.25) / steps
    if y is None:
        # Using model predictions as ground truth to avoid label leaking
        preds_max = preds.data.max(1)[1]
        y = torch.equal(preds, preds_max).float()

    
    # Compute loss
    x_advs = [x.clone() for _ in range(7)] 
    if randinit:
        x_advs = [(x_adv+(2.0 * torch.rand(x_adv.shape) - 1.0) * eps) for x_adv in x_advs]        
        x_advs = [Variable(x_adv.cuda(), requires_grad=True) for x_adv in x_advs]        

    x = x.cuda()
    for t in range(steps):
        for i in range(7):
            x_adv = x_advs[i]
            out_adv_branch = model(x_adv)
            out = out_adv_branch[i]
            loss_adv0 = loss_fn(out, y)
            grad0 = torch.autograd.grad(loss_adv0, x_adv, only_inputs=True)[0]
            x_adv.data.add_(gamma * torch.sign(grad0.data))
            linfball_proj(x, eps, x_adv, in_place=True)
            x_adv = torch.clamp(x_adv, 0, 1)
            x_advs[i] = x_adv


    # determine adv images
    losses = []
    for i in range(7):
        x_adv = x_advs[i]
        out_adv_branch = model(x_adv)
        for j in range(7):
            out = out_adv_branch[j]
            if j == 0:
                loss_adv0 = torch.nn.functional.cross_entropy(input=out, target=y, reduce=False)
            else:
                loss_adv0 += torch.nn.functional.cross_entropy(input=out, target=y, reduce=False)

        losses.append(loss_adv0)
    losses =  torch.stack(losses, dim=-1)
    x_advs =  torch.stack(x_advs, dim=1)
    _, idxs = losses.topk(1,dim=-1)
    idxs = idxs.long().view(-1,1)
    idxs =  idxs.unsqueeze(2).unsqueeze(3).unsqueeze(4).repeat(1, 1, 3, 32, 32)
    x_adv = torch.gather(x_advs, 1, idxs).squeeze(1)
    return x_adv






def wrm(x, preds, loss_fn, y=None, eps=None, model=None, steps=None, gamma=None, randinit=False, **kwargs):
    if len(kwargs) > 0:
        assert set(kwargs.keys()).issubset(_extra_args)
    # if gamma is None:
        # gamma = 1.3
    # gamma = 0.5/eps
    gamma = None
    if y is None:
        # Using model predictions as ground truth to avoid label leaking
        preds_max = preds.data.max(1)[1]
        y = torch.equal(preds, preds_max).float()

    # Compute loss

    x_adv = x.clone()
    if randinit:
        # x_adv += torch.randn_like(x_adv).clamp_(min=-1.0, max=1.0) * eps
        x_adv += (2.0 * torch.rand(x_adv.shape) - 1.0) * eps

    x_adv = Variable(x_adv.cuda(), requires_grad=True)
    x = Variable(x.cuda(), requires_grad=True)

    ord = 2
    for t in range(steps):
        loss_adv0 = eps * loss_fn(model(x_adv), y) - \
                    0.5 * torch.sum(torch.norm((x_adv - x).view(x_adv.size(0), -1), p=ord, dim=1) ** 2)
        
        grad0 = torch.autograd.grad(loss_adv0, x_adv, only_inputs=True)[0]
        scale = float(1./np.sqrt(t+2))
        x_adv.data.add_(scale * grad0.data)
        x_adv = torch.clamp(x_adv, 0, 1)


    return x_adv


def wrm_k(x, preds, loss_fn, y=None, eps=None, model=None, steps=3, gamma=None, randinit=False, **kwargs):
    k = int(kwargs['branch_num'])
    #if len(kwargs) > 0:
    #    assert set(kwargs.keys()).issubset(_extra_args)
    if eps is None:
        # eps = 0.07972772183418274
        # eps = 0.45474205
        eps = 0.062
    if gamma is None:
        gamma = (eps * 1.25) / steps
    if y is None:
        # Using model predictions as ground truth to avoid label leaking
        preds_max = preds.data.max(1)[1]
        y = torch.equal(preds, preds_max).float()

    # Compute loss

    x_adv = x.clone()
    if randinit:
        x_adv += (2.0 * torch.rand(x_adv.shape) - 1.0) * eps
    x_adv = Variable(x_adv.cuda(), requires_grad=True)
    x = Variable(x.cuda(), requires_grad=True)

    ord = 2
    for t in range(steps):
        out_adv_branch = model(x_adv)
        out = out_adv_branch[k]
        loss_adv0 = eps * loss_fn(out, y) - \
                    0.5 * torch.sum(torch.norm((x_adv - x).view(x_adv.size(0), -1), p=ord, dim=1) ** 2)
        
        grad0 = torch.autograd.grad(loss_adv0, x_adv, only_inputs=True)[0]
        scale = float(1./np.sqrt(t+2))
        x_adv.data.add_(scale * grad0.data)
        x_adv = torch.clamp(x_adv, 0, 1)
        
    return x_adv



def wrm_branchy(x, preds, loss_fn, y=None, eps=None, model=None, steps=3, gamma=None, randinit=False, **kwargs):
    #if len(kwargs) > 0:
    #    assert set(kwargs.keys()).issubset(_extra_args)
    if eps is None:
        # eps = 0.07972772183418274
        # eps = 0.45474205
        eps = 0.062
    if gamma is None:
        gamma = (eps * 1.25) / steps
    if y is None:
        # Using model predictions as ground truth to avoid label leaking
        preds_max = preds.data.max(1)[1]
        y = torch.equal(preds, preds_max).float()

    # Compute loss

    x_adv = x.clone()
    if randinit:
        x_adv += (2.0 * torch.rand(x_adv.shape) - 1.0) * eps
    x_adv = Variable(x_adv.cuda(), requires_grad=True)
    x = Variable(x.cuda(), requires_grad=True)

    ord = 2
    for t in range(steps):
        out_adv_branch = model(x_adv)
        loss_adv0 = eps * loss_fn(out_adv_branch[-1], y) - \
                    0.5 * torch.sum(torch.norm((x_adv - x).view(x_adv.size(0), -1), p=ord, dim=1) ** 2)

        for i in range(6):
            loss_adv0 += eps * loss_fn(out_adv_branch[i], y) - \
                    0.5 * torch.sum(torch.norm((x_adv - x).view(x_adv.size(0), -1), p=ord, dim=1) ** 2)
        grad0 = torch.autograd.grad(loss_adv0, x_adv, only_inputs=True)[0]
        scale = float(1./np.sqrt(t+2))
        x_adv.data.add_(scale * grad0.data)
        x_adv = torch.clamp(x_adv, 0, 1)

    return x_adv



def wrm_max_v2(x, preds, loss_fn, y=None, eps=None, model=None, steps=3, gamma=None, randinit=False, **kwargs):
    #if len(kwargs) > 0:
    #    assert set(kwargs.keys()).issubset(_extra_args)
    if eps is None:
        # eps = 0.07972772183418274
        # eps = 0.45474205
        eps = 0.062
    if gamma is None:
        gamma = (eps * 1.25) / steps
    if y is None:
        # Using model predictions as ground truth to avoid label leaking
        preds_max = preds.data.max(1)[1]
        y = torch.equal(preds, preds_max).float()

    
    # Compute loss
    x_advs = [x.clone() for _ in range(7)] 
    if randinit:
        x_advs = [(x_adv+(2.0 * torch.rand(x_adv.shape) - 1.0) * eps) for x_adv in x_advs]        
        x_advs = [Variable(x_adv.cuda(), requires_grad=True) for x_adv in x_advs]        

    x = Variable(x.cuda(), requires_grad=True)
    ord = 2
    for t in range(steps):
        for i in range(7):
            x_adv = x_advs[i]
            out_adv_branch = model(x_adv)
            out = out_adv_branch[i]
            loss_adv0 = eps * loss_fn(out, y) - \
                    0.5 * torch.sum(torch.norm((x_adv - x).view(x_adv.size(0), -1), p=ord, dim=1) ** 2)

            grad0 = torch.autograd.grad(loss_adv0, x_adv, only_inputs=True)[0]
            scale = float(1./np.sqrt(t+2))
            x_adv.data.add_(scale * grad0.data)
            x_adv = torch.clamp(x_adv, 0, 1)
            x_advs[i] = x_adv

    # calculate loss
    losses = []
    for i in range(7):
        x_adv = x_advs[i]
        out_adv_branch = model(x_adv)
        for j in range(7):
            out = out_adv_branch[j]
            if j == 0:
                loss_adv0 = torch.nn.functional.cross_entropy(input=out, target=y, reduce=False)
            else:
                loss_adv0 += torch.nn.functional.cross_entropy(input=out, target=y, reduce=False)

        losses.append(loss_adv0)
    losses =  torch.stack(losses, dim=-1)
    x_advs =  torch.stack(x_advs, dim=1)
    _, idxs = losses.topk(1,dim=-1)
    idxs = idxs.long().view(-1,1)
    idxs =  idxs.unsqueeze(2).unsqueeze(3).unsqueeze(4).repeat(1, 1, 3, 32, 32)
    x_adv = torch.gather(x_advs, 1, idxs).squeeze(1)
    return x_adv





def fgm(x, preds, loss_fn, y=None, eps=None, model=None, steps=None, gamma=None, randinit=None, **kwargs):
    if len(kwargs) > 0:
        assert set(kwargs.keys()).issubset(_extra_args)
    if eps is None:
        # eps = 0.07972772183418274
        # eps = 0.45474205
        eps = 0.062
    if y is None:
        # Using model predictions as ground truth to avoid label leaking
        preds_max = preds.data.max(1)[1]
        y = torch.equal(preds, preds_max).float()

    # Compute loss

    x_adv = x.clone()
    x_adv = Variable(x_adv.cuda(), requires_grad=True)
    x = x.cuda()


    loss_adv0 = loss_fn(model(x_adv), y)
    grad0 = torch.autograd.grad(loss_adv0, x_adv, only_inputs=True)[0]
    x_adv.data.add_(eps * torch.sign(grad0.data))
    x_adv = torch.clamp(x_adv, 0, 1)

    return x_adv


def fgm_k(x, preds, loss_fn, y=None, eps=None, model=None, steps=3, gamma=None, randinit=False, **kwargs):
    k = int(kwargs['branch_num'])
    #if len(kwargs) > 0:
    #    assert set(kwargs.keys()).issubset(_extra_args)
    if eps is None:
        # eps = 0.07972772183418274
        # eps = 0.45474205
        eps = 0.062
    if gamma is None:
        gamma = (eps * 1.25) / steps
    if y is None:
        # Using model predictions as ground truth to avoid label leaking
        preds_max = preds.data.max(1)[1]
        y = torch.equal(preds, preds_max).float()

    # Compute loss

    x_adv = x.clone()
    if randinit:
        x_adv += (2.0 * torch.rand(x_adv.shape) - 1.0) * eps
    x_adv = Variable(x_adv.cuda(), requires_grad=True)
    x = x.cuda()
    out_adv_branch = model(x_adv)
    out = out_adv_branch[k]
    loss_adv0 = loss_fn(out, y)
    grad0 = torch.autograd.grad(loss_adv0, x_adv, only_inputs=True)[0]
    x_adv.data.add_(eps * torch.sign(grad0.data))
    x_adv = torch.clamp(x_adv, 0, 1)
        
    return x_adv


def fgm_branchy(x, preds, loss_fn, y=None, eps=None, model=None, steps=3, gamma=None, randinit=False, **kwargs):
    #if len(kwargs) > 0:
    #    assert set(kwargs.keys()).issubset(_extra_args)
    if eps is None:
        # eps = 0.07972772183418274
        # eps = 0.45474205
        eps = 0.062
    if gamma is None:
        gamma = (eps * 1.25) / steps
    if y is None:
        # Using model predictions as ground truth to avoid label leaking
        preds_max = preds.data.max(1)[1]
        y = torch.equal(preds, preds_max).float()

    # Compute loss

    x_adv = x.clone()
    if randinit:
        x_adv += (2.0 * torch.rand(x_adv.shape) - 1.0) * eps
    x_adv = Variable(x_adv.cuda(), requires_grad=True)
    x = x.cuda()
    out_adv_branch = model(x_adv)
    counter = 0
    loss_adv0 = loss_fn(out_adv_branch[-1], y) * (1.0/7)
    for i in range(6):
        loss_adv0 += loss_fn(out_adv_branch[i], y) * (1.0/7)
    grad0 = torch.autograd.grad(loss_adv0, x_adv, only_inputs=True)[0]
    x_adv.data.add_(eps * torch.sign(grad0.data))
    x_adv = torch.clamp(x_adv, 0, 1)
        
    return x_adv


def fgm_max_v2(x, preds, loss_fn, y=None, eps=None, model=None, steps=3, gamma=None, randinit=False, **kwargs):
    #if len(kwargs) > 0:
    #    assert set(kwargs.keys()).issubset(_extra_args)
    if eps is None:
        # eps = 0.07972772183418274
        # eps = 0.45474205
        eps = 0.062
    if gamma is None:
        gamma = (eps * 1.25) / steps
    if y is None:
        # Using model predictions as ground truth to avoid label leaking
        preds_max = preds.data.max(1)[1]
        y = torch.equal(preds, preds_max).float()

    
    # Compute loss
    x_advs = [x.clone() for _ in range(7)] 
    if randinit:
        x_advs = [(x_adv+(2.0 * torch.rand(x_adv.shape) - 1.0) * eps) for x_adv in x_advs]        
        x_advs = [Variable(x_adv.cuda(), requires_grad=True) for x_adv in x_advs]        

    x = x.cuda()
    for i in range(7):
        x_adv = x_advs[i]
        out_adv_branch = model(x_adv)
        out = out_adv_branch[i]
        loss_adv0 = loss_fn(out, y)
        grad0 = torch.autograd.grad(loss_adv0, x_adv, only_inputs=True)[0]
        x_adv.data.add_(eps * torch.sign(grad0.data))
        x_adv = torch.clamp(x_adv, 0, 1)
        x_advs[i] = x_adv

    losses = []
    for i in range(7):
        x_adv = x_advs[i]
        out_adv_branch = model(x_adv)
        for j in range(7):
            out = out_adv_branch[j]
            if j == 0:
                loss_adv0 = torch.nn.functional.cross_entropy(input=out, target=y, reduce=False)
            else:
                loss_adv0 += torch.nn.functional.cross_entropy(input=out, target=y, reduce=False)

        losses.append(loss_adv0)
    losses =  torch.stack(losses, dim=-1)
    x_advs =  torch.stack(x_advs, dim=1)
    _, idxs = losses.topk(1,dim=-1)
    idxs = idxs.long().view(-1,1)
    idxs =  idxs.unsqueeze(2).unsqueeze(3).unsqueeze(4).repeat(1, 1, 3, 32, 32)
    x_adv = torch.gather(x_advs, 1, idxs).squeeze(1)
    return x_adv


def ifgm_idx(x, preds, loss_fn, y=None, eps=None, model=None, steps=3, gamma=None, randinit=False, **kwargs):
    #if len(kwargs) > 0:
    #    assert set(kwargs.keys()).issubset(_extra_args)
    
    idxs = kwargs['idx'].cuda()
    if eps is None:
        # eps = 0.07972772183418274
        # eps = 0.45474205
        eps = 0.062
    if gamma is None:
        gamma = (eps * 1.25) / steps
    if y is None:
        # Using model predictions as ground truth to avoid label leaking
        preds_max = preds.data.max(1)[1]
        y = torch.equal(preds, preds_max).float()

    
    # Compute loss
    x_advs = [x.clone() for _ in range(7)] 
    if randinit:
        x_advs = [(x_adv+(2.0 * torch.rand(x_adv.shape) - 1.0) * eps) for x_adv in x_advs]        
        x_advs = [Variable(x_adv.cuda(), requires_grad=True) for x_adv in x_advs]        

    x = x.cuda()
    for t in range(steps):
        for i in range(7):
            x_adv = x_advs[i]
            out_adv_branch = model(x_adv)
            out = out_adv_branch[i]
            loss_adv0 = loss_fn(out, y)
            grad0 = torch.autograd.grad(loss_adv0, x_adv, only_inputs=True)[0]
            x_adv.data.add_(gamma * torch.sign(grad0.data))
            linfball_proj(x, eps, x_adv, in_place=True)
            x_adv = torch.clamp(x_adv, 0, 1)
            x_advs[i] = x_adv

    x_advs =  torch.stack(x_advs, dim=1)
    idxs = idxs.long().view(-1,1)
    idxs =  idxs.unsqueeze(2).unsqueeze(3).unsqueeze(4).repeat(1, 1, 3, 32, 32)
    x_adv = torch.gather(x_advs, 1, idxs).squeeze(1)
    return x_adv


def ifgm_decision_random(x, preds, loss_fn, y=None, eps=None, model=None, steps=3, gamma=None, randinit=False, **kwargs):

    T_ori = kwargs['T']
    if eps is None:
        eps = 0.062
    if gamma is None:
        gamma = (eps * 1.25) / steps
    if y is None:
        # Using model predictions as ground truth to avoid label leaking
        preds_max = preds.data.max(1)[1]
        y = torch.equal(preds, preds_max).float()

    # Compute loss

    x_adv = x.clone()
    if randinit:
        x_adv += (2.0 * torch.rand(x_adv.shape) - 1.0) * eps
    x_adv = Variable(x_adv.cuda(), requires_grad=True)
    x = x.cuda()
    for t in range(steps):
        out_adv_branch = model(x_adv)
        out = torch.stack([out_adv_branch[i] for i in range(7)], dim=1)
        sm = nn.functional.softmax
        prob_branch1 = sm(out_adv_branch[0])
        prob_branch2 = sm(out_adv_branch[1])
        prob_branch3 = sm(out_adv_branch[2])
        prob_branch4 = sm(out_adv_branch[3])
        prob_branch5 = sm(out_adv_branch[4])
        prob_branch6 = sm(out_adv_branch[5])
        prob_main = sm(out_adv_branch[6])
        measure_branch1 = torch.sum(torch.mul(-prob_branch1, torch.log(prob_branch1 + 1e-5)), dim=1)
        measure_branch2 = torch.sum(torch.mul(-prob_branch2, torch.log(prob_branch2 + 1e-5)), dim=1)
        measure_branch3 = torch.sum(torch.mul(-prob_branch3, torch.log(prob_branch3 + 1e-5)), dim=1)
        measure_branch4 = torch.sum(torch.mul(-prob_branch4, torch.log(prob_branch4 + 1e-5)), dim=1)
        measure_branch5 = torch.sum(torch.mul(-prob_branch5, torch.log(prob_branch5 + 1e-5)), dim=1)
        measure_branch6 = torch.sum(torch.mul(-prob_branch6, torch.log(prob_branch6 + 1e-5)), dim=1)
        idxs = []
        #T = np.asarray(T_ori) *  np.random.uniform(low=-4.5, high=4.5, size=(6,))
        T = T_ori
        # random entropy       
        
        #r = 0.5
        #r = [0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
        r = [1, 1, 1, 1, 1, 1] 
        for j in range(x.size(0)):
            if (measure_branch1.data).cpu().numpy()[j] * ( 1 +  np.random.uniform(low=-r[0], high=r[0]))  < T[0] :
                idxs.append(0)
            elif (measure_branch2.data).cpu().numpy()[j] * ( 1 + np.random.uniform(low=-r[1], high=r[1]))  < T[1] :
                idxs.append(1)
            elif (measure_branch3.data).cpu().numpy()[j] * ( 1 + np.random.uniform(low=-r[2], high=r[2])) < T[2] :
                idxs.append(2)
            elif (measure_branch4.data).cpu().numpy()[j] * ( 1 + np.random.uniform(low=-r[3], high=r[3])) < T[3] :
                idxs.append(3)
            elif (measure_branch5.data).cpu().numpy()[j] * ( 1 + np.random.uniform(low=-r[4], high=r[4])) < T[4] :
                idxs.append(4)
            elif (measure_branch6.data).cpu().numpy()[j] * ( 1 + np.random.uniform(low=-r[5], high=r[5])) < T[5] :
                idxs.append(5)
            else:
                idxs.append(6)
        #print(idxs)
        
        # random fix probability
        '''
        r = 0.5
        
        for j in range(x.size(0)):
            if (measure_branch1.data).cpu().numpy()[j] * ( 1 +  np.random.uniform(low=-r, high=r))  < T[0] :
                if np.random.choice([0,1,2]) == 0:
                    idxs.append(np.random.choice([1,2,3,4,5,6])) 
                else:
                    idxs.append(0) 
            elif (measure_branch2.data).cpu().numpy()[j] * ( 1 + np.random.uniform(low=-r, high=r))  < T[1] :
                if np.random.choice([0,1,2]) == 0:
                    idxs.append(np.random.choice([0,2,3,4,5,6])) 
                else:
                    idxs.append(1) 
            elif (measure_branch3.data).cpu().numpy()[j] * ( 1 + np.random.uniform(low=-r, high=r)) < T[2] :
                if np.random.choice([0,1,2]) == 0:
                    idxs.append(np.random.choice([0,1,3,4,5,6])) 
                else:
                    idxs.append(2) 
            elif (measure_branch4.data).cpu().numpy()[j] * ( 1 + np.random.uniform(low=-r, high=r)) < T[3] :
                if np.random.choice([0,1,2]) == 0:
                    idxs.append(np.random.choice([0,1,2,4,5,6])) 
                else:
                    idxs.append(3) 
            elif (measure_branch5.data).cpu().numpy()[j] * ( 1 + np.random.uniform(low=-r, high=r)) < T[4] :
                if np.random.choice([0,1,2]) == 0:
                    idxs.append(np.random.choice([0,1,2,3,5,6])) 
                else:
                    idxs.append(4) 
            elif (measure_branch6.data).cpu().numpy()[j] * ( 1 + np.random.uniform(low=-r, high=r)) < T[5] :
                if np.random.choice([0,1,2]) == 0:
                    idxs.append(np.random.choice([0,1,2,3,4,6])) 
                else:
                    idxs.append(5) 
            else:
                if np.random.choice([0,1,2]) == 0:
                    idxs.append(np.random.choice([0,1,2,3,4,5])) 
                else:
                    idxs.append(6) 
        '''
        '''
        # all random
        for j in range(x.size(0)):
            idxs.append(np.random.choice([0,1,2,3,4,5,6]))
        '''
        #print(idxs)
        

        idxs = torch.from_numpy(np.asarray(idxs)).cuda()
        idxs = idxs.long().view(-1,1)
        idxs = idxs.unsqueeze(2).repeat(1, 1, 10)
        out = torch.gather(out, 1, idxs).squeeze(1)
        
        loss_adv0 = loss_fn(out, y)
        grad0 = torch.autograd.grad(loss_adv0, x_adv, only_inputs=True)[0]
        x_adv.data.add_(gamma * torch.sign(grad0.data))
        linfball_proj(x, eps, x_adv, in_place=True)
        x_adv = torch.clamp(x_adv, 0, 1)
        

        
    return x_adv


def ifgm_ensemble(x, preds, loss_fn, y=None, eps=None, model=None, steps=3, gamma=None, randinit=False, **kwargs):

    T = kwargs['T']
    if eps is None:
        eps = 0.062
    if gamma is None:
        gamma = (eps * 1.25) / steps
    if y is None:
        # Using model predictions as ground truth to avoid label leaking
        preds_max = preds.data.max(1)[1]
        y = torch.equal(preds, preds_max).float()

    # Compute loss

    x_adv = x.clone()
    if randinit:
        x_adv += (2.0 * torch.rand(x_adv.shape) - 1.0) * eps
    x_adv = Variable(x_adv.cuda(), requires_grad=True)
    x = x.cuda()
    for t in range(steps):
        out_adv_branch = model(x_adv)
        #out = torch.stack([out_adv_branch[i] for i in range(7)], dim=1)
        sm = nn.functional.softmax
        prob_branch1 = sm(out_adv_branch[0])
        prob_branch2 = sm(out_adv_branch[1])
        prob_branch3 = sm(out_adv_branch[2])
        prob_branch4 = sm(out_adv_branch[3])
        prob_branch5 = sm(out_adv_branch[4])
        prob_branch6 = sm(out_adv_branch[5])
        prob_main = sm(out_adv_branch[6])
        measure_branch1 = torch.sum(torch.mul(-prob_branch1, torch.log(prob_branch1 + 1e-5)), dim=1)
        measure_branch2 = torch.sum(torch.mul(-prob_branch2, torch.log(prob_branch2 + 1e-5)), dim=1)
        measure_branch3 = torch.sum(torch.mul(-prob_branch3, torch.log(prob_branch3 + 1e-5)), dim=1)
        measure_branch4 = torch.sum(torch.mul(-prob_branch4, torch.log(prob_branch4 + 1e-5)), dim=1)
        measure_branch5 = torch.sum(torch.mul(-prob_branch5, torch.log(prob_branch5 + 1e-5)), dim=1)
        measure_branch6 = torch.sum(torch.mul(-prob_branch6, torch.log(prob_branch6 + 1e-5)), dim=1)
        idxs = []
        out = []
        r = 0.0
        for j in range(x.size(0)):
            if (measure_branch1.data).cpu().numpy()[j] * ( 1 +  np.random.uniform(low=-r, high=r))  < T[0] :
                idx = 0
            elif (measure_branch2.data).cpu().numpy()[j] * ( 1 +  np.random.uniform(low=-r, high=r))  < T[1] :
                idx = 1
            elif (measure_branch3.data).cpu().numpy()[j] * ( 1 +  np.random.uniform(low=-r, high=r))  < T[2] :
                idx = 2
            elif (measure_branch4.data).cpu().numpy()[j] * ( 1 +  np.random.uniform(low=-r, high=r))  < T[3] :
                idx = 3
            elif (measure_branch5.data).cpu().numpy()[j] * ( 1 +  np.random.uniform(low=-r, high=r))  < T[4] :
                idx = 4
            elif (measure_branch6.data).cpu().numpy()[j] * ( 1 +  np.random.uniform(low=-r, high=r))  < T[5] :
                idx = 5
            else:
                idx = 6
            current_o = 0
            for k in range(idx+1):
                current_o +=  out_adv_branch[k][j]
            out.append(current_o.unsqueeze(0))
        #print(idxs)
        #idxs = torch.from_numpy(np.asarray(idxs)).cuda()
        #idxs = idxs.long().view(-1,1)
        #idxs = idxs.unsqueeze(2).repeat(1, 1, 10)
        #out = torch.gather(out, 1, idxs).squeeze(1)
        out =  torch.cat(out, 0)    
        #print(out.shape)
        
        loss_adv0 = loss_fn(out, y)
        grad0 = torch.autograd.grad(loss_adv0, x_adv, only_inputs=True)[0]
        x_adv.data.add_(gamma * torch.sign(grad0.data))
        linfball_proj(x, eps, x_adv, in_place=True)
        x_adv = torch.clamp(x_adv, 0, 1)
        

        
    return x_adv




def ifgm_decision(x, preds, loss_fn, y=None, eps=None, model=None, steps=3, gamma=None, randinit=False, **kwargs):

    #T = kwargs['T']
    T = [0.26418662667274473, 0.3697565674781799, 0.3614460229873657, 0.7663927197456359, 1.0350646495819091, 1.3060622]
    if eps is None:
        eps = 0.062
    if gamma is None:
        gamma = (eps * 1.25) / steps
    if y is None:
        # Using model predictions as ground truth to avoid label leaking
        preds_max = preds.data.max(1)[1]
        y = torch.equal(preds, preds_max).float()

    # Compute loss

    x_adv = x.clone()
    if randinit:
        x_adv += (2.0 * torch.rand(x_adv.shape) - 1.0) * eps
    x_adv = Variable(x_adv.cuda(), requires_grad=True)
    x = x.cuda()
    for t in range(steps):
        out_adv_branch = model(x_adv)
        out = torch.stack([out_adv_branch[i] for i in range(7)], dim=1)
        sm = nn.functional.softmax
        prob_branch1 = sm(out_adv_branch[0])
        prob_branch2 = sm(out_adv_branch[1])
        prob_branch3 = sm(out_adv_branch[2])
        prob_branch4 = sm(out_adv_branch[3])
        prob_branch5 = sm(out_adv_branch[4])
        prob_branch6 = sm(out_adv_branch[5])
        prob_main = sm(out_adv_branch[6])
        measure_branch1 = torch.sum(torch.mul(-prob_branch1, torch.log(prob_branch1 + 1e-5)), dim=1)
        measure_branch2 = torch.sum(torch.mul(-prob_branch2, torch.log(prob_branch2 + 1e-5)), dim=1)
        measure_branch3 = torch.sum(torch.mul(-prob_branch3, torch.log(prob_branch3 + 1e-5)), dim=1)
        measure_branch4 = torch.sum(torch.mul(-prob_branch4, torch.log(prob_branch4 + 1e-5)), dim=1)
        measure_branch5 = torch.sum(torch.mul(-prob_branch5, torch.log(prob_branch5 + 1e-5)), dim=1)
        measure_branch6 = torch.sum(torch.mul(-prob_branch6, torch.log(prob_branch6 + 1e-5)), dim=1)
        idxs = []
        for j in range(x.size(0)):
            if (measure_branch1.data).cpu().numpy()[j] < T[0] :
                idxs.append(0)
            elif (measure_branch2.data).cpu().numpy()[j] < T[1] :
                idxs.append(1)
            elif (measure_branch3.data).cpu().numpy()[j] < T[2] :
                idxs.append(2)
            elif (measure_branch4.data).cpu().numpy()[j] < T[3] :
                idxs.append(3)
            elif (measure_branch5.data).cpu().numpy()[j] < T[4] :
                idxs.append(4)
            elif (measure_branch6.data).cpu().numpy()[j] < T[5] :
                idxs.append(5)
            else:
                idxs.append(6)
        idxs = torch.from_numpy(np.asarray(idxs)).cuda()
        idxs = idxs.long().view(-1,1)
        idxs = idxs.unsqueeze(2).repeat(1, 1, 10)
        out = torch.gather(out, 1, idxs).squeeze(1)
        
        loss_adv0 = loss_fn(out, y)
        grad0 = torch.autograd.grad(loss_adv0, x_adv, only_inputs=True)[0]
        x_adv.data.add_(gamma * torch.sign(grad0.data))
        linfball_proj(x, eps, x_adv, in_place=True)
        x_adv = torch.clamp(x_adv, 0, 1)
        

        
    return x_adv

def ifgm_max_v3(x, preds, loss_fn, y=None, eps=None, model=None, steps=3, gamma=None, randinit=False, **kwargs):
    #T = kwargs['T']
    T = [0.26418662667274473, 0.3697565674781799, 0.3614460229873657, 0.7663927197456359, 1.0350646495819091, 1.3060622]
    if eps is None:
        eps = 0.062
    if gamma is None:
        gamma = (eps * 1.25) / steps
    if y is None:
        # Using model predictions as ground truth to avoid label leaking
        preds_max = preds.data.max(1)[1]
        y = torch.equal(preds, preds_max).float()

    
    # Compute loss
    x_advs = [x.clone() for _ in range(8)] 
    if randinit:
        x_advs = [(x_adv+(2.0 * torch.rand(x_adv.shape) - 1.0) * eps) for x_adv in x_advs]        
        x_advs = [Variable(x_adv.cuda(), requires_grad=True) for x_adv in x_advs]        

    x = x.cuda()
    for t in range(steps):
        for i in range(8):
            x_adv = x_advs[i]
            out_adv_branch = model(x_adv)
            if i < len(out_adv_branch):
                out = out_adv_branch[i]
                loss_adv0 = loss_fn(out, y)
            else:
                out = torch.stack([out_adv_branch[i] for i in range(7)], dim=1)
                sm = nn.functional.softmax
                prob_branch1 = sm(out_adv_branch[0])
                prob_branch2 = sm(out_adv_branch[1])
                prob_branch3 = sm(out_adv_branch[2])
                prob_branch4 = sm(out_adv_branch[3])
                prob_branch5 = sm(out_adv_branch[4])
                prob_branch6 = sm(out_adv_branch[5])
                measure_branch1 = torch.sum(torch.mul(-prob_branch1, torch.log(prob_branch1 + 1e-5)), dim=1)
                measure_branch2 = torch.sum(torch.mul(-prob_branch2, torch.log(prob_branch2 + 1e-5)), dim=1)
                measure_branch3 = torch.sum(torch.mul(-prob_branch3, torch.log(prob_branch3 + 1e-5)), dim=1)
                measure_branch4 = torch.sum(torch.mul(-prob_branch4, torch.log(prob_branch4 + 1e-5)), dim=1)
                measure_branch5 = torch.sum(torch.mul(-prob_branch5, torch.log(prob_branch5 + 1e-5)), dim=1)
                measure_branch6 = torch.sum(torch.mul(-prob_branch6, torch.log(prob_branch6 + 1e-5)), dim=1)
                indexs = []
                for j in range(x.size(0)):
                    if (measure_branch1.data).cpu().numpy()[j] < T[0] :
                        indexs.append(0)
                    elif (measure_branch2.data).cpu().numpy()[j] < T[1] :
                        indexs.append(1)
                    elif (measure_branch3.data).cpu().numpy()[j] < T[2] :
                        indexs.append(2)
                    elif (measure_branch4.data).cpu().numpy()[j] < T[3] :
                        indexs.append(3)
                    elif (measure_branch5.data).cpu().numpy()[j] < T[4] :
                        indexs.append(4)
                    elif (measure_branch6.data).cpu().numpy()[j] < T[5] :
                        indexs.append(5)
                    else:
                        indexs.append(6)
                indexs = torch.from_numpy(np.asarray(indexs)).cuda()
                indexs = indexs.long().view(-1,1)
                indexs = indexs.unsqueeze(2).repeat(1, 1, 10)
                out = torch.gather(out, 1, indexs).squeeze(1)
                loss_adv0 = loss_fn(out, y)
                
                
            grad0 = torch.autograd.grad(loss_adv0, x_adv, only_inputs=True)[0]
            x_adv.data.add_(gamma * torch.sign(grad0.data))
            linfball_proj(x, eps, x_adv, in_place=True)
            x_adv = torch.clamp(x_adv, 0, 1)
            x_advs[i] = x_adv
    # determine adv images
    losses = []
    for i in range(8):
        x_adv = x_advs[i]
        out_adv_branch = model(x_adv)
        for j in range(7):
            out = out_adv_branch[j]
            if j == 0:
                loss_adv0 = torch.nn.functional.cross_entropy(input=out, target=y, reduce=False)
            else:
                loss_adv0 += torch.nn.functional.cross_entropy(input=out, target=y, reduce=False)

        losses.append(loss_adv0)
    losses =  torch.stack(losses, dim=-1)
    x_advs =  torch.stack(x_advs, dim=1)
    _, idxs = losses.topk(1,dim=-1)
    idxs = idxs.long().view(-1,1)
    idxs =  idxs.unsqueeze(2).unsqueeze(3).unsqueeze(4).repeat(1, 1, 3, 32, 32)
    x_adv = torch.gather(x_advs, 1, idxs).squeeze(1)   
    return x_adv

def ifgm_max_v4(x, preds, loss_fn, y=None, eps=None, model=None, steps=3, gamma=None, randinit=False, **kwargs):
    if eps is None:
        eps = 0.062
    if gamma is None:
        gamma = (eps * 1.25) / steps
    if y is None:
        # Using model predictions as ground truth to avoid label leaking
        preds_max = preds.data.max(1)[1]
        y = torch.equal(preds, preds_max).float()

    
    # Compute loss
    x_advs = [x.clone() for _ in range(8)] 
    if randinit:
        x_advs = [(x_adv+(2.0 * torch.rand(x_adv.shape) - 1.0) * eps) for x_adv in x_advs]        
        x_advs = [Variable(x_adv.cuda(), requires_grad=True) for x_adv in x_advs]        

    x = x.cuda()
    for t in range(steps):
        for i in range(8):
            x_adv = x_advs[i]
            out_adv_branch = model(x_adv)
            if i < len(out_adv_branch):
                out = out_adv_branch[i]
                loss_adv0 = loss_fn(out, y)
            else:
                out = torch.stack([out_adv_branch[i] for i in range(7)], dim=1)
                sm = nn.functional.softmax
                prob_branch1 = sm(out_adv_branch[0])
                prob_branch2 = sm(out_adv_branch[1])
                prob_branch3 = sm(out_adv_branch[2])
                prob_branch4 = sm(out_adv_branch[3])
                prob_branch5 = sm(out_adv_branch[4])
                prob_branch6 = sm(out_adv_branch[5])
                prob_main = sm(out_adv_branch[6])
                measure_branch1 = torch.sum(torch.mul(-prob_branch1, torch.log(prob_branch1 + 1e-5)), dim=1)
                measure_branch2 = torch.sum(torch.mul(-prob_branch2, torch.log(prob_branch2 + 1e-5)), dim=1)
                measure_branch3 = torch.sum(torch.mul(-prob_branch3, torch.log(prob_branch3 + 1e-5)), dim=1)
                measure_branch4 = torch.sum(torch.mul(-prob_branch4, torch.log(prob_branch4 + 1e-5)), dim=1)
                measure_branch5 = torch.sum(torch.mul(-prob_branch5, torch.log(prob_branch5 + 1e-5)), dim=1)
                measure_branch6 = torch.sum(torch.mul(-prob_branch6, torch.log(prob_branch6 + 1e-5)), dim=1)
                measure_main = torch.sum(torch.mul(-prob_main, torch.log(prob_main + 1e-5)), dim=1)
                indexs = []

                for j in range(x.size(0)):
                    entropy = []
                    entropy.append((measure_branch1.data).cpu().numpy()[j])
                    entropy.append((measure_branch2.data).cpu().numpy()[j])
                    entropy.append((measure_branch3.data).cpu().numpy()[j])
                    entropy.append((measure_branch4.data).cpu().numpy()[j])
                    entropy.append((measure_branch5.data).cpu().numpy()[j])
                    entropy.append((measure_branch6.data).cpu().numpy()[j])
                    entropy.append((measure_main.data).cpu().numpy()[j])
                    indexs.append(entropy.index(min(entropy)))
                indexs = torch.from_numpy(np.asarray(indexs)).cuda()
                indexs = indexs.long().view(-1,1)
                indexs = indexs.unsqueeze(2).repeat(1, 1, 10)
                out = torch.gather(out, 1, indexs).squeeze(1)
                loss_adv0 = loss_fn(out, y)
                
                
            grad0 = torch.autograd.grad(loss_adv0, x_adv, only_inputs=True)[0]
            x_adv.data.add_(gamma * torch.sign(grad0.data))
            linfball_proj(x, eps, x_adv, in_place=True)
            x_adv = torch.clamp(x_adv, 0, 1)
            x_advs[i] = x_adv
    # determine adv images
    losses = []
    for i in range(8):
        x_adv = x_advs[i]
        out_adv_branch = model(x_adv)
        for j in range(7):
            out = out_adv_branch[j]
            if j == 0:
                loss_adv0 = torch.nn.functional.cross_entropy(input=out, target=y, reduce=False)
            else:
                loss_adv0 += torch.nn.functional.cross_entropy(input=out, target=y, reduce=False)

        losses.append(loss_adv0)
    losses =  torch.stack(losses, dim=-1)
    x_advs =  torch.stack(x_advs, dim=1)
    _, idxs = losses.topk(1,dim=-1)
    idxs = idxs.long().view(-1,1)
    idxs =  idxs.unsqueeze(2).unsqueeze(3).unsqueeze(4).repeat(1, 1, 3, 32, 32)
    x_adv = torch.gather(x_advs, 1, idxs).squeeze(1)   
    return x_adv



def ifgm_routing(x, preds, loss_fn, y=None, eps=None, model=None, steps=3, gamma=None, randinit=False, **kwargs):
    if eps is None:
        eps = 0.062
    if gamma is None:
        gamma = (eps * 1.25) / steps
    if y is None:
        # Using model predictions as ground truth to avoid label leaking
        preds_max = preds.data.max(1)[1]
        y = torch.equal(preds, preds_max).float()

    # Compute loss

    x_adv = x.clone()
    if randinit:
        x_adv += (2.0 * torch.rand(x_adv.shape) - 1.0) * eps
    x_adv = Variable(x_adv.cuda(), requires_grad=True)
    x = x.cuda()
    #out_adv_branch = model(x_adv)

    for t in range(steps):
        out_adv_branch = model(x_adv)
        sm = nn.functional.softmax
        prob_branch1 = sm(out_adv_branch[0])
        prob_branch2 = sm(out_adv_branch[1])
        prob_branch3 = sm(out_adv_branch[2])
        prob_branch4 = sm(out_adv_branch[3])
        prob_branch5 = sm(out_adv_branch[4])
        prob_branch6 = sm(out_adv_branch[5])
        prob_main = sm(out_adv_branch[6])
        measure_branch1 = torch.sum(torch.mul(-prob_branch1, torch.log(prob_branch1 + 1e-5)), dim=1)
        measure_branch2 = torch.sum(torch.mul(-prob_branch2, torch.log(prob_branch2 + 1e-5)), dim=1)
        measure_branch3 = torch.sum(torch.mul(-prob_branch3, torch.log(prob_branch3 + 1e-5)), dim=1)
        measure_branch4 = torch.sum(torch.mul(-prob_branch4, torch.log(prob_branch4 + 1e-5)), dim=1)
        measure_branch5 = torch.sum(torch.mul(-prob_branch5, torch.log(prob_branch5 + 1e-5)), dim=1)
        measure_branch6 = torch.sum(torch.mul(-prob_branch6, torch.log(prob_branch6 + 1e-5)), dim=1)
        measure_main = torch.sum(torch.mul(-prob_main, torch.log(prob_main + 1e-5)), dim=1)
        idxs = []
        for j in range(x.size(0)):
            entropy = []
            entropy.append((measure_branch1.data).cpu().numpy()[j])
            entropy.append((measure_branch2.data).cpu().numpy()[j])
            entropy.append((measure_branch3.data).cpu().numpy()[j])
            entropy.append((measure_branch4.data).cpu().numpy()[j])
            entropy.append((measure_branch5.data).cpu().numpy()[j])
            entropy.append((measure_branch6.data).cpu().numpy()[j])
            entropy.append((measure_main.data).cpu().numpy()[j])
            idxs.append(entropy.index(min(entropy)))
        idxs = torch.from_numpy(np.asarray(idxs)).cuda()
        idxs = idxs.long().view(-1,1)
        idxs = idxs.unsqueeze(2).repeat(1, 1, 10)

        out = torch.stack([out_adv_branch[i] for i in range(7)], dim=1)
        out = torch.gather(out, 1, idxs).squeeze(1)
        loss_adv0 = loss_fn(out, y)
        grad0 = torch.autograd.grad(loss_adv0, x_adv, only_inputs=True)[0]
        x_adv.data.add_(gamma * torch.sign(grad0.data))
        linfball_proj(x, eps, x_adv, in_place=True)
        x_adv = torch.clamp(x_adv, 0, 1)
        
    return x_adv


'''
def ifgm_decision(x, preds, loss_fn, y=None, eps=None, model=None, steps=3, gamma=None, randinit=False, **kwargs):

    T = kwargs['T']
    if eps is None:
        eps = 0.062
    if gamma is None:
        gamma = (eps * 1.25) / steps
    if y is None:
        # Using model predictions as ground truth to avoid label leaking
        preds_max = preds.data.max(1)[1]
        y = torch.equal(preds, preds_max).float()

    # Compute loss

    x_adv = x.clone()
    if randinit:
        x_adv += (2.0 * torch.rand(x_adv.shape) - 1.0) * eps
    x_adv = Variable(x_adv.cuda(), requires_grad=True)
    x = x.cuda()
    out_adv_branch = model(x_adv)
    sm = nn.functional.softmax
    prob_branch1 = sm(out_adv_branch[0])
    prob_branch2 = sm(out_adv_branch[1])
    prob_branch3 = sm(out_adv_branch[2])
    prob_branch4 = sm(out_adv_branch[3])
    prob_branch5 = sm(out_adv_branch[4])
    prob_branch6 = sm(out_adv_branch[5])
    prob_main = sm(out_adv_branch[6])
    measure_branch1 = torch.sum(torch.mul(-prob_branch1, torch.log(prob_branch1 + 1e-5)), dim=1)
    measure_branch2 = torch.sum(torch.mul(-prob_branch2, torch.log(prob_branch2 + 1e-5)), dim=1)
    measure_branch3 = torch.sum(torch.mul(-prob_branch3, torch.log(prob_branch3 + 1e-5)), dim=1)
    measure_branch4 = torch.sum(torch.mul(-prob_branch4, torch.log(prob_branch4 + 1e-5)), dim=1)
    measure_branch5 = torch.sum(torch.mul(-prob_branch5, torch.log(prob_branch5 + 1e-5)), dim=1)
    measure_branch6 = torch.sum(torch.mul(-prob_branch6, torch.log(prob_branch6 + 1e-5)), dim=1)
    idxs = []
    for j in range(x.size(0)):
        if (measure_branch1.data).cpu().numpy()[j] < T[0] :
            idxs.append(0)
        elif (measure_branch2.data).cpu().numpy()[j] < T[1] :
            idxs.append(1)
        elif (measure_branch3.data).cpu().numpy()[j] < T[2] :
            idxs.append(2)
        elif (measure_branch4.data).cpu().numpy()[j] < T[3] :
            idxs.append(3)
        elif (measure_branch5.data).cpu().numpy()[j] < T[4] :
            idxs.append(4)
        elif (measure_branch6.data).cpu().numpy()[j] < T[5] :
            idxs.append(5)
        else:
            idxs.append(6)
    idxs = torch.from_numpy(np.asarray(idxs)).cuda()
    idxs = idxs.long().view(-1,1)
    idxs = idxs.unsqueeze(2).repeat(1, 1, 10)



    for t in range(steps):
        out_adv_branch = model(x_adv)
        out = torch.stack([out_adv_branch[i] for i in range(7)], dim=1)
        out = torch.gather(out, 1, idxs).squeeze(1)
        loss_adv0 = loss_fn(out, y)
        grad0 = torch.autograd.grad(loss_adv0, x_adv, only_inputs=True)[0]
        x_adv.data.add_(gamma * torch.sign(grad0.data))
        linfball_proj(x, eps, x_adv, in_place=True)
        x_adv = torch.clamp(x_adv, 0, 1)
        

        
    return x_adv
'''
