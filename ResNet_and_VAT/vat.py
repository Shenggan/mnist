'''Functions for carrying out virtual adversarial training'''
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import model
from args import args
from torch.autograd import Variable
args = args()

def logit_function(x, use_gpu=True):
    net = model.CNNLarge()
    if use_gpu:
        net = net.cuda()
    return net.forward(x)

def _l2_normalize(d):

    if isinstance(d, Variable):
        d = d.data.numpy()
    elif isinstance(d, torch.Tensor):
        d = d.numpy()
    else:
        d = d[-1].data.cpu().numpy()

    d /= (np.sqrt(np.sum(d ** 2, axis=(1, 2, 3))).reshape((-1, 1, 1, 1)) + 1e-6)
    return torch.from_numpy(d)

def virtual_adversarial_loss(X, logit , xi=1e-6, eps=8.0, Ip=1, use_gpu=True):

    kl_div = nn.KLDivLoss()
    if use_gpu:
        kl_div.cuda()

    pred = logit_function(X, use_gpu)

    # prepare random unit tensor
    d = torch.rand(X.data.shape)
    d = Variable(_l2_normalize(d),requires_grad = True)
    if use_gpu:
        d = d.cuda()
    # calc adversarial direction
    for ip in range(Ip):
        pred_hat = logit_function(X + d / xi, use_gpu)
        adv_distance = kl_div(F.log_softmax(pred_hat, dim=1), pred.detach())
        kk = torch.autograd.grad(adv_distance, [d])
        d = Variable(_l2_normalize(kk))

    # calc LDS
    r_adv = d * args.epsilon
    X1 = X.data
    r_adv = r_adv.data
    if use_gpu:
        X1 = X1.cuda()
        r_adv = r_adv.cuda()
    pred_hat = logit_function(Variable(X1 + r_adv,requires_grad = True), use_gpu)
    pred = logit_function(X, use_gpu)
    LDS = kl_div(F.log_softmax(pred_hat, dim=1), pred.detach())
    return LDS

