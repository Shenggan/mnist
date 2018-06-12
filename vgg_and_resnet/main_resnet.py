'''Train MNIST with either supervised learning or virtual adversarial training in PyTorch.'''
from __future__ import print_function

import os

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from args_res import args
from logger import Logger
from mnistloader import *
import resnet
from torch.autograd import Variable
from utils import progress_bar

plt.ioff()


args = args()

logger = Logger(args.save_log)
use_cuda = torch.cuda.is_available()
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')

train_loader = get_loader_train(size=args.numlabels)
val_loader = get_loader_val(batch_size=48)

classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')

# Model
if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.t7')
    net = checkpoint['net']
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
else:
    print('==> Building model..')
    #net = vgg.__dict__[args.arch]()
    net = resnet.__dict__[args.arch]()

if use_cuda:
    net = torch.nn.DataParallel(net).cuda()
    cudnn.benchmark = True

optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 2 every 30 epochs"""
    lr = args.lr * (0.5 ** (epoch // args.low_lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
# Training


def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        if use_cuda:
           inputs, targets = inputs.cuda(), targets.cuda()
   
        optimizer.zero_grad()
        outputs = net.forward(inputs)

        dll_loss = nn.CrossEntropyLoss()(outputs, targets)

        if args.training == 'supervised':
            additional_loss = 0
        elif args.training == 'vat':
            vat_loss = vat.virtual_adversarial_loss(inputs, outputs, use_gpu=use_cuda)
            additional_loss = vat_loss

        loss = dll_loss + additional_loss

        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()
        train_acc = 100 * float(correct) / total

        progress_bar(batch_idx, len(train_loader)+1, 'Loss: %.5f | Acc: %.5f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100*float(correct)/total, correct, total))

        info = {
            'train_loss': train_loss,
            'train_accuracy': train_acc
        }

        for tag, value in info.items():
            logger.scalar_summary(tag, value, batch_idx+1)


# Testing
def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(val_loader):
        if use_cuda:
           inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        
        outputs = net.forward(inputs)

        dll_loss = nn.CrossEntropyLoss()(outputs, targets)

        if args.training == 'supervised':
            additional_loss = 0
        elif args.training == 'vat':
            vat_loss = vat.virtual_adversarial_loss(inputs, outputs, use_gpu=use_cuda)
            additional_loss = vat_loss

        loss = dll_loss + additional_loss

        test_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()
        test_acc = 100*float(correct)/total

        progress_bar(batch_idx, len(val_loader)+1, 'Loss: %.5f | Acc: %.5f%% (%d/%d)'
                     % (test_loss/(batch_idx+1), 100*float(correct)/total, correct, total))

        # Tensorboard logging
        info = {
            'test_loss': test_loss,
            'test_accuracy': test_acc
        }

        for tag, value in info.items():
            logger.scalar_summary(tag, value, batch_idx+1)

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.module if use_cuda else net,
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt_{0}_{1}.t7'.format(args.arch,args.numlabels))
        best_acc = acc


for epoch in range(start_epoch, start_epoch+200):
    adjust_learning_rate(optimizer, epoch)
    train(epoch)
    test(epoch)
