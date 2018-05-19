'''CNN-Large from Miyato et. al. 'Virtual Adversarial Training: a Regularization Method for Supervised and Semi-supervised Learning' in PyTorch'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from args import args
from torch.autograd import Variable

args = args()

class FC(nn.Module):

    def __init__(self):
        super(FC, self).__init__()

        self.classifier = nn.Sequential(
            nn.Linear(2025, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 10)
        )

    def forward(self, x):

        x = x.view(x.size(0),-1)
        x = self.classifier(x)

        return x

class CNNLarge(nn.Module):

    def __init__(self):
        super(CNNLarge, self).__init__()
        self.conv1 = nn.Conv2d(1, 128, kernel_size=3)
        self.batchnorm1 = nn.BatchNorm2d(128)
        self.lrelu1 = nn.LeakyReLU(negative_slope=args.lrelu_a)
        self.conv2 = nn.Conv2d(128, 128, kernel_size=3)
        self.batchnorm2 = nn.BatchNorm2d(128)
        self.lrelu2 = nn.LeakyReLU(negative_slope=args.lrelu_a)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3)
        self.batchnorm3 = nn.BatchNorm2d(128)
        self.lrelu3 = nn.LeakyReLU(negative_slope=args.lrelu_a)

        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout2d(p=args.keep_prob_hidden)

        self.conv4 = nn.Conv2d(128, 256, kernel_size=3)
        self.batchnorm4 = nn.BatchNorm2d(256)
        self.lrelu4 = nn.LeakyReLU(negative_slope=args.lrelu_a)
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3)
        self.batchnorm5 = nn.BatchNorm2d(256)
        self.lrelu5 = nn.LeakyReLU(negative_slope=args.lrelu_a)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3)
        self.batchnorm6 = nn.BatchNorm2d(256)
        self.lrelu6 = nn.LeakyReLU(negative_slope=args.lrelu_a)

        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout2 = nn.Dropout2d(p=args.keep_prob_hidden)

        self.conv7 = nn.Conv2d(256, 512, kernel_size=3)
        self.batchnorm7 = nn.BatchNorm2d(512)
        self.lrelu7 = nn.LeakyReLU(negative_slope=args.lrelu_a)

        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout3 = nn.Dropout2d(p=args.keep_prob_hidden)

        self.conv8 = nn.Conv2d(512, 256, kernel_size=1)
        self.batchnorm8 = nn.BatchNorm2d(256)
        self.lrelu8 = nn.LeakyReLU(negative_slope=args.lrelu_a)
        self.conv9 = nn.Conv2d(256, 128, kernel_size=1)
        self.batchnorm9 = nn.BatchNorm2d(128)
        self.lrelu9 = nn.LeakyReLU(negative_slope=args.lrelu_a)

        self.globalpool = nn.AdaptiveAvgPool2d((0, 0))
        self.fc = nn.Linear(2048, 10)
        self.batchnormfc = nn.BatchNorm2d(10)

    def forward(self, x):

        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.lrelu1(x)
        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = self.lrelu2(x)
        x = self.conv3(x)
        x = self.batchnorm3(x)
        x = self.lrelu3(x)

        x = self.pool1(x)
        x = self.dropout1(x)

        x = self.conv4(x)
        x = self.batchnorm4(x)
        x = self.lrelu4(x)
        x = self.conv5(x)
        x = self.batchnorm5(x)
        x = self.lrelu5(x)
        x = self.conv6(x)
        x = self.batchnorm6(x)
        x = self.lrelu6(x)

        x = self.pool2(x)
        x = self.dropout2(x)

        x = self.conv7(x)
        x = self.batchnorm7(x)
        x = self.lrelu7(x)

        x = self.conv8(x)
        x = self.batchnorm8(x)
        x = self.lrelu8(x)
        x = self.conv9(x)
        x = self.batchnorm9(x)
        x = self.lrelu9(x)

        x = self.globalpool(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)

        if args.top_bn:
            x = self.batchnormfc(x)
        return x
