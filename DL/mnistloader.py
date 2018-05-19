import torch
import torchvision
from torchvision import transforms
import numpy as np
from PIL import Image

class dataset1(torch.utils.data.Dataset):
    def __init__(self, train=1, size=1000, xform=None):
        if xform:
            self.xform = xform
        self.train = train
        self.size = size
        if self.train:
            self.data = np.fromfile("DATA/mnist_train/mnist_train_data", dtype=np.uint8)
            self.label = np.fromfile("DATA/mnist_train/mnist_train_label", dtype=np.uint8)
        else:
            self.data = np.fromfile("DATA/mnist_test/mnist_test_data", dtype=np.uint8)
            self.label = np.fromfile("DATA/mnist_test/mnist_test_label", dtype=np.uint8)

        if self.train:
            self.data = self.data.reshape(60000, 45, 45)
            self.data = self.data[:self.size]
        else:
            self.data = self.data.reshape(10000, 45, 45)

        self.len = len(self.data)
        print (self.len)

    def __len__(self):
        return self.len

    def __getitem__(self, ith):
        im = self.xform(Image.fromarray(self.data[ith]))
        tag = self.label[ith].astype('int64')
        return im, tag

xform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
    ])

transform_train = transforms.Compose([
    #transforms.RandomCrop(45, padding=4),
    #transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

def get_loader_train(batch_size=32, num_workers=8, size=60000, shuffle=True, pin_memory=False):
    return torch.utils.data.DataLoader(
        dataset1(size=size, xform=transform_train),
        batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=pin_memory)


def get_loader_val(train=0, batch_size=80, num_workers=8, size=10000, shuffle=True, pin_memory=False):
    return torch.utils.data.DataLoader(
        dataset1(train=0, size=size, xform=transform_test),
        batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=pin_memory)
