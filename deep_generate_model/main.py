import sys
from itertools import cycle, repeat

import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
from torch.autograd import Variable

from inference import SVI, DeterministicWarmup, ImportanceWeightedSampler
from models import LadderDeepGenerativeModel


class MNIST2(data.Dataset):
    def __init__(self, root, train=True, transform=None, target_transform=None):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set

        if self.train:
            self.train_data = np.fromfile(
                "../mnist/mnist_train/mnist_train_data", dtype=np.uint8)
            self.train_data = torch.from_numpy(
                np.reshape(self.train_data, (-1, 45, 45)))
            self.train_labels = torch.from_numpy(np.fromfile(
                "DATA/mnist_train/mnist_train_label", dtype=np.uint8))

        else:
            self.test_data = np.fromfile(
                "../mnist/mnist_test/mnist_test_data", dtype=np.uint8)
            self.test_data = torch.from_numpy(
                np.reshape(self.test_data, (-1, 45, 45)))
            self.test_labels = torch.from_numpy(np.fromfile(
                "DATA/mnist_test/mnist_test_label", dtype=np.uint8))

    def __getitem__(self, index):
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')
        img = img.resize((28, 28))

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        if self.train:
            return 60000
        else:
            return 10000


cuda = torch.cuda.is_available()
print("CUDA: {}".format(cuda))


def binary_cross_entropy(r, x):
    "Drop in replacement until PyTorch adds `reduce` keyword."
    return -torch.sum(x * torch.log(r + 1e-8) + (1 - x) * torch.log(1 - r + 1e-8), dim=-1)


n_labels = 10


def get_mnist(location="./DATA", batch_size=64, labels_per_class=100):
    from functools import reduce
    from operator import __or__
    from torch.utils.data.sampler import SubsetRandomSampler
    from torchvision.datasets import MNIST
    import torchvision.transforms as transforms
    from utils import onehot

    def flatten_bernoulli(x): 
        return transforms.ToTensor()(x).view(-1).bernoulli()

    mnist_train = MNIST2(location, train=True,
                         transform=flatten_bernoulli, target_transform=onehot(n_labels))
    mnist_valid = MNIST2(location, train=False,
                         transform=flatten_bernoulli, target_transform=onehot(n_labels))

    def get_sampler(labels, n=None):
        # Only choose digits in n_labels
        (indices,) = np.where(
            reduce(__or__, [labels == i for i in np.arange(n_labels)]))

        # Ensure uniform distribution of labels
        np.random.shuffle(indices)
        indices = np.hstack([list(filter(lambda idx: labels[idx] == i, indices))[
                            :n] for i in range(n_labels)])

        indices = torch.from_numpy(indices)
        sampler = SubsetRandomSampler(indices)
        return sampler

    # Dataloaders for MNIST
    labelled = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, num_workers=2, pin_memory=cuda,
                                           sampler=get_sampler(mnist_train.train_labels.numpy(), labels_per_class))
    unlabelled = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, num_workers=2, pin_memory=cuda,
                                             sampler=get_sampler(mnist_train.train_labels.numpy()))
    validation = torch.utils.data.DataLoader(mnist_valid, batch_size=batch_size, num_workers=2, pin_memory=cuda,
                                             sampler=get_sampler(mnist_valid.test_labels.numpy()))

    return labelled, unlabelled, validation


if __name__ == "__main__":

    labelled, unlabelled, validation = get_mnist(
        location="./", batch_size=100, labels_per_class=10)
    alpha = 0.1 * len(unlabelled) / len(labelled)

    model = LadderDeepGenerativeModel(
        [784, n_labels, [32, 16, 8], [128, 128, 128]])

    if cuda:
        model = model.cuda()

    beta = DeterministicWarmup(n=4*len(unlabelled)*100)
    sampler = ImportanceWeightedSampler(mc=1, iw=1)

    elbo = SVI(model, likelihood=binary_cross_entropy,
               beta=beta, sampler=sampler)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=5e-5, betas=(0.9, 0.999))

    epochs = 251
    best = 0.0

    file = open(model.__class__.__name__ + ".log", 'w+')

    for epoch in range(epochs):
        model.train()
        total_loss, labelled_loss, unlabelled_loss, accuracy = (0, 0, 0, 0)
        for (x, y), (u, _) in zip(cycle(labelled), unlabelled):
            if cuda:
                # They need to be on the same device and be synchronized.
                x, y = x.cuda(), y.cuda()
                u = u.cuda()

            L = -elbo(x, y)
            U = -elbo(u)

            # Add auxiliary classification loss q(y|x)
            logits = model.classify(x)
            classication_loss = torch.sum(
                y * torch.log(logits + 1e-8), dim=1).mean()

            J_alpha = L - alpha * classication_loss + U

            J_alpha.backward()
            optimizer.step()
            optimizer.zero_grad()

            total_loss += J_alpha.item()
            labelled_loss += L.item()
            unlabelled_loss += U.item()

            _, pred_idx = torch.max(logits, 1)
            _, lab_idx = torch.max(y, 1)
            accuracy += torch.mean((pred_idx.data == lab_idx.data).float())

        m = len(unlabelled)
        print(*(total_loss / m, labelled_loss / m, unlabelled_loss /
                m, accuracy / m), sep="\t", file=file)

        if epoch % 1 == 0:
            model.eval()
            print("Epoch: {}".format(epoch))
            print("[Train]\t\t J_a: {:.6f}, L: {:.6f}, U: {:.6f}, accuracy: {:.6f}".format(total_loss / m,
                                                                                           labelled_loss / m,
                                                                                           unlabelled_loss / m,
                                                                                           accuracy / m))

            total_loss, labelled_loss, unlabelled_loss, accuracy = (0, 0, 0, 0)
            for x, y in validation:
                # x, y = Variable(x), Variable(y)

                if cuda:
                    x, y = x.cuda(), y.cuda()

                L = -elbo(x, y)
                U = -elbo(x)

                logits = model.classify(x)
                classication_loss = - \
                    torch.sum(y * torch.log(logits + 1e-8), dim=1).mean()

                J_alpha = L + alpha * classication_loss + U

                total_loss += J_alpha.item()
                labelled_loss += L.item()
                unlabelled_loss += U.item()

                _, pred_idx = torch.max(logits, 1)
                _, lab_idx = torch.max(y, 1)
                accuracy += torch.mean((pred_idx.data == lab_idx.data).float())

            m = len(validation)
            print(*(total_loss / m, labelled_loss / m,
                    unlabelled_loss / m, accuracy / m), sep="\t", file=file)
            print("[Validation]\t J_a: {:.6f}, L: {:.6f}, U: {:.6f}, accuracy: {:.6f}".format(total_loss / m,
                                                                                              labelled_loss / m,
                                                                                              unlabelled_loss / m,
                                                                                              accuracy / m))

        if accuracy > best:
            best = accuracy
            torch.save(model, '{}.pt'.format(model.__class__.__name__))
