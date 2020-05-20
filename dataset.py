from torchvision.datasets import CIFAR10
from torchvision import transforms as T
from sklearn.model_selection import train_test_split
import logging
import torch
from torch.utils.data import sampler

class Sampler(sampler.Sampler):
    def __init__(self, data_source, epoch):
        self.data_source = data_source
        self.epoch = epoch


    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.epoch)
        indices = torch.randperm(len(self.data_source), generator=g).tolist()

        return iter(indices)


    def __len__(self):
        return len(self.data_source)





class cifar10(CIFAR10):
    def __init__(self, root, valid=True, train=True, transform=None, target_transform=None,
                 download=True, p=0.1, seed=0):
        super(cifar10, self).__init__(root, train, transform, target_transform,
                 download)


        name = 'Test dataset'
        if train:
            name = 'Train dataset'
            self.split(p, valid, seed)
        if valid:
            name = 'Valid dataset'

        print(f'{name} has {len(self)} samples!')
    def split(self, p, valid, seed):
        X_train, X_test, y_train, y_test = train_test_split(self.data, self.targets, test_size = p, random_state=seed, stratify=self.targets)

        if valid:
            self.data, self.targets = X_test, y_test
        else:
            self.data, self.targets = X_train, y_train



def get_datasets():


    train_transform = T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomCrop(32, 4),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    valid_transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = cifar10('./data/', valid=False, train=True, transform=train_transform)
    valid_dataset = cifar10('./data/', valid=True, train=True, transform=valid_transform)
    test_dataset = cifar10('./data/', valid=False, train=False, transform=valid_transform)


    return {'train':train_dataset, 'valid':valid_dataset, 'test':test_dataset}