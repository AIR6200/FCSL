import numpy as np
import torch
from torchvision import datasets, transforms
import pandas as pd
import torch.utils.data as Data


def get_dataset(dir, name):
    if name == 'mnist':
        train_dataset = datasets.MNIST(dir, train=True, download=True, transform=transforms.ToTensor())
        eval_dataset = datasets.MNIST(dir, train=False, transform=transforms.ToTensor())

    elif name == 'cifar10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        train_dataset = datasets.CIFAR10(dir, train=True, download=True,
                                         transform=transform_train)
        eval_dataset = datasets.CIFAR10(dir, train=False, transform=transform_test)

    elif name == "cifar100":
        train_dataset = datasets.CIFAR100(dir, train=True,download=True,
                                          transform=transforms.Compose(
                                              [
                                                  transforms.RandomCrop(32, padding=4),
                                                  transforms.RandomHorizontalFlip(),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                                       (0.2023, 0.1994, 0.2010)),
                                              ]))
        eval_dataset = datasets.CIFAR100(dir, train=False,download=True,
                                         transform=transforms.Compose([
                                             transforms.ToTensor(),
                                             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                         ]))

    elif name == 'fmnist':
        data_transformer = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        train_dataset = datasets.FashionMNIST('data', train=True, download=True, transform=data_transformer)
        eval_dataset = datasets.FashionMNIST('data', train=False, transform=data_transformer)

    if name == "svhn":
        X_train, y_train = train_dataset.data, train_dataset.targets
        X_test, y_test = eval_dataset.data, eval_dataset.targets
    else:
        X_train, y_train = train_dataset.data, train_dataset.targets
        X_test, y_test = eval_dataset.data, eval_dataset.targets
    if "cifar10" in name or name == "svhn":
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        X_test = np.array(X_test)
        y_test = np.array(y_test)
    else:
        X_train = X_train.data.numpy()
        y_train = y_train.data.numpy()
        X_test = X_test.data.numpy()
        y_test = y_test.data.numpy()

    return X_train, y_train, X_test, y_test, train_dataset, eval_dataset

def record_net_data_stats(y_train, net_dataidx_map):
    net_cls_counts = {}
    for net_i, dataidx in net_dataidx_map.items():
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        net_cls_counts[net_i] = tmp
    return net_cls_counts


def partition_data(dataset, partition, beta=0.5, num_users=10):
    n_parties = num_users
    X_train, y_train, X_test, y_test, train_dataset, test_dataset = get_dataset("./data",dataset)
    data_size = y_train.shape[0]
    if len(y_train) == 0 or len(y_test) == 0:
        raise ValueError("Dataset is empty. Please check the data loading process.")

    if partition == "iid":
        idxs = np.random.permutation(data_size)
        batch_idxs = np.array_split(idxs, n_parties)
        net_dataidx_map = {i: batch_idxs[i] for i in range(n_parties)}

    elif partition == "dirichlet":
        min_size = 0
        min_require_size = 10
        label = np.unique(y_test).shape[0]
        net_dataidx_map = {}

        while min_size < min_require_size:
            idx_batch = [[] for _ in range(n_parties)]
            for k in range(label):
                idx_k = np.where(y_train == k)[0]
                np.random.shuffle(idx_k)  # shuffle the label
                # random [0.5963643 , 0.03712018, 0.04907753, 0.1115522 , 0.2058858 ]
                proportions = np.random.dirichlet(np.repeat(beta, n_parties))
                proportions = np.array(   # 0 or x
                    [p * (len(idx_j) < data_size / n_parties) for p, idx_j in zip(proportions, idx_batch)])
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])

        for j in range(n_parties):
            np.random.shuffle(idx_batch[j])
            net_dataidx_map[j] = idx_batch[j]

    train_data_cls_counts = record_net_data_stats(y_train, net_dataidx_map)
    return train_dataset, test_dataset, net_dataidx_map, train_data_cls_counts