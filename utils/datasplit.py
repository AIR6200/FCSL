import numpy as np
import torch
from torch.utils.data import DataLoader,Dataset


class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


def record_net_data_stats(y_train, net_dataidx_map):
    net_cls_counts = {}
    for net_i, dataidx in net_dataidx_map.items():
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        net_cls_counts[net_i] = tmp
    return net_cls_counts


def partition_data(X_train, y_train, X_test, y_test, train_dataset, test_dataset, partition, beta=0.5, num_users=10):
    n_parties = num_users
    data_size = y_train.shape[0]

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

    elif partition == "niid_4":
        net_dataidx_map = {}
        idxs = np.arange(data_size)
        labels = train_dataset.targets
        idxs_labels = np.vstack((idxs, labels))
        idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
        idxs = idxs_labels[0, :]
        data_len = int(data_size / num_users)
        for id in range(n_parties):
            net_dataidx_map[id] = idxs[id * data_len:(id + 1) * data_len]

    train_data_cls_counts = record_net_data_stats(y_train, net_dataidx_map)
    return train_dataset, test_dataset, net_dataidx_map, train_data_cls_counts

def get_non_IID_data(Dataset, num_users, id):
    """
    Sample non-I.I.D client data from dataset
    :param dataset:
    :param num_users:
    :return:
    """
    idxs = np.arange(len(Dataset))
    labels = Dataset.targets
    # idxs and labels
    idxs_labels = np.vstack((idxs, labels))
    # sort labels
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]
    data_len = int(len(Dataset) / num_users)
    return idxs[id * data_len:(id + 1) * data_len]

def get_iid_client(train_set,name):
    dataset = []
    train_dataset = train_set
    #cifar100:i in range(0,100),train_indices = get_non_IID_data(train_set, 100, i)1
    #cifar10/FMNIST:i in range(0,10),train_indices = get_non_IID_data(train_set, 100, 10*i)
    if name == "fmnist" or "cifar10":
        for i in range(0, 10):
            train_indices = get_non_IID_data(train_set, 100, 10 * i)
            train_dataset_new = torch.utils.data.Subset(train_set, train_indices)
            dataset.append(train_dataset_new)
            train_dataset = torch.utils.data.ConcatDataset(dataset)
    elif name == 'cifar100':
        for i in range(0, 100):
            train_indices = get_non_IID_data(train_set, 100, i)
            train_dataset_new = torch.utils.data.Subset(train_set, train_indices)
            dataset.append(train_dataset_new)
            train_dataset = torch.utils.data.ConcatDataset(dataset)
    return train_dataset