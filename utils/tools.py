import math
from collections import Counter
import random
import torch.backends.cudnn as cudnn
import numpy as np
import torch
import torch.nn.functional as F


def calculate_median(arr):
    sorted_arr = sorted(arr)
    mid_index = len(arr) // 2
    if len(arr) % 2 == 0:
        return (sorted_arr[mid_index - 1] + sorted_arr[mid_index]) / 2
    else:
        return sorted_arr[mid_index]


def Counter_And_Entropy(data_set):
    target_list = []
    for i in range(len(data_set)):
        _, target = data_set[i]
        target_list.append(target)
    counts = len(target_list)
    counter = Counter(target_list)
    prob = {i[0]: i[1] / counts for i in counter.items()}
    H = - sum([i[1] * math.log2(i[1]) for i in prob.items()])
    return H

def lookfor_maxindex(N,k):
    fdict = {}
    for a in range(len(N)):
        fdict[a] = N[a]
    fsort = dict(sorted(fdict.items(), key=lambda x: x[1], reverse=True))
    max_index = list(fsort.keys())[0:k]
    return max_index


def refine_as_not_true(logits, targets, num_classes):
    nt_positions = torch.arange(0, num_classes).to(logits.device)
    nt_positions = nt_positions.repeat(logits.size(0), 1)
    nt_positions = nt_positions[nt_positions[:, :] != targets.view(-1, 1)]
    nt_positions = nt_positions.view(-1, num_classes - 1)
    logits = torch.gather(logits, 1, nt_positions)
    return logits


def label_counter(data_set):
    variance = 0
    target_list = []
    for i in range(len(data_set)):
        _, target = data_set[i]
        target_list.append(target)
    dict = Counter(target_list)
    mean = 1 / len(target_list)
    for key in dict:
        value = dict[key]/len(target_list)
        s = np.square(value-mean)
        variance += s
    return dict,variance


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.deterministic = True

# 计算本地数据分布P_data
def get_local_data_distribution(train_dataset, num_classes):
    label_counts, _ = label_counter(train_dataset)
    # 将类别计数转换为 numpy 数组
    class_counts = np.zeros(num_classes)
    for label, count in label_counts.items():
        class_counts[label] = count
    # 计算总样本数
    total_samples = sum(label_counts.values())
    # 返回每个类别的比例
    return class_counts / total_samples


# 计算全局模型输出分布P_model
def get_model_output_distribution(client_model, server_model, train_loader, device):
    client_model.eval()
    server_model.eval()
    # 获取server-model输出的维度
    try:
        output_dim = server_model.fc.out_features  # 假设server-model的最后一层是全连接层 (fc):RESNET18可用
    except:
        #output_dim = server_model.block3[-1].out_features #lenet可用
        output_dim = server_model.back[-1].out_features
    softmax_outputs = np.zeros((len(train_loader.dataset), output_dim))  # 存储softmax输出
    idx = 0
    with torch.no_grad():
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            # 先通过client-model
            client_output = client_model(data)
            # 然后通过server-model
            server_output = server_model(client_output)
            # 使用softmax获取概率分布
            softmax_outputs[idx:idx + len(target)] = F.softmax(server_output, dim=1).cpu().numpy()
            idx += len(target)
    # 计算每个类别的平均概率
    return np.mean(softmax_outputs, axis=0)


def get_model_output_distribution_PFSL(front_model, center_model, back_model, train_loader, device):
    front_model.eval()
    center_model.eval()
    back_model.eval()

    # 获取 back_model 输出的维度
    try:
        output_dim = back_model.back_model[-1].out_features  # ResNet可用
    except:
        output_dim = back_model.block3[-1].out_features #lenet可用
    
    softmax_outputs = np.zeros((len(train_loader.dataset), output_dim))  # 存储 softmax 输出
    idx = 0

    with torch.no_grad():
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            # 先通过 front_model
            front_output = front_model(data)
            # 然后通过 center_model
            center_output = center_model(front_output)
            # 最后通过 back_model
            back_output = back_model(center_output)
            # 使用 softmax 获取概率分布
            softmax_outputs[idx:idx + len(target)] = F.softmax(back_output, dim=1).cpu().numpy()
            idx += len(target)

    # 计算每个类别的平均概率
    return np.mean(softmax_outputs, axis=0)

# 计算KL散度
def kl_divergence(p, q):
    p = np.clip(p, 1e-10, 1)
    q = np.clip(q, 1e-10, 1)
    return np.sum(p * np.log(p / q))

# 计算JS散度
def js_divergence(p_data, p_model):
    m = 0.5 * (p_data + p_model)
    return 0.5 * (kl_divergence(p_data, m) + kl_divergence(p_model, m))


