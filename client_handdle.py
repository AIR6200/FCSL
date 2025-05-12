import models, torch, copy
from torch.autograd import Variable
import numpy as np
from utils import tools
from torch.utils.data import DataLoader, RandomSampler, BatchSampler

class Client(object):

    def __init__(self, conf, train_dataset, id=-1):

        self.conf = conf
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.local_model = models.get_model(self.conf["client_model_name"])

        # only use in PFSL
        self.local_front = models.get_model("ResNet18_PFSL_front")
        self.local_back = models.get_model("ResNet18_PFSL_back")
        #self.local_front = models.get_model("LeNet_PFSL_front")
        #self.local_back = models.get_model("LeNet_PFSL_back")

        # only use in MSFL
        self.neighbor_paras = []

        self.global_model = models.get_model(self.conf["client_model_name"])
        self.client_id = id
        self.label_diversity = 0
        self.weight = []
        self.selected_at_epoch = -1
        self.prev_rate = 1
        self.train_dataset = train_dataset
        self.p_data = tools.get_local_data_distribution(self.train_dataset, num_classes=self.conf['num_dataclasses'])
        #self.test_loader = test_loader

        def collate_fn(batch):
            data, targets = zip(*batch)
            return torch.stack(data).to(self.device), torch.tensor(targets).to(self.device)
        self.sampler = BatchSampler(RandomSampler(self.train_dataset), self.conf["batch_size"], drop_last=True)
        self.train_loader = DataLoader(self.train_dataset, batch_sampler=self.sampler, collate_fn=collate_fn)
        #self.train_loader = DataLoader(self.train_dataset, batch_sampler=self.sampler)
        self.optimizer = torch.optim.SGD(self.local_model.parameters(), lr=self.conf['lr'], momentum=self.conf['momentum'], weight_decay=5e-4)


    def label_diversity_Entropy(self):
        self.label_diversity = tools.Counter_And_Entropy(self.train_dataset)


    def client_weight(self):
        _,c = tools.label_counter(self.train_dataset)
        self.weight = [len(self.train_dataset),c,self.label_diversity]

    def adjust_client_prev_rate(self,global_round,client_global_model,server_global_model):
        p_model = tools.get_model_output_distribution(client_global_model, server_global_model, self.train_loader, device=self.device)
        current_rate = tools.js_divergence(self.p_data, p_model)
        # 对 rate 进行平滑处理
        t = global_round + 1  # 使用参数 e 作为当前轮次
        if t == 1:  # 第一轮不需要平滑
            rate = current_rate
        else:
            rate = (t - 1) / t * self.prev_rate + (1 / t) * current_rate
        # 更新客户端的 prev_rate 为当前轮的 rate
        self.prev_rate = rate

    def adjust_client_prev_rate_PFSL(self,global_round,front_model,center_model,back_model):
        p_model = tools.get_model_output_distribution_PFSL(front_model,center_model,back_model, self.train_loader, device=self.device)
        current_rate = tools.js_divergence(self.p_data, p_model)
        # 对 rate 进行平滑处理
        t = global_round + 1  # 使用参数 e 作为当前轮次
        if t == 1:  # 第一轮不需要平滑
            rate = current_rate
        else:
            rate = (t - 1) / t * self.prev_rate + (1 / t) * current_rate
        # 更新客户端的 prev_rate 为当前轮的 rate
        self.prev_rate = rate

    def adjust_local_training(self):
        #self.sampler.batch_size = 2 + int(self.conf["batch_size"] * self.prev_rate)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.conf['lr'] * self.prev_rate

