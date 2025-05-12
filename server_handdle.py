import os
import pickle
import models, torch
from criterion import *
import copy

class Server(object):

    def __init__(self, conf):

        self.conf = conf
        self.local_model = models.get_model(self.conf["server_model_name"],num_dataclasses=self.conf["num_dataclasses"])
        self.global_model = models.get_model(self.conf["server_model_name"],num_dataclasses=self.conf["num_dataclasses"])

        # only use in PFSL
        self.center = models.get_model("ResNet18_PFSL_center")
        #self.center = models.get_model("LeNet_PFSL_center")

        self.loss = NTD_Loss()
        self.loss.num_classes = conf["num_dataclasses"]
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.optimizer = torch.optim.SGD(self.local_model.parameters(), lr=self.conf['lr'], momentum=self.conf['momentum'], weight_decay=5e-4)

    def aggregation(self,w, weight_list):
        w_adp = copy.deepcopy(w[0])
        for k in w_adp.keys():
            w_adp[k] = w_adp[k] * weight_list[0]
        for k in w_adp.keys():
            for j in range(1, len(w)):
                weight = weight_list[j]
                w[j][k] = weight * w[j][k]
                w_adp[k] = w[j][k] + w_adp[k]
        return w_adp

    def adjust_optimizer(self,rate):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.conf['lr'] * rate
