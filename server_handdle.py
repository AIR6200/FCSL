import os
import pickle
import models, torch
from criterion import *

class Server(object):

    def __init__(self, conf):

        self.conf = conf
        self.local_model = models.get_model(self.conf["server_model_name"],num_dataclasses=self.conf["num_dataclasses"])
        self.global_model = models.get_model(self.conf["server_model_name"],num_dataclasses=self.conf["num_dataclasses"])
        self.loss = NTD_Loss()
        self.loss.num_classes = conf["num_dataclasses"]
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
