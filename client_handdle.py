import models, torch, copy
from torch.autograd import Variable
import numpy as np
from utils import tools

class Client(object):

    def __init__(self, conf, train_dataset,eval_dataset, id=-1):

        self.conf = conf
        self.local_model = models.get_model(self.conf["client_model_name"])
        self.global_model = models.get_model(self.conf["client_model_name"])
        self.client_id = id
        self.label_diversity = 0
        self.weight = []
        self.train_dataset = train_dataset
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=conf["batch_size"],shuffle=False)
        self.eval_loader = eval_dataset


    def label_diversity_Entropy(self):
        self.label_diversity = tools.Counter_And_Entropy(self.train_dataset)


    def client_weight(self):
        _,c = tools.label_counter(self.train_dataset)
        self.weight = [len(self.train_dataset),c,self.label_diversity]