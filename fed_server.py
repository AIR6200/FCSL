import os
import pickle
import models, torch

class Fed_server(object):

    def __init__(self, conf):

        self.conf = conf
        self.local_model = models.get_model(self.conf["client_model_name"])#
        self.client_global_model = models.get_model(self.conf["client_model_name"])

        # only use in PFSL
        self.global_front = models.get_model("ResNet18_PFSL_front")
        self.global_back = models.get_model("ResNet18_PFSL_back")
        #self.global_front = models.get_model("LeNet_PFSL_front")
        #self.global_back = models.get_model("LeNet_PFSL_back")

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')