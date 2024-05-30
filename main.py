import argparse, json
import copy
from datasets import partition_data
import numpy
import torch, random
from server_handdle import *
from client_handdle import *
from fed_server import *
import models, datasets
from utils import tools,datasplit
from utils.topsis import Topsis
import wandb
from criterion import *


def FedAvg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg

def FedAdp(w,weight_list):
    w_adp = copy.deepcopy(w[0])
    for k in w_adp.keys():
        w_adp[k] = w_adp[k] * weight_list[0]
    for k in w_adp.keys():
        for j in range(1, len(w)):
            weight = weight_list[j]
            w[j][k] = weight * w[j][k]
            w_adp[k] = w[j][k] + w_adp[k]
    return w_adp


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FCSL')
    parser.add_argument('-c', '--conf', dest='conf')
    args = parser.parse_args()

    with open(args.conf, 'r') as f:
        conf = json.load(f)
        print(conf)

    wandb.init(config=conf,
               name = "proposed FCSL, dirichlet_beta:" + str(conf['non_iid']["dirichlet_beta"]) + "anchor_clients:" + str(conf["anchor_clients"] ),
               project = "Result-"+conf["type"])


    ## Generate a client dataset that satisfies the dirichlet distribution
    tools.setup_seed(0)
    train_datasets, eval_datasets, user_groups, train_data_cls_counts = partition_data(conf["type"],conf['non_iid']["type"],conf['non_iid']["dirichlet_beta"],conf["num_users"])
    test_loader = torch.utils.data.DataLoader(eval_datasets, batch_size=conf["batch_size"], shuffle=False,num_workers=4)

    clients = []
    distill_acc = []
    server = Server(conf)
    fedserver = Fed_server(conf)

    for c in range(conf["num_users"]):
        client_tran_dataset = datasplit.DatasetSplit(train_datasets,user_groups[c])
        clients.append(Client(conf,client_tran_dataset, test_loader, c))

    print(len(clients))
    print("anchor clients selecting......")
    candidates = []
    anchor_clients = []
    follower_clients = []

    client_weight_matrix = numpy.empty(shape=(0, 3))
    for c in clients:
        c.label_diversity_Entropy()
        c.client_weight()
        client_weight_matrix = numpy.append(client_weight_matrix, [c.weight], axis=0)
        candidates.append(c.label_diversity)
    t = Topsis(client_weight_matrix, minvalue=1)
    t.min_to_max()
    t.standard()
    t.score_compute()
    client_weight_score = t.score
    anchor_clients_index = tools.lookfor_maxindex(client_weight_score,conf["anchor_clients"])
    print(client_weight_score)
    print("anchor clients generated:",anchor_clients_index)

    for c in clients:
        if c.client_id in anchor_clients_index:
            anchor_clients.append(c)
        else:
            follower_clients.append(c)

    print("\n\n")


    for e in range(conf["global_epochs"]):
        print("\n\n")
        print("global epochs:", e)
        w_locals_server = []

        fedserver.client_global_model.eval()
        server.global_model.eval()
        total_loss = 0.0
        correct = 0
        dataset_size = 0

        for batch_id, batch in enumerate(test_loader):
            data, target = batch
            data.to(fedserver.device)
            target = target.to(fedserver.device)
            dataset_size += data.size()[0]
            output_eval_client = fedserver.client_global_model(data)
            output_eval_server = server.global_model(output_eval_client)
            loss = torch.nn.functional.cross_entropy(output_eval_server, target, reduction='sum').item()
            pred = output_eval_server.data.max(1)[1]
            total_loss += loss
            correct += pred.eq(target).sum().item()
        acc = 100.0 * (float(correct) / float(dataset_size))
        total_l = total_loss / dataset_size
        distill_acc.append(acc)
        print("acc: %f, loss: %f\n" % (acc, total_l))


        if e <= conf["anchor_epochs"]:

            for c in anchor_clients:
                for name, param in fedserver.client_global_model.state_dict().items():
                    c.local_model.state_dict()[name].copy_(param.clone())
                for name, param in server.global_model.state_dict().items():
                    server.local_model.state_dict()[name].copy_(param.clone())
                optimizer_client = torch.optim.SGD(c.local_model.parameters(), lr=c.conf['lr'],momentum=c.conf['momentum'])
                optimizer_server = torch.optim.SGD(server.local_model.parameters(), lr=server.conf['lr'],momentum=server.conf['momentum'])
                c.local_model.train()
                server.local_model.train()

                for e in range(conf["local_epochs"]):
                    for batch_id, batch in enumerate(c.train_loader):
                        data, target = batch
                        data.to(c.device)
                        target = target.to(c.device)

                        optimizer_client.zero_grad()
                        output_train_client = c.local_model(data).clone().detach().requires_grad_(True)

                        optimizer_server.zero_grad()
                        output_server = server.local_model(output_train_client)
                        dg_logits = server.global_model(output_train_client)
                        loss = server.loss(output_server, target, dg_logits)
                        loss.backward()
                        optimizer_server.step()

                        output_train_client.backward(output_train_client.grad.clone().detach())
                        optimizer_client.step()

                w_locals_server.append(copy.deepcopy(server.local_model.state_dict()))
                if c.client_id == max(anchor_clients_index):
                    fedserver.client_global_model = c.local_model

            w_glob_server = FedAvg(w_locals_server)
            server.global_model.load_state_dict(w_glob_server)

        else:
            for c in clients:
                for name, param in fedserver.client_global_model.state_dict().items():
                    c.local_model.state_dict()[name].copy_(param.clone())
                for name, param in server.global_model.state_dict().items():
                    server.local_model.state_dict()[name].copy_(param.clone())

                optimizer_client = torch.optim.SGD(c.local_model.parameters(), lr=c.conf['lr'],momentum=c.conf['momentum'])
                optimizer_server = torch.optim.SGD(server.local_model.parameters(), lr=server.conf['lr'],momentum=server.conf['momentum'])
                c.local_model.train()
                server.local_model.train()

                for e in range(conf["local_epochs"]):
                    for batch_id, batch in enumerate(c.train_loader):
                        data, target = batch
                        data.to(c.device)
                        target = target.to(c.device)

                        optimizer_client.zero_grad()
                        output_train_client = c.local_model(data).clone().detach().requires_grad_(True)

                        optimizer_server.zero_grad()
                        output_server = server.local_model(output_train_client)
                        dg_logits = server.global_model(output_train_client)
                        loss = server.loss(output_server, target, dg_logits)

                        loss.backward()
                        optimizer_server.step()

                        output_train_client.backward(output_train_client.grad.clone().detach())
                        optimizer_client.step()

                w_locals_server.append(copy.deepcopy(server.local_model.state_dict()))
                if c.client_id == conf["num_users"]:
                    fedserver.client_global_model = c.local_model

            w_glob_server = FedAdp(w_locals_server,client_weight_score)
            server.global_model.load_state_dict(w_glob_server)

    ys = []
    ys.append(distill_acc)
    wandb.log({"global_accuracy": wandb.plot.line_series(
        xs=[i for i in range(conf["global_epochs"])],
        ys=ys,
        title="Test Accuracy")})

    wandb.finish()