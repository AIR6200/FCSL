import argparse, json
import copy
import math
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
from datetime import datetime
import torch.nn.functional as F


def select_clients_based_on_score(clients, client_weight_score):
    sorted_clients = sorted(
        clients,
        key=lambda c: c.prev_rate * client_weight_score[c.client_id],
        reverse=True
    )
    num_selected = max(1, int(len(sorted_clients) * 0.1))
    selected_clients = sorted_clients[:num_selected]
    return selected_clients

def select_clients_based_on_anchor(clients,client_weight_score,anchor_clients_1):
    client_scores = [
        (c.client_id, c.prev_rate * client_weight_score[c.client_id]) 
        for c in clients if c not in anchor_clients_1
    ]
    sorted_clients = sorted(client_scores, key=lambda x: x[1], reverse=True) 
    num_extra_selected = max(1, int(len(clients) * 0.05)) 
    extra_selected_clients = [client[0] for client in sorted_clients[:num_extra_selected]]
    selected_client_ids = [c.client_id for c in anchor_clients_1] + extra_selected_clients
    selected_clients = [c for c in clients if c.client_id in selected_client_ids]
    return selected_clients


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FCSL')
    parser.add_argument('-c', '--conf', dest='conf')
    args = parser.parse_args()

    with open(args.conf, 'r') as f:
        conf = json.load(f)
        print(conf)

    wandb.init(config=conf,
               name = "proposed FCSL, dirichlet_beta:" + str(conf['non_iid']["dirichlet_beta"]) + "anchor_clients:" + str(conf["anchor_clients"])+ "clients:"+ str(conf["num_users"]),
               project = "Result-"+conf["type"])


    ## Generate a client dataset that satisfies the dirichlet distribution
    tools.setup_seed(2020)
    train_datasets, eval_datasets, user_groups, train_data_cls_counts = partition_data(conf["type"],conf['non_iid']["type"],conf['non_iid']["dirichlet_beta"],conf["num_users"])
    test_loader = torch.utils.data.DataLoader(eval_datasets, batch_size=conf["batch_size"], shuffle=False,num_workers=4,pin_memory=True)

    clients = []
    distill_acc = []
    server = Server(conf)
    fedserver = Fed_server(conf)

    for c in range(conf["num_users"]):
        client_tran_dataset = datasplit.DatasetSplit(train_datasets,user_groups[c])
        clients.append(Client(conf,client_tran_dataset, c))

    print("Start time: ",datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("anchor clients selecting......")
    candidates = []
    anchor_clients_1 = []
    anchor_clients_2 = []
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
    anchor_clients_index_1 = tools.lookfor_maxindex(client_weight_score, int(0.05 * conf["num_users"]))
    anchor_clients_index_2 = tools.lookfor_maxindex(client_weight_score, int(0.1 * conf["num_users"]))
    print(client_weight_score)
    for c in clients:
        if c.client_id in anchor_clients_index_1:
            anchor_clients_1.append(c)
        if c.client_id in anchor_clients_index_2:
            anchor_clients_2.append(c)
        if c.client_id not in anchor_clients_index_2: 
            follower_clients.append(c)
    print("Anchor Groups 1:", [c.client_id for c in anchor_clients_1])
    print("Anchor Groups 2:", [c.client_id for c in anchor_clients_2])

    print("\n\n")

    for e in range(conf["global_epochs"]):
        print("\n\n")
        print("global epochs:", e)

        w_locals_front = []
        w_locals_back = []

        fedserver.global_front.eval()
        fedserver.global_back.eval()
        server.center.eval()
        total_loss = 0.0
        correct = 0
        dataset_size = 0

        for batch_id, batch in enumerate(test_loader):
            data, target = batch
            data.to(fedserver.device)
            target = target.to(fedserver.device)
            dataset_size += data.size()[0]
            output_front_eval = fedserver.global_front(data)
            output_center_eval = server.center(output_front_eval)
            output_back_eval = fedserver.global_back(output_center_eval)
            loss = torch.nn.functional.cross_entropy(output_back_eval, target, reduction='sum').item()
            pred = output_back_eval.data.max(1)[1]
            total_loss += loss
            correct += pred.eq(target).sum().item()
        acc = 100.0 * (float(correct) / float(dataset_size))
        total_l = total_loss / dataset_size
        distill_acc.append(acc)
        print("Time:%s, acc: %f, loss: %f\n" % (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), acc, total_l))


        for c in clients:
            c.adjust_client_prev_rate_PFSL(e, fedserver.global_front, server.center, fedserver.global_back)
        if e == 0:
            selected_clients = anchor_clients_2
        elif e % 5 == 0:
            selected_clients = select_clients_based_on_anchor(clients, client_weight_score,anchor_clients_1)

        client_weight_score_corse_learning = []


        for c in selected_clients:

            for name, param in fedserver.global_front.state_dict().items():
                c.local_front.state_dict()[name].copy_(param.clone())
            for name, param in fedserver.global_back.state_dict().items():
                c.local_back.state_dict()[name].copy_(param.clone())

            if e > int(0.1*conf["global_epochs"]):
                optimizer_front = torch.optim.SGD(c.local_front.parameters(), lr=c.conf['lr'] * c.prev_rate,momentum=c.conf['momentum'])
                optimizer_back = torch.optim.SGD(c.local_back.parameters(), lr=c.conf['lr'] * c.prev_rate,momentum=c.conf['momentum'])
                optimizer_center = torch.optim.SGD(server.center.parameters(), lr=server.conf['lr'] * c.prev_rate,momentum=server.conf['momentum'])
                c.sampler.batch_size = 2 + int(conf["batch_size"] * (1/c.prev_rate))
            else:

                optimizer_front = torch.optim.SGD(c.local_front.parameters(), lr=c.conf['lr'],momentum=c.conf['momentum'])
                optimizer_back = torch.optim.SGD(c.local_back.parameters(), lr=c.conf['lr'],momentum=c.conf['momentum'])
                optimizer_center = torch.optim.SGD(server.center.parameters(), lr=server.conf['lr'],momentum=server.conf['momentum'])


            c.local_front.train()
            c.local_back.train()
            server.center.train()

            for e in range(conf["local_epochs"]):
                for batch_id, batch in enumerate(c.train_loader):
                    data, target = batch
                    data = data.to(c.device)  
                    target = target.to(c.device)

                    optimizer_front.zero_grad()
                    optimizer_center.zero_grad()
                    optimizer_back.zero_grad()

                    output_front = c.local_front(data)  
                    output_center = server.center(output_front)  
                    output_back = c.local_back(output_center)

                    loss = torch.nn.functional.cross_entropy(output_back, target)
                    loss.backward()

                    optimizer_back.step()
                    optimizer_center.step()
                    optimizer_front.step()

            client_weight_score_corse_learning.append(client_weight_score[c.client_id])
            w_locals_front.append(copy.deepcopy(c.local_front.state_dict()))
            w_locals_back.append(copy.deepcopy(c.local_back.state_dict()))


        if selected_clients:  
            total_weight = sum(client_weight_score_corse_learning)
            client_weight_score_corse_learning = [weight / total_weight for weight in client_weight_score_corse_learning]
            w_glob_front = server.aggregation(w_locals_front, client_weight_score_corse_learning)
            fedserver.global_front.load_state_dict(w_glob_front)
            w_glob_back = server.aggregation(w_locals_back, client_weight_score_corse_learning)
            fedserver.global_back.load_state_dict(w_glob_back)



    ys = []
    ys.append(distill_acc)
    wandb.log({"global_accuracy": wandb.plot.line_series(
        xs=[i for i in range(conf["global_epochs"])],
        ys=ys,
        title="Test Accuracy")})

    wandb.finish()
