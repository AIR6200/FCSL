import argparse, json

parser = argparse.ArgumentParser(description='FCSL')
parser.add_argument('-c', '--conf', dest='conf')
parser.add_argument('-dt', '--data_type', dest='dt',default="cifar10")
parser.add_argument('-nd', '--num_dataclasses', dest='nd', type=int, default=10)
parser.add_argument('-tp', '--type', dest='tp',default="dirichlet")
parser.add_argument('-db', '--dirichlet_beta', dest='db',type=float,default=0.5)
parser.add_argument('-ac', '--anchor_client', dest='ac',type=int,default=3)
parser.add_argument('-ge', '--global_epochs', dest='ge',type=int,default=200)
parser.add_argument('-le', '--local_epochs', dest='le',type=int,default=1)
parser.add_argument('-nu', '--num_users', dest='nu',type=int,default=50)

args = parser.parse_args()
with open(args.conf, 'r') as f:
    conf = json.load(f)

conf["type"] = args.dt
conf["num_dataclasses"] = args.nd
conf["non_iid"]["type"] = args.tp
conf["non_iid"]["dirichlet_beta"] = args.db
conf["anchor_clients"] = args.ac
conf["global_epochs"] = args.ge
conf["local_epochs"] = args.le
conf["num_users"] = args.nu

with open(args.conf, 'w') as f:
    json.dump(conf,f)