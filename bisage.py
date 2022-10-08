# -----------------------------------------------------------
# BISAGE module for GEM
# 
# September-November 2021
# -----------------------------------------------------------
import random
import time
import torch
from torch import Tensor, transpose, matmul, sigmoid, diag
from torch.nn import Parameter
import torch.nn.functional as F
from torch.optim import Adam, SGD
from torch_sparse import SparseTensor
from torch_geometric.data import Data
from torch_geometric.typing import OptTensor, OptPairTensor, Adj, Size
from torch_geometric.utils import from_networkx, negative_sampling
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear

import networkx as nx
from typing import Union, Tuple

from utils import *
from config import emb_config

random.seed(0)

runs = 1
epochs = 2000
lr = 0.01
weight_decay = 5e-5
early_stopping = 0
hidden = 64
dropout = 0.1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

root_base = os.getcwd()
emb_dim = emb_config["dim"]
num_hidden = 0
dir_name = emb_config["dir_name"]
lr = emb_config["lr"]

class LossFunc(torch.nn.Module):
    def __init__(self, weight=None, size_average=None):
        super(LossFunc, self).__init__()

    def forward(self, output, target):
        loss_m = matmul(output, transpose(target, 0, 1))
        loss_m -= torch.diag(loss_m)
        loss = torch.mean(sigmoid(F.normalize(loss_m, dim=0)))
        # print(sigmoid(F.normalize(loss_m, dim=0)))
        # loss = 0
        # loss -= sum(matmul(output[0], transpose(target[1:], 0, 1)))
        # loss -= sum(matmul(target[0], transpose(output[1:], 0, 1)))
        return loss

class BiSAGEConv(MessagePassing):
    def __init__(self, in_channels: Union[int, Tuple[int, int]], out_channels: int, aggr: str="mean", normalize: bool=True, root_weight: bool = True, bias: bool=True, **kwargs):
        """ BiSAGEConv layer, modified from torch_geometric's implementation of SAGEConv

        Args:
            in_channels (int or tuple): size of each input sample, or -1 to derive the size from the first input(s) to the forward method, 
                                        or tuple corresponds to the sizes of source and target
            dimensionalities
            out_channels (int): size of each output sample
            aggr (str, optional): aggregation scheme, default: "mean"
            normalize(bool, optional): whether the output would be normalized, default: True
            root_weight(bool, optional): whether the layer would add transforme root node feature to the output, default: True
            bias (bool, optional): whether the layer would learn an additive bias, default: True
            **kwargs (optional): additional arguments
        """
        super().__init__(aggr=aggr, **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize
        self.root_weight = root_weight

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        self.lin_l = Linear(in_channels[0], out_channels, bias=bias)
        if self.root_weight:
            self.lin_r = Linear(in_channels[1], out_channels, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_l.reset_parameters()
        if self.root_weight:
            self.lin_r.reset_parameters()

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj, edge_weight: OptTensor = None, size: Size = None) -> Tensor:
        if isinstance(x, Tensor):
            (h, l) = torch.split(x, int(x.size()[0]/2))
            h: OptPairTensor = (h, h)
            l: OptPairTensor = (l, l)

        # propagate_type: (x: OptPairTensor - h or l, edge_weight: OptTensor)
        h_out = self.propagate(edge_index, x=l, edge_weight=edge_weight, size=size)
        h_out = self.lin_l(h_out)
        l_out = self.propagate(edge_index, x=h, edge_weight=edge_weight, size=size)
        l_out = self.lin_l(l_out)

        h_r = h[1]
        l_r = l[1]
        if h_r is not None:
            h_out += self.lin_r(h_r)
        if l_r is not None:
            l_out += self.lin_r(l_r)

        return torch.cat((h_out, l_out), 0)

    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor,
                              x: OptPairTensor) -> Tensor:
        return torch_sparse.matmul(adj_t, x[0], reduce=self.aggr)

class BiSAGE(torch.nn.Module):
    def __init__(self, dataset, aggr='mean'):
        super(BiSAGE, self).__init__()
        self.num_input_features = emb_dim
        self.num_hidden_features = emb_dim
        self.num_output_featrues = emb_dim
        self.num_hiddent_layers = num_hidden
        self.conv1 = BiSAGEConv(self.num_input_features, self.num_hidden_features)
        self.conv2 = BiSAGEConv(self.num_hidden_features, self.num_hidden_features)
        self.conv3 = BiSAGEConv(self.num_hidden_features, self.num_output_featrues)

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight

        if self.num_hiddent_layers==0:
            x = self.conv1(x, edge_index, edge_weight)
            x = F.relu(x)
            x = F.dropout(x, training=self.training)
            x = self.conv3(x, edge_index, edge_weight)
        else:
            x = self.conv1(x, edge_index, edge_weight)
            x = F.relu(x)
            x = F.dropout(x, training=self.training)
            for layer in range(self.num_hiddent_layers):
                x = self.conv2(x, edge_index, edge_weight)
                x = F.relu(x)
                x = F.dropout(x, training=self.training)
            x = self.conv3(x, edge_index, edge_weight)

        return F.log_softmax(x, dim=1)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.conv3.reset_parameters()

def get_data(edgelist_file):
    """ Create torch_geometric.data.Data formatted data using the edgelist file

    Args:
        edgelist_file (string): path of the input edgelist file
    """
    x_set = set()
    edge_index_array = [[],[]]
    edge_weight_array = []

    f_in = open(edgelist_file, "r")
    lines = f_in.readlines()

    # x_set to list
    for line in lines:
        x_set.add(line.split()[0])
        x_set.add(line.split()[1])
    x_list = list(x_set)

    # x
    x_array = []
    for cur_x in x_list:
        new_x = -1 if cur_x.startswith("i") else 1
        x_array.append([random.uniform(0, 1)*new_x for _ in range(emb_dim)])

    # edge_index and edge_weight
    edge_index_array = [[],[]]
    edge_weight_array = []
    for line in lines:
        edge_index_array[0].extend([x_list.index(line.split()[0]), x_list.index(line.split()[1])])
        edge_index_array[1].extend([x_list.index(line.split()[1]), x_list.index(line.split()[0])])
        edge_weight_array.extend([float(line.split()[2]), float(line.split()[2])])

    f_in.close()

    x = torch.tensor([x_array, x_array], dtype=torch.float)
    edge_index = torch.tensor(edge_index_array, dtype=torch.long)
    edge_weight = torch.tensor(edge_weight_array, dtype=torch.float)

    return x_list, Data(x=x, edge_index=edge_index, edge_weight=edge_weight)

def run(x_list, dataset, model, runs, epochs, lr, weight_decay, early_stopping, test_x_list, test_dataset, out_file1, out_file2):
    val_losses, durations = [], []
    for _ in range(runs):
        data = dataset
        data = data.to(device)

        model.to(device).reset_parameters()
        optimizer = SGD(model.parameters(), lr=lr, weight_decay=weight_decay)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t_start = time.perf_counter()

        best_loss = float('inf')
        loss_history = []

        for epoch in range(1, epochs + 1):
            train(model, optimizer, data)
            eval_info = evaluate(model, data)
            eval_info["epoch"] = epoch

            if eval_info["loss"] < best_loss:
                best_loss = eval_info["loss"]

            loss_history.append(eval_info["loss"])
            if early_stopping > 0 and epoch > epochs // 2:
                tmp = Tensor(loss_history[-(early_stopping + 1):-1])
                if eval_info["loss"] > tmp.mean().item():
                    break
            print(eval_info["loss"])

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t_end = time.perf_counter()

        val_losses.append(best_loss)
        durations.append(t_end - t_start)

    loss, duration = Tensor(val_losses), Tensor(durations)

    out = model(dataset)
    cur_out, _ = torch.split(out, int(out.size()[0]/2))
    cur_out = cur_out.reshape((cur_out.size()[1], cur_out.size()[2]))
    cur_out = F.normalize(cur_out, dim=0)

    f_out = open(out_file1, "w")
    f_out.write("{} {}\n".format(cur_out.size()[0], emb_dim))
    for i in range(cur_out.size()[0]):
        f_out.write(f"{x_list[i]} ")
        for j in range(emb_dim):
            f_out.write(f"{cur_out[i][j]} ")
        f_out.write("\n")
    f_out.close()

    print('Val Loss: {:.4f}, Duration: {:.3f}'.
          format(loss.mean().item(), duration.mean().item()))

    start_time = time.time()
    t_out = model(test_dataset.to(device))
    cur_t_out, _ = torch.split(t_out, int(t_out.size()[0]/2))
    cur_t_out = cur_t_out.reshape((cur_t_out.size()[1], cur_t_out.size()[2]))
    cur_t_out = F.normalize(cur_t_out, dim=0)
    delta_t = time.time() - start_time
    print(f"Delta t: {delta_t}, size: {cur_t_out.size()[0]}")
    print(f"Time for each node: {delta_t / cur_t_out.size()[0]}")

    f_out = open(out_file2, "w")
    f_out.write("{} {}\n".format(cur_t_out.size()[0], emb_dim))
    for i in range(cur_t_out.size()[0]):
        f_out.write(f"{test_x_list[i]} ")
        for j in range(emb_dim):
            f_out.write(f"{cur_t_out[i][j]} ")
        f_out.write("\n")
    f_out.close()

def train(model, optimizer, data):
    model.train()
    optimizer.zero_grad()
    out = model(data)

    (h, l) = torch.split(out, int(out.size()[0]/2))
    h = h.reshape((h.size()[1], h.size()[2]))
    l = l.reshape((l.size()[1], l.size()[2]))

    L = LossFunc()
    loss = L(h, l)
    loss.backward()
    optimizer.step()

def evaluate(model, data):
    model.eval()

    with torch.no_grad():
        logits = model(data)

    outs = {}
    (h, l) = torch.split(logits, int(logits.size()[0]/2))
    h = h.reshape((h.size()[1], h.size()[2]))
    l = l.reshape((l.size()[1], l.size()[2]))
    L = LossFunc()
    loss = L(h, l)
    outs["loss"] = loss

    return outs

if __name__ == "__main__":
    building_id = emb_config["id"].split("_")[0]
    cur_set = "train"
    cur_id = building_id+"_"+cur_set
    edgelist_file = osp.abspath(osp.join(root_base, f"{dir_name}/{cur_id}/raw_data/prune_{cur_id}.edgelist"))
    out_file = osp.abspath(osp.join(root_base, f"{dir_name}/{cur_id}/embedding/dim_{emb_dim}.txt"))
    x_list, data = get_data(edgelist_file)
    cur_set = "test"
    cur_id = building_id+"_"+cur_set
    edgelist_file = osp.abspath(osp.join(root_base, f"{dir_name}/{cur_id}/raw_data/prune_{cur_id}.edgelist"))
    test_out_file = osp.abspath(osp.join(root_base, f"{dir_name}/{cur_id}/embedding/dim_{emb_dim}.txt"))
    test_x_list, test_data = get_data(edgelist_file)
    run(x_list, data, BiSAGE(data), runs, epochs, lr, weight_decay, early_stopping, test_x_list, test_data, out_file, test_out_file)  
