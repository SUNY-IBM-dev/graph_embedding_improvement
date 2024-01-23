from modulefinder import Module
import torch
import torch.nn as nn
import torch_geometric.nn as gnn

from torch_geometric.data.batch import Batch

from torch.nn import init   # For Kaiming He weight.init
from torch.nn import Parameter, Linear
from torch_geometric.nn.dense.linear import Linear as geometric_Linear
from torch_geometric.nn.inits import kaiming_uniform, zeros, uniform
import math
from typing import Optional, Tuple, Union
from torch import Tensor

import pandas as pd

import numpy as np

import copy

# ----------------------------
# The following are Pooling layers for the readout functions
# Currently I used two methods
# 1. MeanPool: Takes the mean for node embeddings per node
# 2. Identity: Does not change the node embedings
# ----------------------------


class GNNPool(nn.Module):
    def __init__(self):
        super().__init__()


class GlobalMeanPool(GNNPool):

    """
    Mean pooling
    """

    def __init__(self):
        super().__init__()

    def forward(self, x, batch):
        return gnn.global_mean_pool(x, batch)




class LocalMeanPool(GNNPool):

    """
    JY @ 2023-06-17
    "Local" Mean pooling
     Perform MeanPool based on a subset of node-feature-vectors (e.g. only 'thread' node-feature-vectors).
     Impl. done by masking
    """

    def __init__(self):
        super().__init__()

    # def forward(self, x, mask, batch):
    #     return gnn.global_mean_pool(x, batch)

    def forward(self, x, batch, node_mask_tensor):
        # JY @ 2023-06-17

        masked_x = x[node_mask_tensor]
        masked_batch = batch[node_mask_tensor]

        return gnn.global_mean_pool(masked_x, masked_batch)  # perform gnn.global-mean-pool on masked xs



class IdenticalPool(GNNPool):

    """
    Identity pooling (no change to embeddings)
    """

    def __init__(self):
        super().__init__()

    def forward(self, x, batch):
        return x

# add by mwang150 6/3/2022
class SumPool(GNNPool):

    """
    Sum pooling
    """

    def __init__(self):
        super().__init__()

    def forward(self, x, batch):
        return gnn.global_add_pool(x, batch)


class GlobalMaxPool(GNNPool):

    """
    Max pooling
    """
    def __init__(self):
        super().__init__()
    def forward(self, x, batch):
        return gnn.global_max_pool(x, batch)


# -------------------------------------------------------
# The following are GNN Implementations (GCN, GIN, GAT)
# -------------------------------------------------------


class GNNBasic(torch.nn.Module):

    """
    Parent class for GNNs
    """
    
    def __init__(self):
        super().__init__()

    def arguments_read(self, *args, **kwargs):
        """
        loads the graph samples for forward pass in models
        """
        data: Batch = kwargs.get('data') or None
        if not data:
            # this condition is for testing phase,
            # if a single sample is given for the forward pass
            x, edge_index, edge_attr, batch, ptr, y, name = args[0].x, args[0].edge_index, args[0].edge_attr, args[0].batch, args[0].ptr, args[0].y, args[0].name 
        else:
            # this is when a mini-batch of data is loaded for forward pass
            x, edge_index, edge_attr, batch, ptr, y, name = data.x, data.edge_index, data.edge_attr, data.batch, data.ptr, data.y, data.name
        # note: edge_index == edge_list, the pytorch-geometric library calls it the edge_index
        # note: this function will need to change if edge attributes are available
        return x, edge_index, edge_attr, batch, ptr, y, name 


class GCN(GNNBasic):

    """
    A GCN based GNN model with dropout and readout functions
    The model is for graph classification tasks
    """

    def __init__(self, dim_node : int, dim_hidden : list, num_classes : int, dropout_level : float,
                       # Below Added by JY @ 2022-06-29
                       activation_fn : nn.modules.activation = nn.ReLU,
                       activation_fn_kwargs : dict = {}, 
                       pool : GNNPool = GlobalMeanPool,
                ):
        """
        dim_node (int): the #features per node
        dim_hidden (list): the list of hidden sizes after each convolution step
        num_classes (int): the number of classes
        dropout_level (float): the dropout probability
        # Below Added by JY @ 2022-06-29
        pooling (nn.Module): pooling method
        activation_fn (nn.Module): activation function
        activation_fn_kwargs (dict): activation function keyword-arguments
        """
        super().__init__()
        num_layer = len(dim_hidden)

        self.conv1 = gnn.GCNConv(dim_node, dim_hidden[0])  # the first convolution layer

        # append the rest of the conv. layers
        # if other types of GNN layers are needed this and the above self.conv1 is where to changes them
        # if edge_weights are availabe, must change it here (must refer pytorch-geometric documentation)
        layers = []
        for i in range(num_layer - 1):
            layers.append(gnn.GCNConv(dim_hidden[i], dim_hidden[i + 1]))
        self.convs = nn.ModuleList(layers)
        
        '''
        # can change the activation functions as required
        # in case the data/features contains negative numbers then Relu()
        # might not be ~good?
        Joonyoung @ 2022-06-26: 
        As in the paper, replaced ReLU to LeakyReLU with negative input slope = 0.2 (GAT paper p.3)
        self.relu1 = nn.LeakyReLU(negative_slope = 0.2)       
        self.relus = nn.ModuleList(
            [ nn.LeakyReLU(negative_slope = 0.2) for _ in range(num_layer - 1) ]
        )
        self.readout = GlobalMeanPool()
        '''

        self.act_fn1 = activation_fn( **activation_fn_kwargs )
        self.act_fns = nn.ModuleList(
            [ activation_fn( **activation_fn_kwargs ) for _ in range(num_layer - 1) ]
        ) 

        self.readout = pool()

        self.ffn = nn.Sequential(*(
            [nn.Dropout(p=dropout_level), nn.Linear(dim_hidden[-1], num_classes), nn.Softmax(dim=1)]
        ))

        # can also use an additional Linear layer before predictions as follows:
        # self.ffn = nn.Sequential(*(
        #         [nn.Linear(dim_hidden[-1], dim_hidden[-1])] +
        #         [nn.ReLU(), nn.Dropout(p=dropout_level), nn.Linear(dim_hidden[-1], num_classes)]
        # ))
        return

    def forward(self, *args, **kwargs) -> torch.Tensor:
        """
        Does a single forward pass for the complete model
        """
        x, edge_index, edge_attr, batch, ptr, y, name = self.arguments_read(*args, **kwargs)

        # 1. first conv. pass -> uses the original data sample
        post_conv = self.act_fn1(self.conv1(x, edge_index))
        for conv, act_fn in zip(self.convs, self.act_fns):
            # 2. iteratively do GCN convolution
            post_conv = act_fn(conv(post_conv, edge_index))

        # 3. use the readout (i.e., pooling)
        out_readout = self.readout(post_conv, batch)

        # 4. the class probabilities
        out = self.ffn(out_readout)
        return out

    def get_emb(self, *args, **kwargs) -> torch.Tensor:
        """
        Auxilary function if node embeddings are required seperately
        works similar to the forward pass above, but without readout
        """
        x, edge_index, edge_attr, batch, ptr, y, name = self.arguments_read(*args, **kwargs)
        post_conv = self.act_fn1(self.conv1(x, edge_index))
        for conv, act_fn in zip(self.convs, self.act_fns):
            post_conv = act_fn(conv(post_conv, edge_index))
        return post_conv


class GIN(GNNBasic):
    """
    GINConv (Graph Isomorphic Convolution). model, can handle multiple edge attributes
    paper: https://arxiv.org/pdf/1905.12265.pdf (ICLR, 2020) for model with edge handling capability
    """

    def __init__(self, dim_node : int, dim_hidden : list, num_classes : int, dropout_level : float, edge_dim : int,
                       # Below Added by JY @ 2022-06-29
                       activation_fn : nn.modules.activation = nn.ReLU,
                       activation_fn_kwargs : dict = {}, 
                       pool : GNNPool = GlobalMeanPool, 
                ):
        """
        dim_node (int): the #features per node
        dim_hidden (list): the list of hidden sizes after each convolution step
        num_classes (int): the number of classes
        dropout_level (float): the dropout probability
        edge_dim (int): number of edge attributes
        # Below Added by JY @ 2022-06-29
        pooling (nn.Module): pooling method
        activation_fn (nn.Module): activation function
        activation_fn_kwargs (dict): activation function keyword-arguments
        """
        super().__init__()
        num_layer = len(dim_hidden)


        # Joonyoung: GINEConv == Graph Isomorphism Network with Edge Features, introduced by Strategies for Pre-training Graph Neural Networks
        #             https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.GINEConv
        
        self.conv1 = gnn.GINEConv(
            nn=nn.Sequential(*([
                nn.Linear(dim_node, (dim_node + dim_hidden[0]) // 2),            # Linear은 Linear-Transformation 즉 Weights 들임 layer 사이
                nn.Linear((dim_node + dim_hidden[0]) // 2, dim_hidden[0])
            ])),
            edge_dim=edge_dim)  # the first layer

        # append the rest of GIN layers
        layers = []
        for i in range(num_layer - 1):
            layers.append(
                gnn.GINEConv(
                    nn=nn.Sequential(*([
                        nn.Linear(dim_hidden[i], (dim_hidden[i] + dim_hidden[i + 1]) // 2),
                        nn.Linear((dim_hidden[i] + dim_hidden[i + 1]) // 2, dim_hidden[i + 1])
                    ])),
                    edge_dim=edge_dim))
        self.convs = nn.ModuleList(layers)
        
        # can change the activation functions as required
        # in case the data/features contains negative numbers then Relu()
        # might not be ~good?
        
        self.act_fn1 = activation_fn( **activation_fn_kwargs )
        self.act_fns = nn.ModuleList(
            [ activation_fn( **activation_fn_kwargs ) for _ in range(num_layer - 1) ]
        ) 

        self.readout = pool()       

        self.ffn = nn.Sequential(*(
            [nn.Dropout(p=dropout_level), nn.Linear(dim_hidden[-1], num_classes), nn.Softmax(dim=1)]
        ))
        # can also use an additional Linear layer before predictions as follows:
        # self.ffn = nn.Sequential(*(
        #         [nn.Linear(dim_hidden[-1], dim_hidden[-1])] +
        #         [nn.ReLU(), nn.Dropout(p=dropout_level), nn.Linear(dim_hidden[-1], num_classes)]
        # ))
        return


    def forward(self, *args, **kwargs) -> torch.Tensor:
        """
        Does a single forward pass for the complete model
        """
        x, edge_index, edge_attr, batch, ptr, y, name = self.arguments_read(*args, **kwargs)

        # 1. first conv. pass -> uses the original data sample
        post_conv = self.act_fn1(self.conv1(x=x, edge_index=edge_index, edge_attr=edge_attr))
        # print(post_conv)
        for conv, act_fn in zip(self.convs, self.act_fns):
            # 2. iteratively do GCN convolution
            post_conv = act_fn(conv(x=post_conv, edge_index=edge_index, edge_attr=edge_attr))

        # 3. use the readout (i.e., pooling)
        # print(post_conv)
        out_readout = self.readout(post_conv, batch)
        
        # 4. the class probabilities
        out = self.ffn(out_readout)


        # JY @ 2023-06-05: To Debug -----------------------------------------------------

        # outreadout_out_y_df = pd.DataFrame({'out_readout':  [[round(num, 5) for num in sublist] for sublist in out_readout.tolist()], 
        #                                     'out':  [[round(num, 5) for num in sublist] for sublist in out.tolist()],
        #                                     # 'pred': [0 if sublist[0] > sublist[1] else 1 for sublist in out.tolist() if sublist[0] != sublist[1]], 
        #                                     'y': y.tolist()})

        # outreadout_out_y_df.sort_values('y')

        # print(outreadout_out_y_df.sort_values('y'), flush= True)


        #-------------------------------------------------------------------------------------


        return out

    def get_emb(self, *args, **kwargs) -> torch.Tensor:
        """
        Auxilary function if node embeddings are required seperately
        works similar to the forward pass above, but without readout
        """
        x, edge_index, edge_attr, batch, ptr, y, name = self.arguments_read(*args, **kwargs)
        post_conv = self.act_fn1(self.conv1(x=x, edge_index=edge_index, edge_attr=edge_attr))
        for conv, act_fn in zip(self.convs, self.act_fns):
            post_conv = act_fn(conv(x=post_conv, edge_index=edge_index, edge_attr=edge_attr))
        return post_conv


class GIN_no_edgefeat_simplified(GNNBasic):
    """
    GINConv (Graph Isomorphic Convolution). model, can handle multiple edge attributes
    paper: https://arxiv.org/pdf/1905.12265.pdf (ICLR, 2020) for model with edge handling capability
    """

    def __init__(self, dim_node : int, dim_hidden : list, num_classes : int, dropout_level : float, edge_dim : int,
                       # Below Added by JY @ 2022-06-29
                       activation_fn : nn.modules.activation = nn.ReLU,
                       activation_fn_kwargs : dict = {}, 
                       pool : GNNPool = GlobalMeanPool, 
                ):
        """
        dim_node (int): the #features per node
        dim_hidden (list): the list of hidden sizes after each convolution step
        num_classes (int): the number of classes
        dropout_level (float): the dropout probability
        edge_dim (int): number of edge attributes
        # Below Added by JY @ 2022-06-29
        pooling (nn.Module): pooling method
        activation_fn (nn.Module): activation function
        activation_fn_kwargs (dict): activation function keyword-arguments
        """
        super().__init__()
        num_layer = len(dim_hidden)


        # Joonyoung: GINEConv == Graph Isomorphism Network with Edge Features, introduced by Strategies for Pre-training Graph Neural Networks
        #             https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.GINEConv
        
        self.conv1 = gnn.GINConv(
            nn=nn.Sequential(*([
                nn.Linear(dim_node, dim_hidden[0]),            # Linear은 Linear-Transformation 즉 Weights 들임 layer 사이
            ])))  # the first layer

        # append the rest of GIN layers
        layers = []
        for i in range(num_layer - 1):
            layers.append(
                gnn.GINConv(
                    nn=nn.Sequential(*([
                        nn.Linear(dim_hidden[i], dim_hidden[i + 1]),
                    ]))))
        self.convs = nn.ModuleList(layers)
        
        # can change the activation functions as required
        # in case the data/features contains negative numbers then Relu()
        # might not be ~good?
        
        self.act_fn1 = activation_fn( **activation_fn_kwargs )
        self.act_fns = nn.ModuleList(
            [ activation_fn( **activation_fn_kwargs ) for _ in range(num_layer - 1) ]
        ) 

        self.readout = pool()       

        self.ffn = nn.Sequential(*(
            [nn.Dropout(p=dropout_level), nn.Linear(dim_hidden[-1], num_classes), nn.Softmax(dim=1)]
        ))
        # can also use an additional Linear layer before predictions as follows:
        # self.ffn = nn.Sequential(*(
        #         [nn.Linear(dim_hidden[-1], dim_hidden[-1])] +
        #         [nn.ReLU(), nn.Dropout(p=dropout_level), nn.Linear(dim_hidden[-1], num_classes)]
        # ))
        return


    def forward(self, *args, **kwargs) -> torch.Tensor:
        """
        Does a single forward pass for the complete model
        """
        x, edge_index, edge_attr, batch, ptr, y, name = self.arguments_read(*args, **kwargs)

        # 1. first conv. pass -> uses the original data sample
        post_conv = self.act_fn1(self.conv1(x=x, edge_index=edge_index))
        # print(post_conv)
        for conv, act_fn in zip(self.convs, self.act_fns):
            # 2. iteratively do GCN convolution
            post_conv = act_fn(conv(x=post_conv, edge_index=edge_index))

        # 3. use the readout (i.e., pooling)
        # print(post_conv)
        out_readout = self.readout(post_conv, batch)
        
        # 4. the class probabilities
        out = self.ffn(out_readout)


        # JY @ 2023-06-05: To Debug -----------------------------------------------------

        # outreadout_out_y_df = pd.DataFrame({'out_readout':  [[round(num, 5) for num in sublist] for sublist in out_readout.tolist()], 
        #                                     'out':  [[round(num, 5) for num in sublist] for sublist in out.tolist()],
        #                                     # 'pred': [0 if sublist[0] > sublist[1] else 1 for sublist in out.tolist() if sublist[0] != sublist[1]], 
        #                                     'y': y.tolist()})

        # outreadout_out_y_df.sort_values('y')

        # print(outreadout_out_y_df.sort_values('y'), flush= True)


        #-------------------------------------------------------------------------------------


        return out

    def get_emb(self, *args, **kwargs) -> torch.Tensor:
        """
        Auxilary function if node embeddings are required seperately
        works similar to the forward pass above, but without readout
        """
        x, edge_index, edge_attr, batch, ptr, y, name = self.arguments_read(*args, **kwargs)
        post_conv = self.act_fn1(self.conv1(x=x, edge_index=edge_index))
        for conv, act_fn in zip(self.convs, self.act_fns):
            post_conv = act_fn(conv(x=post_conv, edge_index=edge_index))
        return post_conv


class GIN_no_edgefeat(GNNBasic):
    """
    GINConv (Graph Isomorphic Convolution). model, can handle multiple edge attributes
    paper: https://arxiv.org/pdf/1905.12265.pdf (ICLR, 2020) for model with edge handling capability
    """

    def __init__(self, dim_node : int, dim_hidden : list, num_classes : int, dropout_level : float,
                       # Below Added by JY @ 2022-06-29
                       activation_fn : nn.modules.activation = nn.ReLU,
                       activation_fn_kwargs : dict = {}, 
                       pool : GNNPool = GlobalMeanPool, 
                ):
        """
        dim_node (int): the #features per node
        dim_hidden (list): the list of hidden sizes after each convolution step
        num_classes (int): the number of classes
        dropout_level (float): the dropout probability
        edge_dim (int): number of edge attributes
        # Below Added by JY @ 2022-06-29
        pooling (nn.Module): pooling method
        activation_fn (nn.Module): activation function
        activation_fn_kwargs (dict): activation function keyword-arguments
        """
        super().__init__()
        num_layer = len(dim_hidden)


        # Joonyoung: GINEConv == Graph Isomorphism Network with Edge Features, introduced by Strategies for Pre-training Graph Neural Networks
        #             https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.GINEConv
        
        self.conv1 = gnn.GINConv(
            nn=nn.Sequential(*([
                nn.Linear(dim_node, (dim_node + dim_hidden[0]) // 2),
                nn.Linear((dim_node + dim_hidden[0]) // 2, dim_hidden[0])
            ])))  # the first layer

        # append the rest of GIN layers
        layers = []
        for i in range(num_layer - 1):
            layers.append(
                gnn.GINConv(
                    nn=nn.Sequential(*([
                        nn.Linear(dim_hidden[i], (dim_hidden[i] + dim_hidden[i + 1]) // 2),
                        nn.Linear((dim_hidden[i] + dim_hidden[i + 1]) // 2, dim_hidden[i + 1])
                    ]))))
        self.convs = nn.ModuleList(layers)
        
        # can change the activation functions as required
        # in case the data/features contains negative numbers then Relu()
        # might not be ~good?
        
        self.act_fn1 = activation_fn( **activation_fn_kwargs )
        self.act_fns = nn.ModuleList(
            [ activation_fn( **activation_fn_kwargs ) for _ in range(num_layer - 1) ]
        ) 

        self.readout = pool()       

        self.ffn = nn.Sequential(*(
            [nn.Dropout(p=dropout_level), nn.Linear(dim_hidden[-1], num_classes), nn.Softmax(dim=1)]
        ))
        # can also use an additional Linear layer before predictions as follows:
        # self.ffn = nn.Sequential(*(
        #         [nn.Linear(dim_hidden[-1], dim_hidden[-1])] +
        #         [nn.ReLU(), nn.Dropout(p=dropout_level), nn.Linear(dim_hidden[-1], num_classes)]
        # ))
        return


    def forward(self, *args, **kwargs) -> torch.Tensor:
        """
        Does a single forward pass for the complete model
        """
        x, edge_index, edge_attr, batch = self.arguments_read(*args, **kwargs)

        # 1. first conv. pass -> uses the original data sample
        post_conv = self.act_fn1(self.conv1(x=x, edge_index=edge_index))
        # print(post_conv)
        for conv, act_fn in zip(self.convs, self.act_fns):
            # 2. iteratively do GCN convolution
            post_conv = act_fn(conv(x=post_conv, edge_index=edge_index))

        # 3. use the readout (i.e., pooling)
        # print(post_conv)
        out_readout = self.readout(post_conv, batch)

        # 4. the class probabilities
        out = self.ffn(out_readout)
        return out

    def get_emb(self, *args, **kwargs) -> torch.Tensor:
        """
        Auxilary function if node embeddings are required seperately
        works similar to the forward pass above, but without readout
        """
        x, edge_index, edge_attr, batch = self.arguments_read(*args, **kwargs)
        post_conv = self.act_fn1(self.conv1(x=x, edge_index=edge_index))
        for conv, act_fn in zip(self.convs, self.act_fns):
            post_conv = act_fn(conv(x=post_conv, edge_index=edge_index))
        return post_conv



#################################################################################################################################
class GAT(GNNBasic):
    """
    GATConv (Graph Attention Network Convolution). model, can handle multiple edge attributes
    paper: https://arxiv.org/pdf/1710.10903.pdf
    """

    def __init__(self, dim_node : int, dim_hidden : list, num_classes : int, dropout_level : float, 
                       edge_dim : int , 
                       num_heads : int = 1,
                       # Below Added by JY @ 2022-06-29
                       activation_fn : nn.modules.activation = nn.ReLU,
                       activation_fn_kwargs : dict = {}, 
                       pool : GNNPool = GlobalMeanPool, 
                ):
        """
        dim_node (int): the #features per node
        dim_hidden (list): the list of hidden sizes after each convolution step
        num_classes (int): the number of classes
        dropout_level (float): the dropout probability
        edge_dim (int): number of edge attributes
        num_heads (int): number of multi-headed attentions
        # Below Added by JY @ 2022-06-29
        pooling (nn.Module): pooling method
        activation_fn (nn.Module): activation function
        activation_fn_kwargs (dict): activation function keyword-arguments
        """
        super().__init__()
        num_layer = len(dim_hidden)

        self.conv1 = gnn.GATConv(
            in_channels=dim_node,
            out_channels=dim_hidden[0],
            heads=num_heads,
            edge_dim=edge_dim
        )  # the first layer

        # append the rest of GAT layers
        layers = []
        for i in range(num_layer - 2):
            layers.append(
                gnn.GATConv(
                    in_channels=dim_hidden[i] * num_heads,
                    out_channels=dim_hidden[i + 1],
                    heads=num_heads,
                    edge_dim=edge_dim
                )
            )

        
        # final GAT layer will have heads==1
        layers.append(
            gnn.GATConv(
                in_channels=dim_hidden[-2] * num_heads,
                out_channels=dim_hidden[-1],
                heads=1,
                edge_dim=edge_dim,
                concat=False
            )
        )
           
        # [Added by JY @ 2022-07-18 to incorporate Kaiming He initialization]
        # Refer to : https://towardsdatascience.com/understand-kaiming-initialization-and-implementation-detail-in-pytorch-f7aa967e9138

        # In "https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/conv/gat_conv.html#GATConv"
        # It appears GATConv is already set to be initalized by glorot (xavier?) instead.
        # Look into this.

        '''
        layer = gnn.GATConv(
                in_channels=dim_hidden[-2] * num_heads,
                out_channels=dim_hidden[-1],
                heads=1,
                edge_dim=edge_dim,
                concat=False,
                
        )
        init.kaiming_normal_(layer.weight, mode = 'fan_in')
        '''
            
        
        self.convs = nn.ModuleList(layers)
        
        self.act_fn1 = activation_fn( **activation_fn_kwargs )
        self.act_fns = nn.ModuleList(
            [ activation_fn( **activation_fn_kwargs ) for _ in range(num_layer - 1) ]
        ) 

        self.readout = pool()

        self.ffn = nn.Sequential(*(
            [nn.Dropout(p=dropout_level), nn.Linear(dim_hidden[-1], num_classes), nn.Softmax(dim=1)]
        ))

        # can also use an additional Linear layer before predictions as follows:
        # self.ffn = nn.Sequential(*(
        #         [nn.Linear(dim_hidden[-1], dim_hidden[-1])] +
        #         [nn.ReLU(), nn.Dropout(p=dropout_level), nn.Linear(dim_hidden[-1], num_classes)]
        # ))
        
        
        # self.percentage_of_zero_arr = np.array([])
        
        return

    
    def forward(self, *args, **kwargs) -> torch.Tensor:
        """
        Does a single forward pass for the complete model
        """
        x, edge_index, edge_attr, batch, ptr, y, name = self.arguments_read(*args, **kwargs)

        # JY @ 2023-06-04 : To get the subgraph-level 1-gram feat-vec (could be both node-feat-vec and edge-feat-vec) do it here 



        # JY @ 2023-06-16 Make Node-Type Mask based on first node-input-vector (as node-type)
        #                 based on "/data/d1/jgwak1/STREAMLINED_DATA_GENERATION_MultiGraph_JY/STEP_3_processdata_traintestsplit/data_preprocessing/dp_v2_ONLY_TASKNAME_EDGE_ATTR/data_processor_v2_MultiEdge_5BitNodeAttr.py"
                # if "file" in node_name.lower():
                #     node_attr = [1,0,0,0,0]
                # elif "reg" in node_name.lower():
                #     node_attr = [0,1,0,0,0]
                # elif "net" in node_name.lower():
                #     node_attr = [0,0,1,0,0]
                # elif "proc" in node_name.lower():
                #     node_attr = [0,0,0,1,0]
                # elif "thread" in node_name.lower():
                #     node_attr = [0,0,0,0,1]

        # Thread_Node_Index_list = [4] # last index 
        # threadnode_mask_tensor_2 = x[:, Thread_Node_Index] == 1   

        # Thread_Node_Indices = [4] # last index 
        # import datetime
        # torch_any_check_start = datetime.datetime.now()
        # threadnode_mask_tensor = torch.any(x[:, Thread_Node_Indices] == 1, dim=1)
        # torch_any_check_done = datetime.datetime.now()
        # print(str(torch_any_check_done - torch_any_check_start),flush=True)

        FILE_NODE_INDEX = 0
        REG_NODE_INDEX = 1
        NET_NODE_INDEX = 2
        PROC_NODE_INDEX = 3
        THREAD_NODE_INDEX = 4

        # import datetime
        Node_Indices = [ FILE_NODE_INDEX, REG_NODE_INDEX ]
        # torch_any_check_start = datetime.datetime.now()
        node_mask_tensor = torch.any(x[:, Node_Indices] == 1, dim=1)
        # torch_any_check_done = datetime.datetime.now()
        # print(str(torch_any_check_done - torch_any_check_start),flush=True)



        # direct_check_start = datetime.datetime.now()
        # node_mask_tensor = x[:, FILE_NODE_INDEX] == 1

        # direct_check_done = datetime.datetime.now()
        # print(str(direct_check_done - direct_check_start),flush=True)

        # equality_check = torch.eq(threadnode_mask_tensor, threadnode_mask_tensor_check)

        # print(f"equality_check shape: {equality_check.shape}\n equality_check sum: {sum(equality_check)}\n", flush= True) 
        # print(f"threadnode_mask_tensor shape: {threadnode_mask_tensor.shape}\n threadnode_mask_tensor sum: {sum(threadnode_mask_tensor)}\n", flush= True) 
        # print(f"threadnode_mask_tensor_check shape: {threadnode_mask_tensor_check.shape}\n threadnode_mask_tensor_check sum: {sum(threadnode_mask_tensor_check)}\n", flush= True) 


        # 1. first conv. pass -> uses the original data sample   
         # [ "return_attention_weight = True" added by JY @ 2022-08-22 ] 
         # Refer to: https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.GATConv
        out = self.conv1(x=x, edge_index=edge_index, edge_attr=edge_attr) # return_attention_weights = True
        #print("out (weight-sum of input nuerons): {}\n".format(out))
        post_conv = self.act_fn1( out )
        #print("activation-function applied to out: {}\n".format(post_conv))
        for conv, act_fn in zip( self.convs, self.act_fns ):
            # 2. iteratively do GCN convolution
            
            # [ "return_attention_weight = True" added by JY @ 2022-08-22 ] 
            # Refer to: https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.GATConv
            out = conv(x=post_conv, edge_index=edge_index, edge_attr=edge_attr) #  return_attention_weights = True
            #print("out (weight-sum of input nuerons): {}\n".format(out))
            post_conv = act_fn( out )
            #print("activation-function applied to out: {}\n".format(post_conv))


            # get percentage of ReLU activation-function output's that are 0. (dying relu)
                    #     relu ( layer ( x ))


        # 3. use the readout (i.e., pooling)
        # print(post_conv)


        if str(self.readout) == "LocalMeanPool()":
            out_readout = self.readout(post_conv, batch, node_mask_tensor)
        else:
            out_readout = self.readout(post_conv, batch)

        # 4. the class probabilities

        # JY @ 2023-06-04 : To also give the subgraph-level 1-gram feat-vec to ffn do it here

        out = self.ffn(out_readout)
        return out

    def get_emb(self, *args, **kwargs) -> torch.Tensor:
        """
        Auxilary function if node embeddings are required seperately
        works similar to the forward pass above, but without readout
        """
        x, edge_index, edge_attr, batch, ptr, y, name = self.arguments_read(*args, **kwargs)
        post_conv = self.act_fn1(self.conv1(x=x, edge_index=edge_index, edge_attr=edge_attr))
        for conv, act_fn in zip(self.convs, self.act_fns):
            post_conv = act_fn(conv(x=post_conv, edge_index=edge_index, edge_attr=edge_attr))
        return post_conv




#################################################################################################################################

class GAT_mlp_fed_1gram(GNNBasic):
    """
    GATConv (Graph Attention Network Convolution). model, can handle multiple edge attributes
    paper: https://arxiv.org/pdf/1710.10903.pdf
    """

    def __init__(self, dim_node : int, dim_hidden : list, num_classes : int, dropout_level : float, 
                       edge_dim : int , 
                       num_heads : int = 1,
                       # Below Added by JY @ 2022-06-29
                       activation_fn : nn.modules.activation = nn.ReLU,
                       activation_fn_kwargs : dict = {}, 
                       pool : GNNPool = GlobalMeanPool, 
                ):
        """
        dim_node (int): the #features per node
        dim_hidden (list): the list of hidden sizes after each convolution step
        num_classes (int): the number of classes
        dropout_level (float): the dropout probability
        edge_dim (int): number of edge attributes
        num_heads (int): number of multi-headed attentions
        # Below Added by JY @ 2022-06-29
        pooling (nn.Module): pooling method
        activation_fn (nn.Module): activation function
        activation_fn_kwargs (dict): activation function keyword-arguments
        """
        super().__init__()
        num_layer = len(dim_hidden)

        self.conv1 = gnn.GATConv(
            in_channels=dim_node,
            out_channels=dim_hidden[0],
            heads=num_heads,
            edge_dim=edge_dim
        )  # the first layer

        # append the rest of GAT layers
        layers = []
        for i in range(num_layer - 2):
            layers.append(
                gnn.GATConv(
                    in_channels=dim_hidden[i] * num_heads,
                    out_channels=dim_hidden[i + 1],
                    heads=num_heads,
                    edge_dim=edge_dim
                )
            )

        
        # final GAT layer will have heads==1
        layers.append(
            gnn.GATConv(
                in_channels=dim_hidden[-2] * num_heads,
                out_channels=dim_hidden[-1],
                heads=1,
                edge_dim=edge_dim,
                concat=False
            )
        )
           
        # [Added by JY @ 2022-07-18 to incorporate Kaiming He initialization]
        # Refer to : https://towardsdatascience.com/understand-kaiming-initialization-and-implementation-detail-in-pytorch-f7aa967e9138

        # In "https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/conv/gat_conv.html#GATConv"
        # It appears GATConv is already set to be initalized by glorot (xavier?) instead.
        # Look into this.

        '''
        layer = gnn.GATConv(
                in_channels=dim_hidden[-2] * num_heads,
                out_channels=dim_hidden[-1],
                heads=1,
                edge_dim=edge_dim,
                concat=False,
                
        )
        init.kaiming_normal_(layer.weight, mode = 'fan_in')
        '''
            
        
        self.convs = nn.ModuleList(layers)
        
        self.act_fn1 = activation_fn( **activation_fn_kwargs )
        self.act_fns = nn.ModuleList(
            [ activation_fn( **activation_fn_kwargs ) for _ in range(num_layer - 1) ]
        ) 

        self.readout = pool()

        # self.ffn = nn.Sequential(*(
        #     [nn.Dropout(p=dropout_level), nn.Linear(dim_hidden[-1], num_classes), nn.Softmax(dim=1)]
        # ))

        # can also use an additional Linear layer before predictions as follows:
        # self.ffn = nn.Sequential(*(
        #         [nn.Linear(dim_hidden[-1], dim_hidden[-1])] +
        #         [nn.ReLU(), nn.Dropout(p=dropout_level), nn.Linear(dim_hidden[-1], num_classes)]
        # ))
        


        # JY @ 2023-06-25: As now GAT's MLP layer (ffn; decision-making network)
        #                  is being fed with the concatenation of "graph-summary(1d vector we get after pooling) + normalized 1gram featvec",
        #                  we want to more complexify our "ffn" to increase it's model-capacity.
        #
        #                  The network-architecture of "ffn" will be:
        #                   
        #                  dim_hidden[-1] + 1gram-dim("edge-dim - 1"; subtraction of 1 is b/c we are dropping time-scalar)
        #                  ( (dim_hidden[-1] + 1gram-dim) // 2 )
        #                  Relu() Dropnout
        #                  ( (dim_hidden[-1] + 1gram-dim) // 2 )
        #                  num_classes


        if edge_dim is not None:    # if Non-Edge-Feat-Migrated ver Data            
            self.one_gram_featvec_dim = edge_dim - 1 # -1 because time-info scalar drop
        
        else:                       # if Edge-Feat-Migrated ver Data
            self.one_gram_featvec_dim = dim_node - 5 # -5 for 5bit


        self.ffn = nn.Sequential(*(
                [ nn.Linear(dim_hidden[-1]+self.one_gram_featvec_dim, (dim_hidden[-1]+self.one_gram_featvec_dim) // 2) ] +
                [ nn.ReLU(), 
                  nn.Dropout(p=dropout_level), 
                  nn.Linear( (dim_hidden[-1]+self.one_gram_featvec_dim) // 2, num_classes), 
                  nn.Softmax(dim=1)]
        ))
        

                # nn.Linear(dim_node, (dim_node + dim_hidden[0]) // 2),            # Linear은 Linear-Transformation 즉 Weights 들임 layer 사이
                # nn.Linear((dim_node + dim_hidden[0]) // 2, dim_hidden[0])        
        
        return

    
    def forward(self, *args, **kwargs) -> torch.Tensor:
        """
        Does a single forward pass for the complete model
        """
        x, edge_index, edge_attr, batch, ptr, y, name = self.arguments_read(*args, **kwargs)

        # JY @ 2023-06-24 : To get the subgraph-level 1-gram feat-vec (could be both node-feat-vec and edge-feat-vec) do it here 
        if edge_attr is not None:    # if Non-Edge-Feat-Migrated ver Data            
            
            device = edge_attr.device
            
            # Need to first figure out which graph each edge belongs to, since current 'edge_attr' holds edges of all graphs in batch
            # https://github.com/pyg-team/pytorch_geometric/issues/1827
            src, dst = edge_index
            edge_batch = batch[src]

            # Determine the number of unique groups
            num_groups = edge_batch.max() + 1

            # Create the output tensor
            output = torch.zeros(num_groups, edge_attr[:,:-1].size(1)).to(device)  # edge_attr[:,:-1] b/c we dont consider last index (time-info scalar)     

            # Perform the sum based on groups
            output = output.scatter_add_(0, edge_batch.unsqueeze(1).repeat(1, edge_attr[:,:-1].size(1)), edge_attr[:,:-1])

            # Normalize the tensor between 0 and 1 (p=2 refers to L2(Euclidean) norm)
            # >> ofcourse instead of normalizing could've done torch.mean() however, 
            #    this makes the numbers very small.
            #    decide which one is better, ( option1: vec-sum then L2norm / option2: vec-mean )
            #
            # * normalize as below appears to result in somewhat similar numerical-range as "out_readout", so seemes fine to stick this way

            normalized_one_gram_featvectors = torch.nn.functional.normalize(output, p=2, dim = 1) # "dim = 1" for normalizing for each group
                                                                                                  # "p=2" for L2(Euclidean) norm
            del src, dst, edge_batch, num_groups, output # free gpu-memory, as this var no longer needed, in case device is cuda


        
        else:                       # if Edge-Feat-Migrated ver Data

            device = x.device

            # first 5 elements of "x" correspond to "5-bit node-type onehot vector", 
            # and the rest of "x" correspond to "one_gram_featvec_dim"

            # Determine the number of unique groups
            if batch is not None:
                num_groups = batch.max() + 1

                # Create the output tensor
                output = torch.zeros(num_groups, x[:, 5:].size(1)).to(device)            

                # Perform the sum based on groups
                output = output.scatter_add_(0, batch.unsqueeze(1).repeat(1, x[:, 5:].size(1)), x[:, 5:])

                # Normalize the tensor between 0 and 1 (p=2 refers to L2(Euclidean) norm)
                # >> ofcourse instead of normalizing could've done torch.mean() however, 
                #    this makes the numbers very small.
                #    decide which one is better, ( option1: vec-sum then L2norm / option2: vec-mean )
                #
                # * normalize as below appears to result in somewhat similar numerical-range as "out_readout", so seemes fine to stick this way
                normalized_one_gram_featvectors = torch.nn.functional.normalize(output, p=2, dim = 1) # "dim = 1" for normalizing for each group
                                                                                                    # "p=2" for L2(Euclidean) norm
                del output, num_groups # free gpu-memory, as this var no longer needed, in case device is cuda

            else:
                output = torch.sum(x[:, 5:], dim=0)
                normalized_one_gram_featvector = torch.nn.functional.normalize(output, p=2, dim = 0)


        # JY @ 2023-06-16 Make Node-Type Mask based on first node-input-vector (as node-type)
        #                 based on "/data/d1/jgwak1/STREAMLINED_DATA_GENERATION_MultiGraph_JY/STEP_3_processdata_traintestsplit/data_preprocessing/dp_v2_ONLY_TASKNAME_EDGE_ATTR/data_processor_v2_MultiEdge_5BitNodeAttr.py"
                # if "file" in node_name.lower():
                #     node_attr = [1,0,0,0,0]
                # elif "reg" in node_name.lower():
                #     node_attr = [0,1,0,0,0]
                # elif "net" in node_name.lower():
                #     node_attr = [0,0,1,0,0]
                # elif "proc" in node_name.lower():
                #     node_attr = [0,0,0,1,0]
                # elif "thread" in node_name.lower():
                #     node_attr = [0,0,0,0,1]

        # Thread_Node_Index_list = [4] # last index 
        # threadnode_mask_tensor_2 = x[:, Thread_Node_Index] == 1   

        # Thread_Node_Indices = [4] # last index 
        # import datetime
        # torch_any_check_start = datetime.datetime.now()
        # threadnode_mask_tensor = torch.any(x[:, Thread_Node_Indices] == 1, dim=1)
        # torch_any_check_done = datetime.datetime.now()
        # print(str(torch_any_check_done - torch_any_check_start),flush=True)


        if str(self.readout) == "LocalMeanPool()":

            FILE_NODE_INDEX = 0
            REG_NODE_INDEX = 1
            NET_NODE_INDEX = 2
            PROC_NODE_INDEX = 3
            THREAD_NODE_INDEX = 4

            # import datetime
            Node_Indices = [ FILE_NODE_INDEX, REG_NODE_INDEX ]
            # torch_any_check_start = datetime.datetime.now()
            node_mask_tensor = torch.any(x[:, Node_Indices] == 1, dim=1)
            # torch_any_check_done = datetime.datetime.now()
            # print(str(torch_any_check_done - torch_any_check_start),flush=True)



        # direct_check_start = datetime.datetime.now()
        # node_mask_tensor = x[:, FILE_NODE_INDEX] == 1

        # direct_check_done = datetime.datetime.now()
        # print(str(direct_check_done - direct_check_start),flush=True)

        # equality_check = torch.eq(threadnode_mask_tensor, threadnode_mask_tensor_check)

        # print(f"equality_check shape: {equality_check.shape}\n equality_check sum: {sum(equality_check)}\n", flush= True) 
        # print(f"threadnode_mask_tensor shape: {threadnode_mask_tensor.shape}\n threadnode_mask_tensor sum: {sum(threadnode_mask_tensor)}\n", flush= True) 
        # print(f"threadnode_mask_tensor_check shape: {threadnode_mask_tensor_check.shape}\n threadnode_mask_tensor_check sum: {sum(threadnode_mask_tensor_check)}\n", flush= True) 


        # 1. first conv. pass -> uses the original data sample   
         # [ "return_attention_weight = True" added by JY @ 2022-08-22 ] 
         # Refer to: https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.GATConv
        out = self.conv1(x=x, edge_index=edge_index, edge_attr=edge_attr) # return_attention_weights = True
        #print("out (weight-sum of input nuerons): {}\n".format(out))
        post_conv = self.act_fn1( out )
        #print("activation-function applied to out: {}\n".format(post_conv))
        for conv, act_fn in zip( self.convs, self.act_fns ):
            # 2. iteratively do GCN convolution
            
            # [ "return_attention_weight = True" added by JY @ 2022-08-22 ] 
            # Refer to: https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.GATConv
            out = conv(x=post_conv, edge_index=edge_index, edge_attr=edge_attr) #  return_attention_weights = True
            #print("out (weight-sum of input nuerons): {}\n".format(out))
            post_conv = act_fn( out )
            #print("activation-function applied to out: {}\n".format(post_conv))


            # get percentage of ReLU activation-function output's that are 0. (dying relu)
                    #     relu ( layer ( x ))


        # 3. use the readout (i.e., pooling)
        # print(post_conv)


        if str(self.readout) == "LocalMeanPool()":
            out_readout = self.readout(post_conv, batch, node_mask_tensor)
        else:
            out_readout = self.readout(post_conv, batch)

        # 4. the class probabilities

        # JY @ 2023-06-24 : To also give the subgraph-level 1-gram feat-vec to ffn do it here
        if batch is not None:
            readout_AND_normalized_one_gram_feat_CONCAT_tensor = torch.cat((out_readout, normalized_one_gram_featvectors), dim = 1)

        else:
            readout_AND_normalized_one_gram_feat_CONCAT_tensor = torch.cat((out_readout, normalized_one_gram_featvector.unsqueeze(0)), dim = 1)


        out = self.ffn( readout_AND_normalized_one_gram_feat_CONCAT_tensor )
        return out

    def get_emb(self, *args, **kwargs) -> torch.Tensor:
        """
        Auxilary function if node embeddings are required seperately
        works similar to the forward pass above, but without readout
        """
        x, edge_index, edge_attr, batch, ptr, y, name = self.arguments_read(*args, **kwargs)
        post_conv = self.act_fn1(self.conv1(x=x, edge_index=edge_index, edge_attr=edge_attr))
        for conv, act_fn in zip(self.convs, self.act_fns):
            post_conv = act_fn(conv(x=post_conv, edge_index=edge_index, edge_attr=edge_attr))
        return post_conv 





# -------------------------------------------------------
# The following are GAT_He
# -------------------------------------------------------

class GATConv_He( gnn.GATConv ):

    ''' 
    Replace "Xavier (glorot)" weight-initialization which was originally devised for sigmoid/tanh activation-functions
    to "Kaiming He" weight-initialization devised for ReLU activation-function. 
    '''
    # Referring to : 
    # https://pytorch-geometric.readthedocs.io/en/1.5.0/_modules/torch_geometric/nn/conv/gat_conv.html#GATConv
    # https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.GATConv

    #from torch.nn import Parameter, Linear
    #from torch_geometric.nn.inits import kaiming_uniform

    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        heads: int = 1,
        concat: bool = True,
        negative_slope: float = 0.2,
        dropout: float = 0.0,
        add_self_loops: bool = True,
        edge_dim: Optional[int] = None,
        fill_value: Union[float, Tensor, str] = 'mean',
        bias: bool = True,
        **kwargs,
    ):
        kwargs.setdefault('aggr', 'add')
        super().__init__( 
                          in_channels = in_channels, 
                          out_channels = out_channels,


                          **kwargs
                        )

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops
        self.edge_dim = edge_dim
        self.fill_value = fill_value

        # Refer to: https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html

        # In case we are operating in bipartite graphs, we apply separate
        # transformations 'lin_src' and 'lin_dst' to source and target nodes:
        if isinstance(in_channels, int):
            self.lin_src = geometric_Linear(in_channels, heads * out_channels,
                                            bias=False, weight_initializer='kaiming_uniform')
            self.lin_dst = self.lin_src
        else:
            self.lin_src = geometric_Linear(in_channels[0], heads * out_channels, False,
                                            weight_initializer='kaiming_uniform')
            self.lin_dst = geometric_Linear(in_channels[1], heads * out_channels, False,
                                            weight_initializer='kaiming_uniform')

        # The learnable parameters to compute attention coefficients:
        self.att_src = Parameter(torch.Tensor(1, heads, out_channels))
        self.att_dst = Parameter(torch.Tensor(1, heads, out_channels))

        if edge_dim is not None:
            self.lin_edge = geometric_Linear(edge_dim, heads * out_channels, bias=False,
                                             weight_initializer='kaiming_uniform')
            self.att_edge = Parameter(torch.Tensor(1, heads, out_channels))
        else:
            self.lin_edge = None
            self.register_parameter('att_edge', None)

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()       


    def reset_parameters(self):
        # Replaced 'glorot' to kaiming_uniform and zeros with uniform.
        self.lin_src.reset_parameters()
        self.lin_dst.reset_parameters()
        if self.lin_edge is not None:
            self.lin_edge.reset_parameters()

        # to check implementation of kaiming_uniform,
        # refer to: https://github.com/pyg-team/pytorch_geometric/blob/master/torch_geometric/nn/inits.py
        #           https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/conv/dna_conv.html?highlight=kaiming_uniform
        #           https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/conv/gat_conv.html
        #           https://paperswithcode.com/method/he-initialization#:~:text=Kaiming%20Initialization%2C%20or%20He%20Initialization,functions%2C%20such%20as%20ReLU%20activations.&text=2%20%2F%20n%20l%20)-,That%20is%2C%20a%20zero%2Dcentered%20Gaussian%20with%20standard%20deviation%20of,Biases%20are%20initialized%20at%20.
        #           https://pytorch.org/docs/stable/nn.init.html
        #           https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/dense/linear.html
        # Check what exactly fan is.
        
        # fan in: number of input neurons
        # fan out: number of output neurons
        # a – the negative slope of the rectifier used after this layer (only used with 'leaky_relu')

        # How exactly should I set fan = ? should it be fan in fan out?
        # >> check this: https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/dense/linear.html
        #    "fan" should be equivalent to in_channel


        # 'fan-in' for self.att_src.size(1) or self.in_channels?


        # self.att_src.size() 가 torch.size([1,1,1024]) 이면 중간 1 이 fan-in 해당 1024가 fan-out 해당 확인됨 (glorot impl에서).

        kaiming_uniform(self.att_src, fan = self.att_src.size(-2) , a = 0)    # 'fan-in'  # a = 0 for just relu
        kaiming_uniform(self.att_dst, fan = self.att_dst.size(-2) , a = 0)
        if self.att_edge is not None:
            kaiming_uniform(self.att_edge, fan = self.att_edge.size(-2)  , a = 0)
        zeros(self.bias) # For Kaiming Initalization, Biases are initialized at 0.



class GAT_He(GNNBasic):
    """
    GATConv (Graph Attention Network Convolution). model, can handle multiple edge attributes
    paper: https://arxiv.org/pdf/1710.10903.pdf
    """

    def __init__(self, dim_node : int, dim_hidden : list, num_classes : int, dropout_level : float, 
                       edge_dim : int , 
                       num_heads : int = 1,
                       # Below Added by JY @ 2022-06-29
                       activation_fn : nn.modules.activation = nn.ReLU,
                       activation_fn_kwargs : dict = {}, 
                       pool : GNNPool = GlobalMeanPool, 
                ):
        """
        dim_node (int): the #features per node
        dim_hidden (list): the list of hidden sizes after each convolution step
        num_classes (int): the number of classes
        dropout_level (float): the dropout probability
        edge_dim (int): number of edge attributes
        num_heads (int): number of multi-headed attentions
        # Below Added by JY @ 2022-06-29
        pooling (nn.Module): pooling method
        activation_fn (nn.Module): activation function
        activation_fn_kwargs (dict): activation function keyword-arguments
        """
        super().__init__()
        num_layer = len(dim_hidden)

        self.conv1 = GATConv_He(
            in_channels=dim_node,
            out_channels=dim_hidden[0],
            heads=num_heads,
            edge_dim=edge_dim
        )  # the first layer

        # append the rest of GAT layers
        layers = []
        for i in range(num_layer - 2):
            layers.append(
                GATConv_He(
                    in_channels=dim_hidden[i] * num_heads,
                    out_channels=dim_hidden[i + 1],
                    heads=num_heads,
                    edge_dim=edge_dim
                )
            )

        
        # final GAT layer will have heads==1
        layers.append(
            GATConv_He(
                in_channels=dim_hidden[-2] * num_heads,
                out_channels=dim_hidden[-1],
                heads=1,
                edge_dim=edge_dim,
                concat=False
            )
        )
           
        self.convs = nn.ModuleList(layers)
        
        self.act_fn1 = activation_fn( **activation_fn_kwargs )
        self.act_fns = nn.ModuleList(
            [ activation_fn( **activation_fn_kwargs ) for _ in range(num_layer - 1) ]
        ) 

        self.readout = pool()

        self.ffn = nn.Sequential(*(
            [nn.Dropout(p=dropout_level), nn.Linear(dim_hidden[-1], num_classes), nn.Softmax(dim=1)]
        ))

        # can also use an additional Linear layer before predictions as follows:
        # self.ffn = nn.Sequential(*(
        #         [nn.Linear(dim_hidden[-1], dim_hidden[-1])] +
        #         [nn.ReLU(), nn.Dropout(p=dropout_level), nn.Linear(dim_hidden[-1], num_classes)]
        # ))
        return

    
    def forward(self, *args, **kwargs) -> torch.Tensor:
        """
        Does a single forward pass for the complete model
        """
        x, edge_index, edge_attr, batch, ptr, y, name = self.arguments_read(*args, **kwargs)

        # 1. first conv. pass -> uses the original data sample
        post_conv = self.act_fn1(self.conv1(x=x, edge_index=edge_index, edge_attr=edge_attr))
        # print(post_conv)
        for conv, act_fn in zip(self.convs, self.act_fns):
            # 2. iteratively do GCN convolution

            ''' init.kaiming_normal_( conv.we) '''  # [ Added by JY @ 2022-07-18 ]

            post_conv = act_fn( conv(x=post_conv, edge_index=edge_index, edge_attr=edge_attr) )
                    #     relu ( layer ( x ))


        # 3. use the readout (i.e., pooling)
        # print(post_conv)


        out_readout = self.readout(post_conv, batch)  # JY  @ 2023-06-16 : Pass info of which node is which node-type

        # 4. the class probabilities
        out = self.ffn(out_readout)
        return out

    def get_emb(self, *args, **kwargs) -> torch.Tensor:
        """
        Auxilary function if node embeddings are required seperately
        works similar to the forward pass above, but without readout
        """
        x, edge_index, edge_attr, batch, ptr, y, name = self.arguments_read(*args, **kwargs)
        post_conv = self.act_fn1(self.conv1(x=x, edge_index=edge_index, edge_attr=edge_attr))
        for conv, act_fn in zip(self.convs, self.act_fns):
            post_conv = act_fn(conv(x=post_conv, edge_index=edge_index, edge_attr=edge_attr))
        return post_conv

