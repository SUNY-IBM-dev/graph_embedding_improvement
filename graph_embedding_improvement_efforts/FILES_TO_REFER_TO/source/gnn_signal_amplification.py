
from typing import Callable, Optional, Union

import torch
from torch import Tensor
from torch_geometric.nn.aggr import Aggregation, MultiAggregation
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import Adj, OptPairTensor, Size, SparseTensor
from torch_geometric.utils import spmm
from torch_geometric.nn.conv import MessagePassing

from typing import List, Optional, Tuple, Union

import torch.nn.functional as F
from torch import Tensor
from torch.nn import LSTM

# JY @ 2023-05-24: exactly same as above "from torch_geometric.nn.conv import MessagePassing"
# from source.my_message_passing_JY_for_GNN_Signal_Amplification import MessagePassing_JY_for_GNN_Signal_Amplification  
# from source.my_message_passing_JY_for_GNN_Signal_Amplification_Version2 import MessagePassing_JY_for_GNN_Signal_Amplification__Version2 
from source.my_message_passing_JY_for_GNN_Signal_Amplification import MessagePassing_JY_for_GNN_Signal_Amplification  
from source.my_message_passing_JY_for_GNN_Signal_Amplification_Version2 import MessagePassing_JY_for_GNN_Signal_Amplification__Version2 

from torch_geometric.nn.dense.linear import Linear   # https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.dense.Linear
from torch_geometric.nn.inits import reset
from torch_geometric.typing import (
    Adj,
    OptPairTensor,
    OptTensor,
    Size,
    SparseTensor,
)
from torch_geometric.utils import spmm    # https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.dense.Linear


from typing import Optional

import torch
from torch import Tensor
from torch.nn import Parameter

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import zeros
from torch_geometric.typing import (
    Adj,
    OptPairTensor,
    OptTensor,
    SparseTensor,
    torch_sparse,
)
from torch_geometric.utils import add_remaining_self_loops
from torch_geometric.utils import add_self_loops as add_self_loops_fn
from torch_geometric.utils import (
    is_torch_sparse_tensor,
    scatter,
    spmm,
    to_edge_index,
)
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_geometric.utils.sparse import set_sparse_value

# from model import GIN

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

import numpy as np

# Added by JY @ 2023-05-31
import torch.multiprocessing as mp
# mp.set_start_method("spawn")

from sklearn import metrics

from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.cluster import DBSCAN


from collections import Counter

import pandas as pd
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)

# https://stackoverflow.com/questions/54374935/how-to-fix-this-strange-error-runtimeerror-cuda-error-out-of-memory


# To understand types of nodes in above cycle and non-cycle pairs following can be used:
def FiveBitNodeAttr2NodeTypeStr(set_bit_idx : int):
    # /tabby/ibm/STREAMLINED_DATA_GENERATION_MultiGraph/STEP_3_processdata_traintestsplit/data_preprocessing/dp_v2_ONLY_TASKNAME_EDGE_ATTR/data_processor_v2_MultiEdge_5BitNodeAttr.py
    if set_bit_idx == 0: # [1,0,0,0,0]
        return "file"
    elif set_bit_idx == 1: # [0,1,0,0,0]
        return "reg"
    elif set_bit_idx == 2: # [0,0,1,0,0]
        return "net"
    elif set_bit_idx == 3: # [0,0,0,1,0]
        return "proc"
    elif set_bit_idx == 4: # [0,0,0,0,1]
        return "thread"
    else: # [0,0,0,0,0]
        return "weird-case"    


import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
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

        # JY @ 2023-06-05 -- Also Pass the "name"

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


class GlobalMaxPool(GNNPool):

    """
    Max pooling
    """
    def __init__(self):
        super().__init__()
    def forward(self, x, batch):
        return gnn.global_max_pool(x, batch)



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


########################################################################################################################################

# (Version-1) : Direct incoropration of "signal-amplification". Separate from Dr.Xiaokui's suggestion.
class GNN_Signal_Amplification_Conv__Version_1( MessagePassing_JY_for_GNN_Signal_Amplification ):
    r"""The modified :class:`GINConv` operator from the `"Strategies for
    Pre-training Graph Neural Networks" <https://arxiv.org/abs/1905.12265>`_
    paper
    .. math::
        \mathbf{x}^{\prime}_i = h_{\mathbf{\Theta}} \left( (1 + \epsilon) \cdot
        \mathbf{x}_i + \sum_{j \in \mathcal{N}(i)} \mathrm{ReLU}
        ( \mathbf{x}_j + \mathbf{e}_{j,i} ) \right)
    that is able to incorporate edge features :math:`\mathbf{e}_{j,i}` into
    the aggregation procedure.
    Args:
        nn (torch.nn.Module): A neural network :math:`h_{\mathbf{\Theta}}` that
            maps node features :obj:`x` of shape :obj:`[-1, in_channels]` to
            shape :obj:`[-1, out_channels]`, *e.g.*, defined by
            :class:`torch.nn.Sequential`.
        eps (float, optional): (Initial) :math:`\epsilon`-value.
            (default: :obj:`0.`)
        train_eps (bool, optional): If set to :obj:`True`, :math:`\epsilon`
            will be a trainable parameter. (default: :obj:`False`)
        edge_dim (int, optional): Edge feature dimensionality. If set to
            :obj:`None`, node and edge feature dimensionality is expected to
            match. Other-wise, edge features are linearly transformed to match
            node feature dimensionality. (default: :obj:`None`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    Shapes:
        - **input:**
          node features :math:`(|\mathcal{V}|, F_{in})` or
          :math:`((|\mathcal{V_s}|, F_{s}), (|\mathcal{V_t}|, F_{t}))`
          if bipartite,
          edge indices :math:`(2, |\mathcal{E}|)`,
          edge features :math:`(|\mathcal{E}|, D)` *(optional)*
        - **output:** node features :math:`(|\mathcal{V}|, F_{out})` or
          :math:`(|\mathcal{V}_t|, F_{out})` if bipartite
    """
    def __init__(self, 
                 nn: torch.nn.Module, 
                 signal_amplification_option : str,

                 edge_dim: Optional[int] = None,
                 # added by JY @ 2023-05-24
                 aggr: Optional[Union[str, List[str], Aggregation]] = "add",

                 **kwargs):
        

        #####################################################################################################################
        kwargs.setdefault('aggr', aggr )
        super().__init__(**kwargs)

        self.nn = nn
        self.signal_amplification_option = signal_amplification_option
        self.edge_dim = edge_dim

        self.reset_parameters()


    def reset_parameters(self):
        super().reset_parameters() # aggr-module 
        reset(self.nn)



    def forward(self, x: Union[Tensor, OptPairTensor], 
                edge_index: Adj,
                batch: Tensor,
                ptr,
                edge_attr: OptTensor = None,
                size: Size = None) -> Tensor:

        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        # propagate_type: (x: OptPairTensor)
        out, batch_for_threadnodes = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=size, 
                                                    ptr = ptr)

        # x_r = x[1]
        # if x_r is not None:
        #     # out = out + (1 + self.eps) * x_r

        # "make 'out' to have torch.float64" to avoid "mat1 and mat2 must have the same dtype"
        # out = out.to(torch.float64)
        return self.nn(out), batch_for_threadnodes


    # def message_and_aggregate(self, edge_index, x, edge_attr, adj_t) -> Tensor:

    #     # JY @ 2023-07-28: Here, need both edge_index (src/tar) info, and node-x info. need all info.

    #     # NOTE THAT NOW IT IS UNDIRECTED GRAPH (Might want to do some sanity-checking)
    #     # https://docs.google.com/presentation/d/1nf87G4Ux7U14e6k2w38oLDO_HcnAT_rOwV8qSSt6WqA/edit#slide=id.g25398450085_0_38 -- slide 337
    #     edge_src_indices = edge_index[0]
    #     edge_tar_indices = edge_index[1]

    #     # 1. first identify all thread-nodes -------------------------------------------------------------------------
    #     import datetime
    #     start = datetime.datetime.now()
    #     thread_node_indices = torch.nonzero(torch.all(torch.eq( x[:,:5], torch.tensor([0, 0, 0, 0, 1], device= x.device)), dim=1), as_tuple=False).flatten()
        
    #     thread_node_indices__expanded_for_broatcasting = thread_node_indices.unsqueeze(1)
    #     thread_node_is_src_of_edge__mask = (thread_node_indices__expanded_for_broatcasting == edge_src_indices) # # Create a mask where matching values are True
    #     thread_node_is_src_of_edge__edge_indices = [torch.nonzero(m).squeeze(1) for m in thread_node_is_src_of_edge__mask]
    #     thread_node_is_src_of_edge__to__edge_index__dict = {value.item(): indices for value, indices in zip(thread_node_indices, thread_node_is_src_of_edge__edge_indices)}

        


        '''
        JY @ 2023-07-28
        Doing everything in here (getting incoming/outgoing edge etc.) is not scalable.
        Need to make it into an undirected graph.
        # ***************************************************
        # JY @ 2023-07-28: Here, need both edge_index (src/tar) info, and node-x info. need all info.

        # edge_src_indices = edge_index[0]
        # edge_tar_indices = edge_index[1]

        # # 1. first identify all thread-nodes -------------------------------------------------------------------------
        # import datetime
        # start = datetime.datetime.now()
        # thread_node_indices = torch.nonzero(torch.all(torch.eq( x[:,:5], torch.tensor([0, 0, 0, 0, 1], device= x.device)), dim=1), as_tuple=False).flatten()
        
        # # 2. for each thread-node, get all incoming-and-outgoing-edges and 1gram-event-dist --------------------------
        # thread_node_indices__expanded_for_broatcasting = thread_node_indices.unsqueeze(1)
        # # thread-node is source
        # thread_node_is_src_of_edge__mask = (thread_node_indices__expanded_for_broatcasting == edge_src_indices) # # Create a mask where matching values are True
        # thread_node_is_src_of_edge__edge_indices = [torch.nonzero(m).squeeze(1) for m in thread_node_is_src_of_edge__mask]
        # thread_node_is_src_of_edge__to__edge_index__dict = {value.item(): indices for value, indices in zip(thread_node_indices, thread_node_is_src_of_edge__edge_indices)}
        # # thread-node is target
        # thread_node_is_tar_of_edge__mask = (thread_node_indices__expanded_for_broatcasting == edge_tar_indices) # # Create a mask where matching values are True
        # thread_node_is_tar_of_edge__edge_indices = [torch.nonzero(m).squeeze(1) for m in thread_node_is_tar_of_edge__mask]
        # thread_node_is_tar_of_edge__to__edge_index__dict = {value.item(): indices for value, indices in zip(thread_node_indices, thread_node_is_tar_of_edge__edge_indices)}
        # # thread-node both-directions 
        # thread_node_both_direction__edge_indices_set = [torch.cat((t1, t2), dim= 0) for t1, t2 in zip(thread_node_is_src_of_edge__edge_indices, thread_node_is_tar_of_edge__edge_indices)]
        # thread_node_both_direction__edge_index__dict = {value.item(): indices for value, indices in zip(thread_node_indices, thread_node_both_direction__edge_indices)}

        # # 3. for each thread-node, get all unique-adjacent-nodes (nodes that interacted with the thread; avoid duplicate-count), and get node-dict ---------------



        # # 4. now 
        # # torch.tensor
        # for thread_node_both_direction__edge_indices in thread_node_both_direction__edge_indices_set:
        #         torch.sum( torch.stack([torch.index_select(edge_attr[:,:-1], dim=0, index=tensor) for tensor in thread_node_both_direction__edge_indices]), dim = 0)
             
        #         #  [torch.sum(torch.index_select(edge_attr[:,:-1], dim=1, index=tensor)) for tensor in thread_node_both_direction__edge_indices]

        # # TODO: might need to get a mapping from thread-node to corresponding graph in graph-batch, -- 'batch' or 'ptr' could do
        # done = datetime.datetime.now()
        # print(done -start)

        # # 4. aggregate

        # # output = self.aggr_module( x = inputs,  index = index ) # batch?

        # # return output # output-dimension dim=0 should match the max of index - 1 (due to starting from indx 0)

        # # 4. Also, note that now as dealing with NN, might want to normlaized (but could do the two components separately.)


        '''


        # Refer to: https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#aggregation-operators




        # return super().message_and_aggregate(adj_t)



    def message(self, 
                x_j: Tensor, 
                edge_attr: Tensor, 
                x, 
                edge_index) -> Tensor:
        
        # This function is called in propagate()


        # Added by JY @ 2023-05-24:
        assert edge_attr.shape[0] == x_j.shape[0], "edge and source-node number mismatch"
        # What I will do is concatenate the source-node-feature and edge-feature.
        # (node)<-----(edge; edge-feat)-----(sourc-node; node-feat)

        # return (x_j + edge_attr).relu() 



        


        ###############################################################################################
        # JY @ 2023-07-28: Compromising for now.
        #                  Due to multi-graph and handling both incoming and outgoing-edge, adjacent nodes can be counted multiple times.
        #                  For straightforwardness, signal-amplification counts only once for an adjacent node (which the thread-node interacted).
        #                  Thus, here, keep "message" only as event.
        #                  In aggregate, function, can deal with incoming-and-outgoing-messages etc. 
        # # message = edge_attr[:,:-1]
        # from torch_geometric.utils import get_neighbors


        # # Calculate the number of unique neighbors for each node
        # from torch_scatter import scatter_add
        
        # # num_nodes = edge_index.max().item() + 1
        # unique_neighbors = scatter_add(torch.ones_like(edge_index[1]), edge_index[0], dim=0, dim_size=num_nodes)

        # # Sample graph representation (using edge_index for example)
        # # Get unique node indices using torch.unique
        # unique_node_indices, inverse_indices = torch.unique(edge_index, return_inverse=True)

        # # Use scatter to create a list of unique neighbors for each node
        # unique_neighbors = scatter(unique_node_indices[1], inverse_indices, dim=0, reduce='list')

        # # pair_to_check = torch.tensor([0,2771])
        # exists = torch.equal(edge_attr, pair_to_check.unsqueeze(0))

        # Compromise for now. -- for non-unique ajacent nodes *

        # zero_vectors = torch.zeros( edge_attr.shape[0], 
        #                             x_j.shape[1] - edge_attr[:,:-1].shape[1], # "edge_attr[:,:-1]" is to drop the time-information scalar 
        #                             device = edge_attr.device )               # "x_j.shape[1] - edge_attr[:,:-1].shape[1]" is to front-pad "edge_attr[:,:-1]"
        
        # padded__edge_attr = torch.cat((zero_vectors, edge_attr[:,:-1]), dim = 1) # "edge_attr[:,:-1]" is to drop the time-information scalar 
        #                                                                           # front-pad "edge_attr[:,:-1]" with zero-vector
        #                                                                           # so that vec-opertation can be done with "x_j"
        # message = torch.add(x_j, padded__edge_attr) # note that "x_j" was already padded in the data-processing step.

        # # zero_vectors = torch.zeros( edge_attr.shape[0], 
        # #                             x_j.shape[1] - edge_attr[:,:-1].shape[1], # "edge_attr[:,:-1]" is to drop the time-information scalar 
        # #                             device = edge_attr.device )               # "x_j.shape[1] - edge_attr[:,:-1].shape[1]" is to front-pad "edge_attr[:,:-1]"
        
        # # padded__edge_attr = torch.cat((zero_vectors, edge_attr[:,:-1]), dim = 1) # "edge_attr[:,:-1]" is to drop the time-information scalar 
        # #                                                                           # front-pad "edge_attr[:,:-1]" with zero-vector
        # #                                                                           # so that vec-opertation can be done with "x_j"

        # message = torch.add(x_j, padded__edge_attr) # note that "x_j" was already padded in the data-processing step.


    
        if self.signal_amplification_option == "signal_amplified__event_1gram":
            message = edge_attr[:,:-1]

        elif self.signal_amplification_option == "signal_amplified__event_1gram_nodetype_5bit":
            
            # Either compromise the unique-node, or not. 
            # It is tricky though,

            pass
        elif self.signal_amplification_option == "signal_amplified__event_1gram_nodetype_5bit_and_Ahoc_Identifier":
            pass


        message = edge_attr[:,:-1]
        return message


    def aggregate(self, 
                  inputs: Tensor,  # messages
                #   batch : Tensor, # added by JY @ 2023-06-27
                  index: Tensor,   # "indices of target-nodes (i of x_i)" , # index vector defines themapping from input elements to their location in output
                  edge_index, # b/c of signal-amplification
                  x, # b/c of signal-amplification
                  ptr: Tensor or None = None, 
                  dim_size: int or None = None) -> Tensor:  # |x_i| <-- here it's global
        
        # Refer to: https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#aggregation-operators

        # ***************************************************
        # JY @ 2023-07-28: Here, need both edge_index (src/tar) info, and node-x info. need all info.

        # 1. first identify all thread-nodes
        
        thread_node_indices = torch.nonzero(torch.all(torch.eq( x[:,:5], torch.tensor([0, 0, 0, 0, 1], device= x.device)), dim=1), as_tuple=False).flatten()
        edge_src_indices = edge_index[0]
        edge_tar_indices = edge_index[1]

        output = self.aggr_module( x = inputs,  index = index ) # returns for all x
        output_only_thread_nodes = output[thread_node_indices]

        # get an updated 'batch' variable equivlaent for thread-nodes based on 'ptr'
        batch_for_threadnodes = torch.bucketize(thread_node_indices, ptr, right=True) - 1 # -1 is to start from 0
        # output-dimension dim=0 should match the max of index - 1 (due to starting from indx 0)
        return output_only_thread_nodes, batch_for_threadnodes

    def __repr__(self) -> str:
         return f'{self.__class__.__name__}(nn={self.nn})'
    
#####################################################################################################
# (Version-1) : Direct incoropration of "signal-amplification". Separate from Dr.Xiaokui's suggestion.
class GNN_Signal_Amplification__ver1(GNNBasic):
    """
    GINConv (Graph Isomorphic Convolution). model, can handle multiple edge attributes
    paper: https://arxiv.org/pdf/1905.12265.pdf (ICLR, 2020) for model with edge handling capability
    """

    def __init__(self, 
                 
                 signal_amplification_option : str,

                 expanded_dim_node : int, 
                 edge_dim : int,
                 num_classes : int,

                 embedding_dim : list, # since considering 1-step convolution, so 1
                 ffn_dims: list, 
                  
                 dropout_level : float, 
                 activation_fn : nn.modules.activation = nn.ReLU,
                 conv_activation_fn : nn.modules.activation = nn.ReLU,
                 activation_fn_kwargs : dict = {}, 

                 pool : GNNPool = GlobalMaxPool, # 
                 neighborhood_aggr : str = "add"  
                ):

        super().__init__()

        if len(embedding_dim) > 2:
            ValueError("Currently considering 1-step convolution. Thus len(embedding_dim) should be 1.")


        self.conv1 = GNN_Signal_Amplification_Conv__Version_1(
            nn = nn.Sequential(*( [ nn.Linear(expanded_dim_node, embedding_dim[0]) ] )),
            signal_amplification_option= signal_amplification_option,
            edge_dim = edge_dim,
            aggr = neighborhood_aggr)  # the first layer

        
        self.act_fn = activation_fn( **activation_fn_kwargs )
        # self.conv_act_fn = conv_activation_fn
        # self.act_fns = nn.ModuleList(
        #     [ activation_fn( **activation_fn_kwargs ) for _ in range(num_layer - 1) ]
        # ) 

        self.readout = pool()       
        # ----------------------------------------------------------------------------------------------------------------

        # Need caution here.
        ffn__num_layers = len(ffn_dims)
        layers = []
        self.ffn_first_layer = [ nn.Linear( embedding_dim[0], ffn_dims[0]), nn.ReLU(), nn.Dropout(p=dropout_level) ] 
        for i in range(ffn__num_layers - 1):
            layers.append(
                    [nn.Linear(ffn_dims[i], ffn_dims[i + 1]), 
                     nn.ReLU(), nn.Dropout(p=dropout_level)]
            )
        flattend_layers = [element for sublist in layers for element in sublist]

        self.ffn_last_layer = [ nn.Linear( ffn_dims[-1], num_classes ), nn.Softmax(dim=1) ]

        self.ffn = nn.Sequential( *(self.ffn_first_layer + flattend_layers + self.ffn_last_layer) )
            
        # self.convs = nn.ModuleList(layers)

        # self.ffn = nn.Sequential(*(
        #         [nn.Linear( embedding_dim[0], (dim_hidden[-1] + num_classes) // 2 )] +
        #         [nn.ReLU(), nn.Dropout(p=dropout_level), nn.Linear( (dim_hidden[-1] + num_classes) // 2 , num_classes), nn.Softmax(dim=1)]
        # ))
        

        return


    def forward(self, *args, **kwargs) -> torch.Tensor:
        """
        Does a single forward pass for the complete model
        """
        x, edge_index, edge_attr, batch, ptr, y, name = self.arguments_read(*args, **kwargs)


        # 1. first conv. pass -> uses the original data sample
        # if self.conv_act_fn:
        
        #     post_conv, batch_for_thread_nodes = self.conv1(x=x, edge_index=edge_index, edge_attr=edge_attr, 
        #                                                     batch = batch, # added by JY @ 2023-06-27
        #                                                     ptr = ptr,
                                                            
        #                                                     # y = y,  # added by JY @ 2023-07-05
        #                                                     # name = name,  # added by JY @ 2023-07-05
        #                                                     )
        #     post_conv = self.conv_act_fn( post_conv )

        # print(f"ptr: {ptr}", flush=True)


        # else:
        post_conv, batch_for_thread_nodes = self.conv1(x=x, edge_index=edge_index, edge_attr=edge_attr, 
                                                        batch = batch, 
                                                        ptr = ptr,
                                                # added by JY @ 2023-06-27

                                                # y = y,  # added by JY @ 2023-07-05
                                                # name = name,  # added by JY @ 2023-07-05
                                                )

        # # print(post_conv)
        # for conv, act_fn in zip(self.convs, self.act_fns):
        #     # 2. iteratively do GCN convolution
        #     post_conv = act_fn(conv(x=post_conv, edge_index=edge_index, edge_attr=edge_attr))


        # 3. Use the readout (i.e., pooling)

        #   Added by JY @ 2023-07-28: Need an updated batch using ptr.
        # out_readout = self.readout(post_conv, batch)
        out_readout = self.readout(post_conv, batch_for_thread_nodes)


        # 4. the class probabilities
        out = self.ffn(out_readout)
        #[p.requires_grad for p in self.ffn.parameters()]

        return out


    def get_emb(self, *args, **kwargs) -> torch.Tensor:
        """
        Auxilary function if node embeddings are required seperately
        works similar to the forward pass above, but without readout
        """
        x, edge_index, edge_attr, batch = self.arguments_read(*args, **kwargs)
        post_conv = self.act_fn1(self.conv1(x=x, edge_index=edge_index, edge_attr=edge_attr))
        for conv, act_fn in zip(self.convs, self.act_fns):
            post_conv = act_fn(conv(x=post_conv, edge_index=edge_index, edge_attr=edge_attr))
        return post_conv


#######################################################################################################################################################
#######################################################################################################################################################
#######################################################################################################################################################
#######################################################################################################################################################



#######################################################################################################################################################
from typing import Optional

from torch import Tensor
from torch.nn import LSTM

from torch_geometric.nn.aggr import Aggregation

from torch_geometric.utils import scatter

# from torch_geometric.nn.base import Aggregation


# JY @ 2023-07-30: For GNN-signaml-amplification version-2
class Weighted_Sum_Aggregator(nn.Module):

    def __init__(self, input_size):
        super(Weighted_Sum_Aggregator, self).__init__()
        self.input_size = input_size
        self.weights = nn.Parameter(torch.Tensor(input_size)) # learnable. confirmed error backpropagates to here.
        self.reset_parameters()


    def reset_parameters(self):
        nn.init.ones_(self.weights) # weights are initialized as one



    # Added by JY @ 2023-07-30 : Note that this is nn.Module subclass not a usual Aggregation-class.
    def reduce(self,
                x: Tensor, index: Optional[Tensor] = None,
                ptr: Optional[Tensor] = None, dim_size: Optional[int] = None,
                dim: int = -2, reduce: str = 'sum') -> Tensor:

        assert index is not None
        return scatter(x, index, dim, dim_size, reduce)


    def forward(self, x: Tensor, index: Optional[Tensor] = None,
                ptr: Optional[Tensor] = None, dim_size: Optional[int] = None,
                dim: int = -2) -> Tensor:
        # Ensure that the number of inputs matches the number of weights
        if x.shape[1] != self.input_size:
            raise ValueError("input-dimension must match the number of weights, since doing element-wise multiplication")

        # Element-wise-multiplication-using-broatcasting of inputs and weights, and then summing them upw
        # -- Another option could be weight afeter reduce but either could be fine for now, since not sure which one will be better
        # weighted_inputs = x * self.weights.unsqueeze(0)

                # https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/aggr/basic.html#SumAggregation
                # https://github.com/pyg-team/pytorch_geometric/blob/4889a1e0852ef7a3d04d125432b8c08dde023ef9/torch_geometric/nn/aggr/base.py
                # https://github.com/pyg-team/pytorch_geometric/blob/4889a1e0852ef7a3d04d125432b8c08dde023ef9/torch_geometric/utils/segment.py#L7
                # https://pytorch-scatter.readthedocs.io/en/latest/functions/segment_csr.html
                # https://pytorch-scatter.readthedocs.io/en/latest/_modules/torch_scatter/segment_csr.html#segment_csr

        # https://github.com/pyg-team/pytorch_geometric/blob/4889a1e0852ef7a3d04d125432b8c08dde023ef9/torch_geometric/nn/aggr/base.py#L160
        # return nn.aggr.reduce(torch_weighted_inputs, index, ptr, dim_size, dim, reduce='sum')
        summ = self.reduce(x, index, dim, dim_size, reduce = 'sum')
        weighted_sum = summ * self.weights.unsqueeze(0)
        

        # return self.reduce(weighted_inputs, index, dim, dim_size, reduce = 'sum')    # https://github.com/pyg-team/pytorch_geometric/blob/4889a1e0852ef7a3d04d125432b8c08dde023ef9/torch_geometric/nn/aggr/base.py#L169

        # print(f"self.weights: {self.weights}", flush=True)
        return weighted_sum    # https://github.com/pyg-team/pytorch_geometric/blob/4889a1e0852ef7a3d04d125432b8c08dde023ef9/torch_geometric/nn/aggr/base.py#L169


        # aggregated_output = torch.stack(weighted_inputs).sum(dim=0)

        # return aggregated_output


    # def forward(self, x: Tensor, index: Optional[Tensor] = None,
    #             ptr: Optional[Tensor] = None, dim_size: Optional[int] = None,
    #             dim: int = -2) -> Tensor:

    #     # https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/aggr/basic.html#SumAggregation
    #     # https://github.com/pyg-team/pytorch_geometric/blob/4889a1e0852ef7a3d04d125432b8c08dde023ef9/torch_geometric/nn/aggr/base.py
    #     # https://github.com/pyg-team/pytorch_geometric/blob/4889a1e0852ef7a3d04d125432b8c08dde023ef9/torch_geometric/utils/segment.py#L7
    #     # https://pytorch-scatter.readthedocs.io/en/latest/functions/segment_csr.html
    #     # https://pytorch-scatter.readthedocs.io/en/latest/_modules/torch_scatter/segment_csr.html#segment_csr
    #     return self.reduce(x, index, ptr, dim_size, dim, reduce='sum')





# (Version-2) : Weighted-sum aggregation.
class GNN_Signal_Amplification_Conv__Version_2( MessagePassing_JY_for_GNN_Signal_Amplification__Version2 ):
    r"""The modified :class:`GINConv` operator from the `"Strategies for
    Pre-training Graph Neural Networks" <https://arxiv.org/abs/1905.12265>`_
    paper
    .. math::
        \mathbf{x}^{\prime}_i = h_{\mathbf{\Theta}} \left( (1 + \epsilon) \cdot
        \mathbf{x}_i + \sum_{j \in \mathcal{N}(i)} \mathrm{ReLU}
        ( \mathbf{x}_j + \mathbf{e}_{j,i} ) \right)
    that is able to incorporate edge features :math:`\mathbf{e}_{j,i}` into
    the aggregation procedure.
    Args:
        nn (torch.nn.Module): A neural network :math:`h_{\mathbf{\Theta}}` that
            maps node features :obj:`x` of shape :obj:`[-1, in_channels]` to
            shape :obj:`[-1, out_channels]`, *e.g.*, defined by
            :class:`torch.nn.Sequential`.
        eps (float, optional): (Initial) :math:`\epsilon`-value.
            (default: :obj:`0.`)
        train_eps (bool, optional): If set to :obj:`True`, :math:`\epsilon`
            will be a trainable parameter. (default: :obj:`False`)
        edge_dim (int, optional): Edge feature dimensionality. If set to
            :obj:`None`, node and edge feature dimensionality is expected to
            match. Other-wise, edge features are linearly transformed to match
            node feature dimensionality. (default: :obj:`None`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    Shapes:
        - **input:**
          node features :math:`(|\mathcal{V}|, F_{in})` or
          :math:`((|\mathcal{V_s}|, F_{s}), (|\mathcal{V_t}|, F_{t}))`
          if bipartite,
          edge indices :math:`(2, |\mathcal{E}|)`,
          edge features :math:`(|\mathcal{E}|, D)` *(optional)*
        - **output:** node features :math:`(|\mathcal{V}|, F_{out})` or
          :math:`(|\mathcal{V}_t|, F_{out})` if bipartite
    """
    def __init__(self, 
                #  nn: torch.nn.Module, 
                 signal_amplification_option : str,

                 edge_dim: Optional[int] = None,
                 # added by JY @ 2023-05-24
                 aggr: str = "weighted_sum",

                 **kwargs):
        

        #####################################################################################################################
        # kwargs.setdefault('aggr', aggr )
        if aggr != "weighted_sum":
            raise ValueError("'GNN_Signal_Amplification_Conv__Version_2' is for 'weighted_sum' aggregation")
        super().__init__(**kwargs)  
        self.aggr_module = None # just to make sure no conflict with what happens within 'super().__init__(**kwargs)' internally
        self.aggr = aggr # "weighted_sum"


        # self.nn = nn
        self.signal_amplification_option = signal_amplification_option
        self.edge_dim = edge_dim

        if self.signal_amplification_option == "signal_amplified__event_1gram":
            self.aggr_module = Weighted_Sum_Aggregator( input_size = self.edge_dim - 1 )

        self.reset_parameters()


    def reset_parameters(self):
        super().reset_parameters() # aggr-module 
        # reset(self.nn)



    def forward(self, x: Union[Tensor, OptPairTensor], 
                edge_index: Adj,
                batch: Tensor,
                ptr,
                edge_attr: OptTensor = None,
                size: Size = None) -> Tensor:

        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        # propagate_type: (x: OptPairTensor)
        out, batch_for_threadnodes = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=size, 
                                                    ptr = ptr)

        # x_r = x[1]
        # if x_r is not None:
        #     # out = out + (1 + self.eps) * x_r

        # "make 'out' to have torch.float64" to avoid "mat1 and mat2 must have the same dtype"
        # out = out.to(torch.float64)
        return out, batch_for_threadnodes



    def message(self, 
                x_j: Tensor, 
                edge_attr: Tensor, 
                x, 
                edge_index) -> Tensor:
        
        # This function is called in propagate()


        # Added by JY @ 2023-05-24:
        assert edge_attr.shape[0] == x_j.shape[0], "edge and source-node number mismatch"
        # What I will do is concatenate the source-node-feature and edge-feature.
        # (node)<-----(edge; edge-feat)-----(sourc-node; node-feat)

    
        if self.signal_amplification_option == "signal_amplified__event_1gram":
            message = edge_attr[:,:-1]

        elif self.signal_amplification_option == "signal_amplified__event_1gram_nodetype_5bit":
            
            # Either compromise the unique-node, or not. 
            # It is tricky though,

            pass
        elif self.signal_amplification_option == "signal_amplified__event_1gram_nodetype_5bit_and_Ahoc_Identifier":
            pass


        return message


    def aggregate(self, 
                  inputs: Tensor,  # messages
                #   batch : Tensor, # added by JY @ 2023-06-27
                  index: Tensor,   # "indices of target-nodes (i of x_i)" , # index vector defines themapping from input elements to their location in output
                  edge_index, # b/c of signal-amplification
                  x, # b/c of signal-amplification
                  ptr: Tensor or None = None, 
                  dim_size: int or None = None) -> Tensor:  # |x_i| <-- here it's global
        
        # Refer to: https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#aggregation-operators

        # ***************************************************
        # JY @ 2023-07-28: Here, need both edge_index (src/tar) info, and node-x info. need all info.

        # 1. first identify all thread-nodes
        
        thread_node_indices = torch.nonzero(torch.all(torch.eq( x[:,:5], torch.tensor([0, 0, 0, 0, 1], device= x.device)), dim=1), as_tuple=False).flatten()
        # edge_src_indices = edge_index[0]
        # edge_tar_indices = edge_index[1]

        #################################################################
        # TODO: Find a way to do this mem-efficiently

        # following leads to weird b/c self.aggr_module is designed to return same-dim as 'x'
        # Here it got slow b/c of cpu() ;
        # 1: go to panther since it has more cuda-memory 
        # 2. lower the batch-size
        # 3. find a more efficient way that uses cuda beter than following.

        # Since "mask = torch.any(index[:, None] == thread_node_indices), dim=1)" can lead to memory issue, divide it by chunck then perform.

        # mask_debug = torch.any(index[:, None] == thread_node_indices, dim=1) 

        num_chunks = 5  # Adjust the chunk size as needed based on your available memory and tensor sizes
        # Split the index and thread_node_indices tensors into smaller chunks
        index_chunks = torch.chunk(index, chunks= num_chunks, dim=0)
        mask_chunks = []
        for index_chunk in index_chunks:
            mask_chunk = torch.any(index_chunk[:, None] == thread_node_indices, dim=1)
            mask_chunks.append(mask_chunk)
        mask = torch.cat(mask_chunks, dim=0).to(inputs.device)


        # mask_2 = (index.unsqueeze(1) == thread_node_indices).any(dim=1) # this leads to cuda-out-memory

        # print(f"index.shape: {index.shape}")        
        # print(f"thread_node_indices.shape: {thread_node_indices.shape}")
        # print(f"mask.shape: {mask.shape}")        
        # print(f"mask_debug.shape: {mask_debug.shape}")        
        # print(torch.equal(mask, mask_debug))

        inputs_of_target_nodes__threadnode = inputs[mask]
        indices_of_target_nodes__threadnode= index[mask]
        # 아니면 여기서 "inputs_of_target_nodes__threadnode"의 각 index를 uniuq하게 map 해야됨; 해보자<-- ㄴㄴ ptr때매 <-- ㄴㄴ 그래도해봐

        # JY: Get unique-thread-node-lements,
        #     and 'inverse_indices' corresponds to 'indices_of_target_nodes__threadnode' but replaced with their inverse_indies. (very helpful for my case.)
        #     "return_inverse: Whether to also return the indices for where elements in the original input ended up in the returned unique list."
        #     "inverse_indices" here is EXACTLY what I needed. b/c later self.aggr_module scatter function 'scatters'
        #
        #     Note that "inverse_indices" has correspondence to "indices_of_target_nodes__threadnode" but range-adjusted.

        unique_elements, inverse_indices = torch.unique(indices_of_target_nodes__threadnode, return_inverse=True) # 

        # JY: Following commented-out part could be used for verification later, but is not needed.
            # sorted_elements, sorted_indices = torch.sort(unique_elements)
        #     # # Step 3: Create a mapping from original values to sorted values
            # value_to_index = {value.item(): index for index, value in enumerate(sorted_elements)}
        #     # Map each value in the original tensor to its sorted value starting from 0
            # sorted_tensor = sorted_indices[inverse_indices]

        # JY: self.aggr_module's scatter will return a scattered output of filling all non-provided indices with zero-vectors
        #     Therefore, output__threadnodes.shape == thread_node_indices.shape,
        #     and there is a correspondence bettween indices.
        output__threadnodes = self.aggr_module( x = inputs_of_target_nodes__threadnode,  
                                               index = inverse_indices) 
        #################################################################

        # --- ** Here, output.shape[0] matches number-of-target-nodes.
        # # sorted(list(set(indices_of_target_nodes__threadnode.tolist())) )
        # len(set(index.tolist()))
        # output = self.aggr_module( x = inputs,  index = index)
        # output.shape

        #############################################################################
        # JY:
        # Get an updated 'batch' variable equivlaent for thread-nodes based on 'ptr'
        # JY: Since 'output_threadnodes' corresponds to 'thread_node_indices', ( check "value_to_index" )
        #     could use torch.bucketize to 'thread_node_indicies' which does the job for 'output_threadnodes'.
        batch_for_threadnodes = torch.bucketize(thread_node_indices, ptr, right=True) - 1 # -1 is to start from 0
        
        del unique_elements, indices_of_target_nodes__threadnode, inputs_of_target_nodes__threadnode, mask, index_chunks, thread_node_indices

        # output-dimension dim=0 should match the max of index - 1 (due to starting from indx 0)
        return output__threadnodes, batch_for_threadnodes


    
#####################################################################################################
class GNN_Signal_Amplification__ver2(GNNBasic):
    """
    GINConv (Graph Isomorphic Convolution). model, can handle multiple edge attributes
    paper: https://arxiv.org/pdf/1905.12265.pdf (ICLR, 2020) for model with edge handling capability
    """

    def __init__(self, 
                 
                 signal_amplification_option : str,

                 expanded_dim_node : int, 
                 edge_dim : int,
                 num_classes : int,

                 ffn_dims: list, 
                  
                 dropout_level : float, 
                 activation_fn : nn.modules.activation = nn.ReLU,
                #  conv_activation_fn : nn.modules.activation = nn.ReLU,
                 activation_fn_kwargs : dict = {}, 

                 pool : GNNPool = GlobalMaxPool, # 
                 neighborhood_aggr : str = "weighted_sum"  
                ):
        super().__init__()

        # 1-step Convolution.
        self.conv1 = GNN_Signal_Amplification_Conv__Version_2( signal_amplification_option = signal_amplification_option,
                                                               edge_dim = edge_dim,
                                                               aggr = neighborhood_aggr )
        
        self.act_fn = activation_fn( **activation_fn_kwargs )
        # self.conv_act_fn = conv_activation_fn

        self.readout = pool()       
        # ----------------------------------------------------------------------------------------------------------------

        # Need caution here.
        ffn__num_layers = len(ffn_dims)
        layers = []
        self.ffn_first_layer = [ nn.Linear( expanded_dim_node , ffn_dims[0]), nn.ReLU(), nn.Dropout(p=dropout_level) ] 
        for i in range(ffn__num_layers - 1):
            layers.append(
                    [nn.Linear(ffn_dims[i], ffn_dims[i + 1]), 
                     nn.ReLU(), nn.Dropout(p=dropout_level)]
            )
        flattend_layers = [element for sublist in layers for element in sublist]

        self.ffn_last_layer = [ nn.Linear( ffn_dims[-1], num_classes ), nn.Softmax(dim=1) ]

        self.ffn = nn.Sequential( *(self.ffn_first_layer + flattend_layers + self.ffn_last_layer) )
            
        return


    def forward(self, *args, **kwargs) -> torch.Tensor:
        """
        Does a single forward pass for the complete model
        """
        x, edge_index, edge_attr, batch, ptr, y, name = self.arguments_read(*args, **kwargs)


        post_conv, batch_for_thread_nodes = self.conv1(x=x, 
                                                       edge_index=edge_index, 
                                                       edge_attr=edge_attr, 
                                                       batch = batch, 
                                                       ptr = ptr,
                                                )
        # 3. Use the readout (i.e., pooling)

        #   Added by JY @ 2023-07-28: Need an updated batch using ptr.
        # out_readout = self.readout(post_conv, batch)
        out_readout = self.readout(post_conv, batch_for_thread_nodes)


        # 4. the class probabilities
        out = self.ffn(out_readout)
        #[p.requires_grad for p in self.ffn.parameters()]

        return out


    def get_emb(self, *args, **kwargs) -> torch.Tensor:
        """
        Auxilary function if node embeddings are required seperately
        works similar to the forward pass above, but without readout
        """
        x, edge_index, edge_attr, batch = self.arguments_read(*args, **kwargs)
        post_conv = self.act_fn1(self.conv1(x=x, edge_index=edge_index, edge_attr=edge_attr))
        for conv, act_fn in zip(self.convs, self.act_fns):
            post_conv = act_fn(conv(x=post_conv, edge_index=edge_index, edge_attr=edge_attr))
        return post_conv




#######################################################################################################################################################
#######################################################################################################################################################
#######################################################################################################################################################
#######################################################################################################################################################

# import torch
# import torch.nn.functional as F
# from torch import Tensor
# from torch.nn import Parameter

# from torch_geometric.nn.conv import MessagePassing
# from torch_geometric.nn.dense.linear import Linear
# from torch_geometric.nn.inits import glorot, zeros
# from torch_geometric.typing import NoneType  # noqa
# from torch_geometric.typing import (
#     Adj,
#     OptPairTensor,
#     OptTensor,
#     Size,
#     SparseTensor,
#     torch_sparse,
# )
# from torch_geometric.utils import (
#     add_self_loops,
#     is_torch_sparse_tensor,
#     remove_self_loops,
#     softmax,
# )
# from torch_geometric.utils.sparse import set_sparse_value


# 
# (GNN_Signal_Amplification__Version3 ) : 
# #             Tries to learn edge-weights(attentions)--in-GAT-style
#               for each edge during our customized-message-passing(signal-amplficaition)
#               Basic structure inherited from GAT_conv
#               https://github.com/pyg-team/pytorch_geometric/blob/master/torch_geometric/nn/conv/gat_conv.py
# class GNN_Signal_Amplification_Conv__Version3(MessagePassing_JY_for_GNN_Signal_Amplification__Version2):
#     r"""The graph attentional operator from the `"Graph Attention Networks"
#     <https://arxiv.org/abs/1710.10903>`_ paper

#     .. math::
#         \mathbf{x}^{\prime}_i = \alpha_{i,i}\mathbf{\Theta}\mathbf{x}_{i} +
#         \sum_{j \in \mathcal{N}(i)} \alpha_{i,j}\mathbf{\Theta}\mathbf{x}_{j},

#     where the attention coefficients :math:`\alpha_{i,j}` are computed as

#     .. math::
#         \alpha_{i,j} =
#         \frac{
#         \exp\left(\mathrm{LeakyReLU}\left(\mathbf{a}^{\top}
#         [\mathbf{\Theta}\mathbf{x}_i \, \Vert \, \mathbf{\Theta}\mathbf{x}_j]
#         \right)\right)}
#         {\sum_{k \in \mathcal{N}(i) \cup \{ i \}}
#         \exp\left(\mathrm{LeakyReLU}\left(\mathbf{a}^{\top}
#         [\mathbf{\Theta}\mathbf{x}_i \, \Vert \, \mathbf{\Theta}\mathbf{x}_k]
#         \right)\right)}.

#     If the graph has multi-dimensional edge features :math:`\mathbf{e}_{i,j}`,
#     the attention coefficients :math:`\alpha_{i,j}` are computed as

#     .. math::
#         \alpha_{i,j} =
#         \frac{
#         \exp\left(\mathrm{LeakyReLU}\left(\mathbf{a}^{\top}
#         [\mathbf{\Theta}\mathbf{x}_i \, \Vert \, \mathbf{\Theta}\mathbf{x}_j
#         \, \Vert \, \mathbf{\Theta}_{e} \mathbf{e}_{i,j}]\right)\right)}
#         {\sum_{k \in \mathcal{N}(i) \cup \{ i \}}
#         \exp\left(\mathrm{LeakyReLU}\left(\mathbf{a}^{\top}
#         [\mathbf{\Theta}\mathbf{x}_i \, \Vert \, \mathbf{\Theta}\mathbf{x}_k
#         \, \Vert \, \mathbf{\Theta}_{e} \mathbf{e}_{i,k}]\right)\right)}.

#     Args:
#         in_channels (int or tuple): Size of each input sample, or :obj:`-1` to
#             derive the size from the first input(s) to the forward method.
#             A tuple corresponds to the sizes of source and target
#             dimensionalities.
#         out_channels (int): Size of each output sample.
#         heads (int, optional): Number of multi-head-attentions.
#             (default: :obj:`1`)
#         concat (bool, optional): If set to :obj:`False`, the multi-head
#             attentions are averaged instead of concatenated.
#             (default: :obj:`True`)
#         negative_slope (float, optional): LeakyReLU angle of the negative
#             slope. (default: :obj:`0.2`)
#         dropout (float, optional): Dropout probability of the normalized
#             attention coefficients which exposes each node to a stochastically
#             sampled neighborhood during training. (default: :obj:`0`)
#         add_self_loops (bool, optional): If set to :obj:`False`, will not add
#             self-loops to the input graph. (default: :obj:`True`)
#         edge_dim (int, optional): Edge feature dimensionality (in case
#             there are any). (default: :obj:`None`)
#         fill_value (float or torch.Tensor or str, optional): The way to
#             generate edge features of self-loops (in case
#             :obj:`edge_dim != None`).
#             If given as :obj:`float` or :class:`torch.Tensor`, edge features of
#             self-loops will be directly given by :obj:`fill_value`.
#             If given as :obj:`str`, edge features of self-loops are computed by
#             aggregating all features of edges that point to the specific node,
#             according to a reduce operation. (:obj:`"add"`, :obj:`"mean"`,
#             :obj:`"min"`, :obj:`"max"`, :obj:`"mul"`). (default: :obj:`"mean"`)
#         bias (bool, optional): If set to :obj:`False`, the layer will not learn
#             an additive bias. (default: :obj:`True`)
#         **kwargs (optional): Additional arguments of
#             :class:`torch_geometric.nn.conv.MessagePassing`.

#     Shapes:
#         - **input:**
#           node features :math:`(|\mathcal{V}|, F_{in})` or
#           :math:`((|\mathcal{V_s}|, F_{s}), (|\mathcal{V_t}|, F_{t}))`
#           if bipartite,
#           edge indices :math:`(2, |\mathcal{E}|)`,
#           edge features :math:`(|\mathcal{E}|, D)` *(optional)*
#         - **output:** node features :math:`(|\mathcal{V}|, H * F_{out})` or
#           :math:`((|\mathcal{V}_t|, H * F_{out})` if bipartite.
#           If :obj:`return_attention_weights=True`, then
#           :math:`((|\mathcal{V}|, H * F_{out}),
#           ((2, |\mathcal{E}|), (|\mathcal{E}|, H)))`
#           or :math:`((|\mathcal{V_t}|, H * F_{out}), ((2, |\mathcal{E}|),
#           (|\mathcal{E}|, H)))` if bipartite
#     """
#     def __init__(
#         self,
#         in_channels: Union[int, Tuple[int, int]],
#         out_channels: int,
#         heads: int = 1,
#         concat: bool = True,
#         negative_slope: float = 0.2,
#         dropout: float = 0.0,
#         add_self_loops: bool = True,
#         edge_dim: Optional[int] = None,
#         fill_value: Union[float, Tensor, str] = 'mean',
#         bias: bool = True,
#         **kwargs,
#     ):
#         kwargs.setdefault('aggr', 'add')
#         super().__init__(node_dim=0, **kwargs)

#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.heads = heads
#         self.concat = concat
#         self.negative_slope = negative_slope
#         self.dropout = dropout
#         self.add_self_loops = add_self_loops
#         self.edge_dim = edge_dim
#         self.fill_value = fill_value

#         # In case we are operating in bipartite graphs, we apply separate
#         # transformations 'lin_src' and 'lin_dst' to source and target nodes:
#         if isinstance(in_channels, int):
#             self.lin_src = Linear(in_channels, heads * out_channels,
#                                   bias=False, weight_initializer='glorot')
#             self.lin_dst = self.lin_src
#         else:
#             self.lin_src = Linear(in_channels[0], heads * out_channels, False,
#                                   weight_initializer='glorot')
#             self.lin_dst = Linear(in_channels[1], heads * out_channels, False,
#                                   weight_initializer='glorot')

#         # The learnable parameters to compute attention coefficients:
#         self.att_src = Parameter(torch.empty(1, heads, out_channels))
#         self.att_dst = Parameter(torch.empty(1, heads, out_channels))

#         if edge_dim is not None:
#             self.lin_edge = Linear(edge_dim, heads * out_channels, bias=False,
#                                    weight_initializer='glorot')
#             self.att_edge = Parameter(torch.empty(1, heads, out_channels))
#         else:
#             self.lin_edge = None
#             self.register_parameter('att_edge', None)

#         if bias and concat:
#             self.bias = Parameter(torch.empty(heads * out_channels))
#         elif bias and not concat:
#             self.bias = Parameter(torch.empty(out_channels))
#         else:
#             self.register_parameter('bias', None)

#         self.reset_parameters()

#     def reset_parameters(self):
#         super().reset_parameters()
#         self.lin_src.reset_parameters()
#         self.lin_dst.reset_parameters()
#         if self.lin_edge is not None:
#             self.lin_edge.reset_parameters()
#         glorot(self.att_src)
#         glorot(self.att_dst)
#         glorot(self.att_edge)
#         zeros(self.bias)

#     def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
#                 edge_attr: OptTensor = None, size: Size = None,
#                 return_attention_weights=None):
#         # type: (Union[Tensor, OptPairTensor], Tensor, OptTensor, Size, NoneType) -> Tensor  # noqa
#         # type: (Union[Tensor, OptPairTensor], SparseTensor, OptTensor, Size, NoneType) -> Tensor  # noqa
#         # type: (Union[Tensor, OptPairTensor], Tensor, OptTensor, Size, bool) -> Tuple[Tensor, Tuple[Tensor, Tensor]]  # noqa
#         # type: (Union[Tensor, OptPairTensor], SparseTensor, OptTensor, Size, bool) -> Tuple[Tensor, SparseTensor]  # noqa
#         r"""Runs the forward pass of the module.

#         Args:
#             return_attention_weights (bool, optional): If set to :obj:`True`,
#                 will additionally return the tuple
#                 :obj:`(edge_index, attention_weights)`, holding the computed
#                 attention weights for each edge. (default: :obj:`None`)
#         """
#         # NOTE: attention weights will be returned whenever
#         # `return_attention_weights` is set to a value, regardless of its
#         # actual value (might be `True` or `False`). This is a current somewhat
#         # hacky workaround to allow for TorchScript support via the
#         # `torch.jit._overload` decorator, as we can only change the output
#         # arguments conditioned on type (`None` or `bool`), not based on its
#         # actual value.

#         H, C = self.heads, self.out_channels

#         # We first transform the input node features. If a tuple is passed, we
#         # transform source and target node features via separate weights:
#         if isinstance(x, Tensor):
#             assert x.dim() == 2, "Static graphs not supported in 'GATConv'"
#             x_src = x_dst = self.lin_src(x).view(-1, H, C)
#         else:  # Tuple of source and target node features:
#             x_src, x_dst = x
#             assert x_src.dim() == 2, "Static graphs not supported in 'GATConv'"
#             x_src = self.lin_src(x_src).view(-1, H, C)
#             if x_dst is not None:
#                 x_dst = self.lin_dst(x_dst).view(-1, H, C)

#         x = (x_src, x_dst)

#         # Next, we compute node-level attention coefficients, both for source
#         # and target nodes (if present):
#         alpha_src = (x_src * self.att_src).sum(dim=-1)
#         alpha_dst = None if x_dst is None else (x_dst * self.att_dst).sum(-1)
#         alpha = (alpha_src, alpha_dst)

#         if self.add_self_loops:
#             if isinstance(edge_index, Tensor):
#                 # We only want to add self-loops for nodes that appear both as
#                 # source and target nodes:
#                 num_nodes = x_src.size(0)
#                 if x_dst is not None:
#                     num_nodes = min(num_nodes, x_dst.size(0))
#                 num_nodes = min(size) if size is not None else num_nodes
#                 edge_index, edge_attr = remove_self_loops(
#                     edge_index, edge_attr)
#                 edge_index, edge_attr = add_self_loops(
#                     edge_index, edge_attr, fill_value=self.fill_value,
#                     num_nodes=num_nodes)
#             elif isinstance(edge_index, SparseTensor):
#                 if self.edge_dim is None:
#                     edge_index = torch_sparse.set_diag(edge_index)
#                 else:
#                     raise NotImplementedError(
#                         "The usage of 'edge_attr' and 'add_self_loops' "
#                         "simultaneously is currently not yet supported for "
#                         "'edge_index' in a 'SparseTensor' form")

#         # edge_updater_type: (alpha: OptPairTensor, edge_attr: OptTensor)
#         alpha = self.edge_updater(edge_index, alpha=alpha, edge_attr=edge_attr)

#         # propagate_type: (x: OptPairTensor, alpha: Tensor)
#         out = self.propagate(edge_index, x=x, alpha=alpha, size=size)

#         if self.concat:
#             out = out.view(-1, self.heads * self.out_channels)
#         else:
#             out = out.mean(dim=1)

#         if self.bias is not None:
#             out = out + self.bias

#         if isinstance(return_attention_weights, bool):
#             if isinstance(edge_index, Tensor):
#                 if is_torch_sparse_tensor(edge_index):
#                     # TODO TorchScript requires to return a tuple
#                     adj = set_sparse_value(edge_index, alpha)
#                     return out, (adj, alpha)
#                 else:
#                     return out, (edge_index, alpha)
#             elif isinstance(edge_index, SparseTensor):
#                 return out, edge_index.set_value(alpha, layout='coo')
#         else:
#             return out

#     def edge_update(self, alpha_j: Tensor, alpha_i: OptTensor,
#                     edge_attr: OptTensor, index: Tensor, ptr: OptTensor,
#                     size_i: Optional[int]) -> Tensor:
#         # Given edge-level attention coefficients for source and target nodes,
#         # we simply need to sum them up to "emulate" concatenation:
#         alpha = alpha_j if alpha_i is None else alpha_j + alpha_i
#         if index.numel() == 0:
#             return alpha
#         if edge_attr is not None and self.lin_edge is not None:
#             if edge_attr.dim() == 1:
#                 edge_attr = edge_attr.view(-1, 1)
#             edge_attr = self.lin_edge(edge_attr)
#             edge_attr = edge_attr.view(-1, self.heads, self.out_channels)
#             alpha_edge = (edge_attr * self.att_edge).sum(dim=-1)
#             alpha = alpha + alpha_edge

#         alpha = F.leaky_relu(alpha, self.negative_slope)
#         alpha = softmax(alpha, index, ptr, size_i)
#         alpha = F.dropout(alpha, p=self.dropout, training=self.training)
#         return alpha

#     def message(self, x_j: Tensor, alpha: Tensor) -> Tensor:
#         return alpha.unsqueeze(-1) * x_j

#     def __repr__(self) -> str:
#         return (f'{self.__class__.__name__}({self.in_channels}, '
#                 f'{self.out_channels}, heads={self.heads})')




# (GNN_Signal_Amplification__Version2 ) : 
# #             Tries to learn edge-weights(attentions)--in-GAT-style
#               for each edge during our customized-message-passing(signal-amplficaition)
# class GNN_Signal_Amplification__Version2(GNNBasic):
#     """
#     GATConv (Graph Attention Network Convolution). model, can handle multiple edge attributes
#     paper: https://arxiv.org/pdf/1710.10903.pdf
#     """

#                                             dim_node = dim_node__expanded_for_compatibility_with_signal_amplification,
#                                             signal_amplification_option = signal_amplification_option,

#                                             edge_dim = _dim_edge,
#                                             num_classes = _num_classes, 
#                                             num_heads = 1,

#                                             ffn_dims=[19, 8],                                            
#                                             activation_fn = nn.ReLU,
#                                             conv_activation_fn = nn.ReLU,                                            
#                                             neighborhood_aggr= "add", # 'add' or 'mean'
#                                             pool = GlobalMaxPool,

#                                             dropout_level = dropout_level, 

#     def __init__(self, 
#                  dim_node : int, 
#                  dim_hidden : list, 
#                  num_classes : int, 
#                  dropout_level : float, 
#                  edge_dim : int , 
#                  num_heads : int = 1,
#                  # Below Added by JY @ 2022-06-29
#                  activation_fn : nn.modules.activation = nn.ReLU,
#                  activation_fn_kwargs : dict = {}, 
#                  pool : GNNPool = GlobalMeanPool, 
#                 ):
#         """
#         dim_node (int): the #features per node
#         dim_hidden (list): the list of hidden sizes after each convolution step
#         num_classes (int): the number of classes
#         dropout_level (float): the dropout probability
#         edge_dim (int): number of edge attributes
#         num_heads (int): number of multi-headed attentions
#         # Below Added by JY @ 2022-06-29
#         pooling (nn.Module): pooling method
#         activation_fn (nn.Module): activation function
#         activation_fn_kwargs (dict): activation function keyword-arguments
#         """

#         super().__init__()
#         num_layer = len(dim_hidden)

#         self.conv1 = gnn.GATConv(
#             in_channels=dim_node,
#             out_channels=dim_hidden[0],
#             heads=num_heads,
#             edge_dim=edge_dim
#         )  # the first layer

#         # append the rest of GAT layers
#         layers = []
#         for i in range(num_layer - 2):
#             layers.append(
#                 gnn.GATConv(
#                     in_channels=dim_hidden[i] * num_heads,
#                     out_channels=dim_hidden[i + 1],
#                     heads=num_heads,
#                     edge_dim=edge_dim
#                 )
#             )

        
#         # final GAT layer will have heads==1
#         layers.append(
#             gnn.GATConv(
#                 in_channels=dim_hidden[-2] * num_heads,
#                 out_channels=dim_hidden[-1],
#                 heads=1,
#                 edge_dim=edge_dim,
#                 concat=False
#             )
#         )
           
#         # [Added by JY @ 2022-07-18 to incorporate Kaiming He initialization]
#         # Refer to : https://towardsdatascience.com/understand-kaiming-initialization-and-implementation-detail-in-pytorch-f7aa967e9138

#         # In "https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/conv/gat_conv.html#GATConv"
#         # It appears GATConv is already set to be initalized by glorot (xavier?) instead.
#         # Look into this.

#         '''
#         layer = gnn.GATConv(
#                 in_channels=dim_hidden[-2] * num_heads,
#                 out_channels=dim_hidden[-1],
#                 heads=1,
#                 edge_dim=edge_dim,
#                 concat=False,
                
#         )
#         init.kaiming_normal_(layer.weight, mode = 'fan_in')
#         '''
            
        
#         self.convs = nn.ModuleList(layers)
        
#         self.act_fn1 = activation_fn( **activation_fn_kwargs )
#         self.act_fns = nn.ModuleList(
#             [ activation_fn( **activation_fn_kwargs ) for _ in range(num_layer - 1) ]
#         ) 

#         self.readout = pool()

#         self.ffn = nn.Sequential(*(
#             [nn.Dropout(p=dropout_level), nn.Linear(dim_hidden[-1], num_classes), nn.Softmax(dim=1)]
#         ))

#         # can also use an additional Linear layer before predictions as follows:
#         # self.ffn = nn.Sequential(*(
#         #         [nn.Linear(dim_hidden[-1], dim_hidden[-1])] +
#         #         [nn.ReLU(), nn.Dropout(p=dropout_level), nn.Linear(dim_hidden[-1], num_classes)]
#         # ))
        
        
#         # self.percentage_of_zero_arr = np.array([])
        
#         return

    
#     def forward(self, *args, **kwargs) -> torch.Tensor:
#         """
#         Does a single forward pass for the complete model
#         """
#         x, edge_index, edge_attr, batch, ptr, y, name = self.arguments_read(*args, **kwargs)

#         # JY @ 2023-06-04 : To get the subgraph-level 1-gram feat-vec (could be both node-feat-vec and edge-feat-vec) do it here 



#         # JY @ 2023-06-16 Make Node-Type Mask based on first node-input-vector (as node-type)
#         #                 based on "/data/d1/jgwak1/STREAMLINED_DATA_GENERATION_MultiGraph_JY/STEP_3_processdata_traintestsplit/data_preprocessing/dp_v2_ONLY_TASKNAME_EDGE_ATTR/data_processor_v2_MultiEdge_5BitNodeAttr.py"
#                 # if "file" in node_name.lower():
#                 #     node_attr = [1,0,0,0,0]
#                 # elif "reg" in node_name.lower():
#                 #     node_attr = [0,1,0,0,0]
#                 # elif "net" in node_name.lower():
#                 #     node_attr = [0,0,1,0,0]
#                 # elif "proc" in node_name.lower():
#                 #     node_attr = [0,0,0,1,0]
#                 # elif "thread" in node_name.lower():
#                 #     node_attr = [0,0,0,0,1]

#         # Thread_Node_Index_list = [4] # last index 
#         # threadnode_mask_tensor_2 = x[:, Thread_Node_Index] == 1   

#         # Thread_Node_Indices = [4] # last index 
#         # import datetime
#         # torch_any_check_start = datetime.datetime.now()
#         # threadnode_mask_tensor = torch.any(x[:, Thread_Node_Indices] == 1, dim=1)
#         # torch_any_check_done = datetime.datetime.now()
#         # print(str(torch_any_check_done - torch_any_check_start),flush=True)

#         FILE_NODE_INDEX = 0
#         REG_NODE_INDEX = 1
#         NET_NODE_INDEX = 2
#         PROC_NODE_INDEX = 3
#         THREAD_NODE_INDEX = 4

#         # import datetime
#         Node_Indices = [ FILE_NODE_INDEX, REG_NODE_INDEX ]
#         # torch_any_check_start = datetime.datetime.now()
#         node_mask_tensor = torch.any(x[:, Node_Indices] == 1, dim=1)
#         # torch_any_check_done = datetime.datetime.now()
#         # print(str(torch_any_check_done - torch_any_check_start),flush=True)



#         # direct_check_start = datetime.datetime.now()
#         # node_mask_tensor = x[:, FILE_NODE_INDEX] == 1

#         # direct_check_done = datetime.datetime.now()
#         # print(str(direct_check_done - direct_check_start),flush=True)

#         # equality_check = torch.eq(threadnode_mask_tensor, threadnode_mask_tensor_check)

#         # print(f"equality_check shape: {equality_check.shape}\n equality_check sum: {sum(equality_check)}\n", flush= True) 
#         # print(f"threadnode_mask_tensor shape: {threadnode_mask_tensor.shape}\n threadnode_mask_tensor sum: {sum(threadnode_mask_tensor)}\n", flush= True) 
#         # print(f"threadnode_mask_tensor_check shape: {threadnode_mask_tensor_check.shape}\n threadnode_mask_tensor_check sum: {sum(threadnode_mask_tensor_check)}\n", flush= True) 


#         # 1. first conv. pass -> uses the original data sample   
#          # [ "return_attention_weight = True" added by JY @ 2022-08-22 ] 
#          # Refer to: https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.GATConv
#         out = self.conv1(x=x, edge_index=edge_index, edge_attr=edge_attr) # return_attention_weights = True
#         #print("out (weight-sum of input nuerons): {}\n".format(out))
#         post_conv = self.act_fn1( out )
#         #print("activation-function applied to out: {}\n".format(post_conv))
#         for conv, act_fn in zip( self.convs, self.act_fns ):
#             # 2. iteratively do GCN convolution
            
#             # [ "return_attention_weight = True" added by JY @ 2022-08-22 ] 
#             # Refer to: https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.GATConv
#             out = conv(x=post_conv, edge_index=edge_index, edge_attr=edge_attr) #  return_attention_weights = True
#             #print("out (weight-sum of input nuerons): {}\n".format(out))
#             post_conv = act_fn( out )
#             #print("activation-function applied to out: {}\n".format(post_conv))


#             # get percentage of ReLU activation-function output's that are 0. (dying relu)
#                     #     relu ( layer ( x ))


#         # 3. use the readout (i.e., pooling)
#         # print(post_conv)


#         if str(self.readout) == "LocalMeanPool()":
#             out_readout = self.readout(post_conv, batch, node_mask_tensor)
#         else:
#             out_readout = self.readout(post_conv, batch)

#         # 4. the class probabilities

#         # JY @ 2023-06-04 : To also give the subgraph-level 1-gram feat-vec to ffn do it here

#         out = self.ffn(out_readout)
#         return out

#     def get_emb(self, *args, **kwargs) -> torch.Tensor:
#         """
#         Auxilary function if node embeddings are required seperately
#         works similar to the forward pass above, but without readout
#         """
#         x, edge_index, edge_attr, batch, ptr, y, name = self.arguments_read(*args, **kwargs)
#         post_conv = self.act_fn1(self.conv1(x=x, edge_index=edge_index, edge_attr=edge_attr))
#         for conv, act_fn in zip(self.convs, self.act_fns):
#             post_conv = act_fn(conv(x=post_conv, edge_index=edge_index, edge_attr=edge_attr))
#         return post_conv




#******************************************************************************************************************************************************
#******************************************************************************************************************************************************
#******************************************************************************************************************************************************
###################################################################################################################################
###################################################################################################################################
###################################################################################################################################
###################################################################################################################################

from torch import nn
import gc

from source.trainer import TrainModel
from source.dataprocessor_graphs import LoadGraphs

##### For profiling 
from torch.profiler import profile, record_function, ProfilerActivity
import datetime

from sklearn import metrics

def main():

    torch.set_default_tensor_type(torch.DoubleTensor)

    ###############################################################################################################################################
    # Set data paths
    projection_datapath_Benign_Train_dict = {
      "Dataset_1_B148_M148": \
         "/home/jgwak1/tabby/BASELINE_COMPARISONS/Sequential/RF+Ngrams/RESULTS__RF_1gram_flatten_subgraph_psh__at_20230718_11-53-31__SupposedlyCleanerDataset__5BitPlusADHOC/GNN_TrainSet_same_as_RFexp/Processed_Benign_ONLY_TaskName_edgeattr",
      # ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
      "Dataset_2_B239_M239": \
         "/home/jgwak1/tabby/OFFLINE_TRAINTEST_DATA/dataset_2/GNN_TrainSet_same_as_RFexp/Processed_Benign_ONLY_TaskName_edgeattr"
    
    }
    projection_datapath_Malware_Train_dict = {
      "Dataset_1_B148_M148": \
         "/home/jgwak1/tabby/BASELINE_COMPARISONS/Sequential/RF+Ngrams/RESULTS__RF_1gram_flatten_subgraph_psh__at_20230718_11-53-31__SupposedlyCleanerDataset__5BitPlusADHOC/GNN_TrainSet_same_as_RFexp/Processed_Malware_ONLY_TaskName_edgeattr",
      # ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
      "Dataset_2_B239_M239": \
         "/home/jgwak1/tabby/OFFLINE_TRAINTEST_DATA/dataset_2/GNN_TrainSet_same_as_RFexp/Processed_Malware_ONLY_TaskName_edgeattr"
    }
    projection_datapath_Benign_Test_dict = {
      # Dataset-1 (B#148, M#148) ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
      "Dataset_1_B148_M148": \
         "/home/jgwak1/tabby/BASELINE_COMPARISONS/Sequential/RF+Ngrams/RESULTS__RF_1gram_flatten_subgraph_psh__at_20230718_11-53-31__SupposedlyCleanerDataset__5BitPlusADHOC/GNN_TestSet_same_as_RFexp/Processed_Benign_ONLY_TaskName_edgeattr",
      # ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
      "Dataset_2_B239_M239": \
         "/home/jgwak1/tabby/OFFLINE_TRAINTEST_DATA/dataset_2/GNN_TestSet_same_as_RFexp/Processed_Benign_ONLY_TaskName_edgeattr"
    }
    projection_datapath_Malware_Test_dict = {
      # Dataset-1 (B#148, M#148) ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
      "Dataset_1_B148_M148": \
         "/home/jgwak1/tabby/BASELINE_COMPARISONS/Sequential/RF+Ngrams/RESULTS__RF_1gram_flatten_subgraph_psh__at_20230718_11-53-31__SupposedlyCleanerDataset__5BitPlusADHOC/GNN_TestSet_same_as_RFexp/Processed_Malware_ONLY_TaskName_edgeattr",
      # ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
      "Dataset_2_B239_M239": \
         "/home/jgwak1/tabby/OFFLINE_TRAINTEST_DATA/dataset_2/GNN_TestSet_same_as_RFexp/Processed_Malware_ONLY_TaskName_edgeattr"
    }
    ###############################################################################################################################################

    _num_classes = 2  # number of class labels and always binary classification.


    _dim_node = 46   # num node features ; the #feats
    _dim_edge = 72    # (or edge_dim) ; num edge features

    dataset_choice = "Dataset_1_B148_M148"
    signal_amplification_option = "signal_amplified__event_1gram"

    
    if signal_amplification_option == "signal_amplified__event_1gram":
        dim_node__expanded_for_compatibility_with_signal_amplification = _dim_edge - 1
    # elif signal_amplification_option == "signal_amplified__event_1gram_nodetype_5bit": # NOT COMPATIBLE B/C HANDLING FOR 5BIT IS HARDLY IMPLEMENTALBE WITH PYTORCH GPU ACCELERATION RELATED CODE
    #     dim_node__expanded_for_compatibility_with_signal_amplification = 5 + (_dim_edge - 1)     
    # elif signal_amplification_option == "signal_amplified__event_1gram_nodetype_5bit_and_Ahoc_Identifier": # NOT COMPATIBLE B/C HANDLING FOR 5BIT IS HARDLY IMPLEMENTALBE WITH PYTORCH GPU ACCELERATION RELATED CODE + DOESNOT WORK WELL
    #     dim_node__expanded_for_compatibility_with_signal_amplification = _dim_node + (_dim_edge - 1)     
    else:
        ValueError("Invalid signal-amplification-option")


    _benign_train_data_path = projection_datapath_Benign_Train_dict[dataset_choice]
    _malware_train_data_path = projection_datapath_Malware_Train_dict[dataset_choice]
    _benign_final_test_data_path = projection_datapath_Benign_Test_dict[dataset_choice]
    _malware_final_test_data_path = projection_datapath_Malware_Test_dict[dataset_choice]

    print(f"dataset_choice: {dataset_choice}", flush=True)
    print("data-paths:", flush=True)
    print(f"_benign_train_data_path: {_benign_train_data_path}", flush=True)
    print(f"_malware_train_data_path: {_malware_train_data_path}", flush=True)
    print(f"_benign_final_test_data_path: {_benign_final_test_data_path}", flush=True)
    print(f"_malware_final_test_data_path: {_malware_final_test_data_path}", flush=True)
    print(f"\n_dim_node: {_dim_node}", flush=True)
    print(f"_dim_edge: {_dim_edge}", flush=True)
    print("", flush=True)
    print(f"signal_amplification_option: {signal_amplification_option}", flush=True)    
    print(f"dim_node__expanded_for_compatibility_with_signal_amplification: {dim_node__expanded_for_compatibility_with_signal_amplification}", flush=True)
    print("", flush=True)

   # Load both benign and malware graphs """


    dataprocessor = LoadGraphs()
    # expand-dim-node to carry place-holder 
    benign_train_dataset = dataprocessor.parse_all_data__expand_dim_node_compatible_with_SignalAmplification(_benign_train_data_path, _dim_node, _dim_edge,
                                                                                                              signal_amplification_option= signal_amplification_option,
                                                                                                             )
    malware_train_dataset = dataprocessor.parse_all_data__expand_dim_node_compatible_with_SignalAmplification(_malware_train_data_path, _dim_node, _dim_edge,
                                                                                                             signal_amplification_option= signal_amplification_option)
    train_dataset = benign_train_dataset + malware_train_dataset
    print('+ train data loaded #Malware = {} | #Benign = {}'.format(len(benign_train_dataset), len(malware_train_dataset)), flush=True)

    # Load test benign and malware graphs """
    benign_final_test_dataset = dataprocessor.parse_all_data__expand_dim_node_compatible_with_SignalAmplification(_benign_final_test_data_path, _dim_node, _dim_edge,
                                                                                                                  signal_amplification_option= signal_amplification_option)
                                                                                                                  
    malware_final_test_dataset = dataprocessor.parse_all_data__expand_dim_node_compatible_with_SignalAmplification(_malware_final_test_data_path, _dim_node, _dim_edge,
                                                                                                                   signal_amplification_option= signal_amplification_option)

    final_test_dataset = benign_final_test_dataset + malware_final_test_dataset
    print('+ test data loaded #Malware = {} | #Benign = {}'.format(len(malware_final_test_dataset), len(benign_final_test_dataset)), flush=True)


    ######################################################################################################################################################

    # "Memory-Inefficient BUT Torch-Geometric-Conv-Layer Friendly Data Modificiation"
    # *   Goal: Make our Directed Subgraph into Undirected Subgraph.
    # * Reason: 
    #
    # * References
    #           https://github.com/pyg-team/pytorch_geometric/discussions/3043
    #           https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/conv/gat_conv.html#GATConv
    #           https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/transforms/to_undirected.html


    # from torch_geometric.utils import to_undirected


    print(f"\nStart process of converting directed-graph into undirected-graph", flush=True)
    print(f"Reason:\nFor GNN to update a target-node with both its incoming and outgoing edges and associated nodes", flush=True)
    print(f"Implementation (considering compatibility with torch-geometric library GNN message-passing implemntation; NOT very memory-efficient):\
         \n(1) add reverse-edges for all existing directed-edges ('doubling edges with reverse-edges')\
         \n(2) for all added reverse-edges, add edge-features that are identical their original-edge counterparts ('doubling edge-features with duplicate edge-features')", flush=True)
   
    # before_process_dataset = copy.deepcopy(dataset)

    for data in train_dataset:

      print(f"start processing(direct-G --> Undirect-G) '{data.name}'", flush=True)

      # 1. 'doubling edges with reverse-edges'
      #     https://pytorch-geometric.readthedocs.io/en/latest/get_started/introduction.html
      original_edge_src_node_index_tensor = data.edge_index[0]
      original_edge_dst_node_index_tensor = data.edge_index[1]
      reverse_edge_src_node_index_tensor = data.edge_index[1]  # reverse-edge's source-node is original-edge's destination-node, and vice versa.
      reverse_edge_dst_node_index_tensor = data.edge_index[0]  # reverse-edge's destination-node is original-edge's source-node, and vice versa.
      doubled_edge_src_node_index_tensor = torch.cat([original_edge_src_node_index_tensor, reverse_edge_src_node_index_tensor], dim = 0)
      doubled_edge_dst_node_index_tensor = torch.cat([original_edge_dst_node_index_tensor, reverse_edge_dst_node_index_tensor], dim = 0)

      # check
      # e.g. original edge-index (for C->A, A->B, B->D )
      #      > src : [ C, A, B ] 
      #      > dst : [ A, B, D ]
      # e.g. doubled edge-index (for C->A, A->B, B->D and reverse-edges A->C, B->A, D->B )
      #      > src : [ C, A, B ] + [ A, B, D ]
      #      > dst : [ A, B, D ] + [ C, A, B ]

      original_node_index_dim = original_edge_src_node_index_tensor.shape[0]
      assert torch.all(doubled_edge_src_node_index_tensor[: original_node_index_dim] == doubled_edge_dst_node_index_tensor[original_node_index_dim:]), "mismatch"
      assert torch.all(doubled_edge_dst_node_index_tensor[: original_node_index_dim] == doubled_edge_src_node_index_tensor[original_node_index_dim:]), "mismatch"

      # torch.stack([doubled_edge_src_node_index_tensor, doubled_edge_dst_node_index_tensor]).shape
      # data.edge_index.shape
      doubled_edge_index = torch.stack([doubled_edge_src_node_index_tensor, doubled_edge_dst_node_index_tensor])
      # update
      #  data.edge_index.shape      
      #  doubled_edge_index.shape
      data.edge_index = doubled_edge_index
      
      # -------------------------------------------------------------------------------------------------------------------------
      # 2.  IF NOT "Edge-Feat-Migrated":
      #        'doubling edge-features with duplicate edge-features' 
      #     ELSE:
      #        'ignore this step' 
      #     
      #     ** This scheme makes more sense with “Non-Edge-Feat-Migrated” and might have some logical-problem with "Edge-Feat-Migrated"
      #        As in “Edge-Feat-Migrated” each Node-Attr already is a vector-sum of all incoming-edges-events (task-freqs). 
      #        So it’s more natural to do with “Non-Edge-Feat-Migrated” than “Edge-Feat-Migrated” unless we change the definition of migrated-edge-feat to node-feat as vector-sum of not only incoming edge events, but also outgoing edge-evens

      if data.edge_attr is not None:

         doubled_edge_attr = torch.cat([data.edge_attr, data.edge_attr], dim = 0)

         # check
         # e.g. original edge-index (for C->A, A->B, B->D )
         #      > src : [ C, A, B ] 
         #      > dst : [ A, B, D ]
         #      edge-attr:
         #              [ [ 2, 1, 1 ],   # C -> A 
         #                [ 1, 0, 0 ],   # A -> B
         #                [ 3, 0, 0 ] ]  # B -> D
         #
         #
         # e.g. doubled edge-index (for C->A, A->B, B->D and reverse-edges A->C, B->A, D->B )
         #      > src : [ C, A, B ] + [ A, B, D ]
         #      > dst : [ A, B, D ] + [ C, A, B ]
         #      dobuled edge-attr:
         #              [ [ 2, 1, 1 ],   # C -> A 
         #                [ 1, 0, 0 ],   # A -> B
         #                [ 3, 0, 0 ],   # B -> D
         #                [ 2, 1, 1 ],   # A -> C (reverse-edge)        
         #                [ 1, 0, 0 ],   # B -> A (reverse-edge)
         #                [ 3, 0, 0 ],   # D -> B (reverse-edge)
         #
         #      Thus, "doubled edge-attr" is simply concatenating original edge-attr with itself ("doubled-edge-attr == edge-attr + edge-attr")

         # update
         #  data.edge_attr.shape
         #  doubled_edge_attr.shape
         data.edge_attr = doubled_edge_attr
      print(f"Done processing(direct-G --> Undirect-G) '{data.name}'", flush=True)

    print(f"Done processing(doubling for direct-G --> Undirect-G) all samples", flush=True)








   ######################################################################################################################################################



    # Use Sensible Defaults (https://fullstackdeeplearning.com/spring2021/lecture-7/)

    lr = 0.001
    dropout_level = 0.0
    weight_decay = 5e-8
    batch_size = 32
    device = 'cuda:1'
    num_epochs = 1000

    # N_target_nodes_for_group

    # classifier = GNN_Signal_Amplification__ver1(
    #                                         expanded_dim_node = dim_node__expanded_for_compatibility_with_signal_amplification, 
    #                                         signal_amplification_option = signal_amplification_option,
                                            
    #                                         edge_dim = _dim_edge,
    #                                         num_classes = _num_classes, 

    #                                         embedding_dim = [38], 
    #                                         ffn_dims=[19, 8],                                            
    #                                         activation_fn = nn.ReLU,
    #                                         # conv_activation_fn = nn.ReLU,                                            
    #                                         neighborhood_aggr= "add", # 'add' or 'mean'
    #                                         pool = GlobalMaxPool,

    #                                         dropout_level = dropout_level, 
    #                                      )



    classifier = GNN_Signal_Amplification__ver2(
                                                expanded_dim_node = dim_node__expanded_for_compatibility_with_signal_amplification,
                                                signal_amplification_option = signal_amplification_option,

                                                edge_dim = _dim_edge,
                                                num_classes = _num_classes, 

                                                ffn_dims=[16, 8],                                            
                                                activation_fn = nn.ReLU,
                                                # conv_activation_fn = nn.ReLU,                                            
                                                neighborhood_aggr= "weighted_sum", # 'add' or 'mean'
                                                pool = GlobalMaxPool,

                                                dropout_level = dropout_level, 
                                                )


    dataloader_params = {
                           'batch_size' : batch_size,
                           'data_split_ratio': [1.0, 0, 0],  # train : 100% / eval: 0% / test: 0%  (test should be 0%, since we are just 'fitting' here.)
                           'split_shuffle_seed': 0
                          }
               
    # dataset batch_size , data_split_ratio, seed
    model_trainer = TrainModel(
                                 model = classifier, 

                                 dataset= train_dataset, # no need to pass corresponding y_train 
                                                   # b/c data in X_train already contains y (label) themselves 
                                 
                                 best_criteria = 'train_loss', 

                                #  device = 'cuda:0',

                                 device = device,
                                #  device= 'cpu',
                                 dataloader_params = dataloader_params
                                )

    train_params = { 
                       'num_epochs': num_epochs, 
                       'num_early_stop': num_epochs,
                       'milestones': None,
                       'gamma': None
                     }
      
    optimizer_params = { 
                           'lr': lr, 
                           'weight_decay': weight_decay
                         }
     

    print(f"_dim_node: {_dim_node}")
    print(f"_dim_edge: {_dim_edge}")
    print(classifier, flush= True)
    print(dataloader_params, flush= True)
    print(train_params, flush= True)
    # print(f"dim_hidden: {dim_hidden}")
    print(f"dropout_level: {dropout_level}")   
    print(optimizer_params, flush=True)


    # model_trainer.train returns the best-model based on best_criteria
    best_model = model_trainer.train( 
                                       train_params = train_params, 
                                       optimizer_params = optimizer_params
                                      )


    #######################################################################################################################
    # Test to unseen data.


    all_trues = []
    all_preds = []

    column_names = ['sample', 'truth', 'pred', 'benign_softmax', 'malware_softmax' ]
    test_results_df = pd.DataFrame(columns=column_names)

    best_model.to("cpu")
    for test_data in final_test_dataset:

      softmax_outputs = best_model(test_data)   
      pred = softmax_outputs.argmax(-1)


        #print(f"{f} | softmax-coeffs: {softmax_outputs} | benign-prob: {class_probs['benign_prob']} | malware-prob: {class_probs['malware_prob']} | pred: {pred}")         
        # print("{:<20s} {:<20s} {:<20s} {:<20s}".format(f, str(class_probs['benign_prob']), str(class_probs['malware_prob']), str(pred.tolist()[0])))

      sample_pred_info = {'sample': test_data.name.lstrip("Processed_SUBGRAPH_P3_").rstrip(".pickle"), 
                            'truth': test_data.y.item(), 
                            'pred': pred.item(), 
                            'benign_softmax': softmax_outputs[0][0].item(), 
                            'malware_softmax': softmax_outputs[0][1].item()}      

      all_trues.append(test_data.y.item())
      all_preds.append(pred.item())

      test_results_df = pd.concat([test_results_df, pd.DataFrame([sample_pred_info])], axis=0, ignore_index=True)
      



    test_results_df = test_results_df.sort_values("truth")
    test_accuracy = metrics.accuracy_score(y_true = all_trues, y_pred = all_preds)
    test_f1 = metrics.f1_score(y_true = all_trues, y_pred = all_preds)
    test_precision = metrics.precision_score(y_true = all_trues, y_pred = all_preds)
    test_recall = metrics.recall_score(y_true = all_trues, y_pred = all_preds)

    print("\nTest Result:\n", flush=True)
    print(f"test accuracy: {test_accuracy}", flush = True)
    print(f"test f1: {test_f1}", flush = True)
    print(f"test precision: {test_precision}", flush = True)
    print(f"test recall: {test_recall}", flush = True)   

    print(test_results_df, flush=True)
            

if __name__ == "__main__":
    # mp.set_start_method('spawn')  # Set the start method as early as possible

    main()

