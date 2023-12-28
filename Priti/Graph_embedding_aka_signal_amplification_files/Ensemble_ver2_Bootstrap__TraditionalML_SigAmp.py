import os
import sys
import torch
import gc
# import pytorch_lightning as pl
import numpy as np
import re
from collections import Counter

'''
Replaced 'code' to 'source' as there's conflict with vscode debugger.
from code.dataprocessor_graphs import LoadGraphs
from code.model import GIN
from code.trainer import TrainModel
'''
from source.dataprocessor_graphs import LoadGraphs
from source.model import GIN
from source.trainer_meng_ver import TrainModel

from itertools import product
import pandas as pd
from datetime import datetime

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold


''' Applying 'Nested Stratified K-fold Cross-Validation' with GAT, GIN to Project-2-Dataset '''
#SKLEARN: https://scikit-learn.org/stable/modules/classes.html#module-sklearn.model_selectio
#         https://scikit-learn.org/0.21/modules/model_evaluation.html#classification-metrics
# splitter-classes
from pickletools import optimize
from grpc import stream_unary_rpc_method_handler
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
# hyperparameter-optimizers
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
# model-validation
from sklearn.model_selection import cross_val_score # evaluate score by cross-validation.
# performance-metrics
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score 
from sklearn.metrics import make_scorer
# custom-estimator-class
from sklearn.base import BaseEstimator, ClassifierMixin

#GNN-TRAINING
from source.dataprocessor_graphs import LoadGraphs
from source.model import GAT, GIN, GIN_no_edgefeat, GIN_no_edgefeat_simplified, GCN, GAT_He, GAT_mlp_fed_1gram 
from source.model import GNNBasic, LocalMeanPool,  SumPool, GlobalMeanPool

sys.path.append("/data/d1/jgwak1/stratkfold/source")
from source.gnn_dynamicgraph import GNN_DynamicGraph__ImplScheme_1

# from source.trainer import TrainModel, get_dataloader

#ETC
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
torch.set_printoptions(profile="full")
import numpy as np 
import random
from datetime import datetime
import os
import json
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)
import re
import gc
from distutils.util import strtobool
from argparse import ArgumentParser

import pprint
from collections import defaultdict
import time
#**********************************************************************************************************************************************************************

taskname_colnames = [
   # Based on "TN2int_Revert()" of "/data/d1/jgwak1/tabby/BASELINE_COMPARISONS/Sequential/RF+Ngrams/RFSVM_ngram_flattened_subgraph_only_psh.py"

   "None", #0

   'CLEANUP', #1
   'CLOSE', #2
   'CREATE', #3
   'CREATENEWFILE', #4
   'DELETEPATH',#5
   'DIRENUM',#6
   'DIRNOTIFY',#7
   'FLUSH',#8
   'FSCTL',#9
   'NAMECREATE',#10
   'NAMEDELETE',#11
   'OPERATIONEND',#12
   'QUERYINFO',#13
   'QUERYINFORMATION',#14
   'QUERYEA',#15
   'QUERYSECURITY',#16
   'READ',#17
   'WRITE',#18
   'SETDELETE',#19
   'SETINFORMATION', #20
   'PAGEPRIORITYCHANGE',#21
   'IOPRIORITYCHANGE',#22
   'CPUBASEPRIORITYCHANGE',#23
   'IMAGEPRIORITYCHANGE',#24
   'CPUPRIORITYCHANGE',#25
   'IMAGELOAD',#26
   'IMAGEUNLOAD',#27
   'PROCESSSTOP',#28
   'PROCESSSTART',#29
   'PROCESSFREEZE',#30
   'PSDISKIOATTRIBUTE',#31
   'PSIORATECONTROL',#32 
   'THREADSTART',#33
   'THREADSTOP',#34
   'THREADWORKONBEHALFUPDATE', #35
   'JOBSTART',#36
   'JOBTERMINATE',#37
   'LOSTEVENT',#38
   'PSDISKIOATTRIBUTION',#39
   'RENAME',#40
   'RENAMEPATH',#41
   'THISGROUPOFEVENTSTRACKSTHEPERFORMANCEOFFLUSHINGHIVES',#42(index in final bit vector)

   "UDPIP42_DatasentoverUDPprotocol",  #43
   "UDPIP43_DatareceivedoverUDPprotocol", #44
   "UDPIP49_UDPconnectionattemptfailed", #45

   "TCPIP10_TCPIPDatasent",   # 46
   "TCPIP11_TCPIPDatareceived", # 47
   "TCPIP12_TCPIPConnectionattempted", # 48
   "TCPIP13_TCPIPDisconnectissued", # 49
   "TCPIP14_TCPIPDataretransmitted", # 50
   "TCPIP15_TCPIPConnectionaccepted", # 51
   "TCPIP16_TCPIPReconnectattempted", # 52
   "TCPIP17_TCPIPTCPconnectionattemptfailed", # 53
   "TCPIP18_TCPIPProtocolcopieddataonbehalfofuser", # 54

   "REGISTRY32_CreateKey", # 55
   "REGISTRY33_OpenKey", # 56
   "REGISTRY34_DeleteKey", # 57
   "REGISTRY35_QueryKey", # 58
   "REGISTRY36_SetValueKey", # 59
   "REGISTRY37_DeleteValueKey", # 60
   "REGISTRY38_QueryValueKey", # 61
   "REGISTRY39_EnumerateKey", # 62
   "REGISTRY40_EnumerateValueKey", # 63
   "REGISTRY41_QueryMultipleValueKey", # 64
   "REGISTRY42_SetInformationKey", # 65
   "REGISTRY43_FlushKey", # 66
   "REGISTRY44_CloseKey", # 67
   "REGISTRY45_QuerySecurityKey", # 68
   "REGISTRY46_SetSecurityKey", # 69

   "Else" # 70

]


# Signal-Amplification Function (Thread-level Event-Dist. 1gram + Adjacent Node's Node-Type 5Bit)
def get_signal_amplified_thread_level_eventdist_adjacent_5bit_dist_dict( dataset : list ):
      
      # A little different from "get_thread_level_eventdist_adjacentFRNpatterndist_dict"
      # Differencce is that "get_thread_level_eventdist_adjacentFRNpatterndist_dict" only considers adjacent-nodes Adhoc-pattern-node-attr
      # While this considers the 5Bit+Adhoc-pattern-node-attr

      file_node_tensor = torch.tensor([1, 0, 0, 0, 0])
      reg_node_tensor = torch.tensor([0, 1, 0, 0, 0])
      net_node_tensor = torch.tensor([0, 0, 1, 0, 0])
      proc_node_tensor = torch.tensor([0, 0, 0, 1, 0])
      thread_node_tensor = torch.tensor([0, 0, 0, 0, 1])

      data_dict = dict()

      cnt = 1
      for data in dataset:
            
            print(f"signal-amplifying: {data.name} ||  {cnt}/{len(dataset)}", flush=True)

            # Added by JY @ 2023-07-18 to handle node-attr dim > 5  
            if data.x.shape[1] != 5:
               data_x_first5 = data.x[:,:5]

            file_node_indices = torch.nonzero(torch.all(torch.eq( data_x_first5, file_node_tensor), dim=1), as_tuple=False).flatten().tolist()
            reg_node_indices = torch.nonzero(torch.all(torch.eq( data_x_first5, reg_node_tensor), dim=1), as_tuple=False).flatten().tolist()
            net_node_indices = torch.nonzero(torch.all(torch.eq( data_x_first5, net_node_tensor), dim=1), as_tuple=False).flatten().tolist()
            proc_node_indices = torch.nonzero(torch.all(torch.eq( data_x_first5, proc_node_tensor), dim=1), as_tuple=False).flatten().tolist()
            thread_node_indices = torch.nonzero(torch.all(torch.eq( data_x_first5, thread_node_tensor), dim=1), as_tuple=False).flatten().tolist()

            # which this node is a source-node (outgoing-edge w.r.t this node)
            
            data_thread_node_incoming_edges_edge_attrs = torch.tensor([])
            data_thread_node_outgoing_edges_edge_attrs = torch.tensor([])
            data_thread_node_both_direction_edges_edge_attrs = torch.tensor([])
            data_thread_node_all_unique_adjacent_nodes_5bit_dists = torch.tensor([]) # Added by JY @ 2023-07-19

            for thread_node_idx in thread_node_indices:

               edge_src_indices = data.edge_index[0]
               edge_tar_indices = data.edge_index[1]

               # which this node is a target-node (outgoing-edge w.r.t this node)
               outgoing_edges_from_thread_node_idx = torch.nonzero( edge_src_indices == thread_node_idx ).flatten()
               # which this node is a target-node (incoming-edge w.r.t this node)
               incoming_edges_to_thread_node_idx = torch.nonzero( edge_tar_indices == thread_node_idx ).flatten()

               # Following is to deal with edge-attr (event-dist & time-scalar) -------------------------------------------------------------------------------------
               edge_attr_of_incoming_edges_from_thread_node_idx = data.edge_attr[incoming_edges_to_thread_node_idx]
               edge_attr_of_outgoing_edges_from_thread_node_idx = data.edge_attr[outgoing_edges_from_thread_node_idx]

               edge_attr_of_incoming_edges_from_thread_node_idx_sum = torch.sum(edge_attr_of_incoming_edges_from_thread_node_idx[:,:-1], dim = 0)
               edge_attr_of_outgoing_edges_from_thread_node_idx_sum = torch.sum(edge_attr_of_outgoing_edges_from_thread_node_idx[:,:-1], dim = 0)
               edge_attr_of_both_direction_edges_from_thread_node_idx_sum = torch.add(edge_attr_of_incoming_edges_from_thread_node_idx_sum, 
                                                                                      edge_attr_of_outgoing_edges_from_thread_node_idx_sum)
               
               data_thread_node_both_direction_edges_edge_attrs = torch.cat(( data_thread_node_both_direction_edges_edge_attrs,
                                                                              edge_attr_of_both_direction_edges_from_thread_node_idx_sum.unsqueeze(0) ), dim = 0)
               # -------------------------------------------------------------------------------------------------------------------------------------------------------

               # JY @ 2023-07-18: Now get all Adhoc identifier-pattern distributions of adjacent F/R/N t

               # But also need to consider the multi-graph aspect here. 
               # So here, do not count twice for duplicate adjacent nodes due to multigraph.
               # Just need to get the distribution.
               # Find unique column-wise pairs (dropping duplicates - src/dst pairs that come from multi-graph)
               unique_outgoing_edges_from_thread_node, _ = torch.unique( data.edge_index[:, outgoing_edges_from_thread_node_idx ], dim=1, return_inverse=True)

               # Find unique column-wise pairs (dropping duplicates - src/dst pairs that come from multi-graph)
               unique_incoming_edges_to_thread_node, _ = torch.unique( data.edge_index[:, incoming_edges_to_thread_node_idx ], dim=1, return_inverse=True)

               target_nodes_of_outgoing_edges_from_thread_node = unique_outgoing_edges_from_thread_node[1] # edge-target is index 1
               source_nodes_of_incoming_edges_to_thread_node = unique_incoming_edges_to_thread_node[0] # edge-src is index 0

               # "5bit 
               data__5bit = data.x[:,:5]

               #-- Option-1 --------------------------------------------------------------------------------------------------------------------------------------------------------------
               # # Already handled multi-graph case, but how about the bi-directional edge case?
               # # For, T-->F and T<--F, information of F will be recorded twice."Dont Let it happen"
               unique_adjacent_nodes_of_both_direction_edges_of_thread_node = torch.unique( torch.cat( [ target_nodes_of_outgoing_edges_from_thread_node, 
                                                                                                         source_nodes_of_incoming_edges_to_thread_node ] ) )
               integrated_5bit_of_all_unique_adjacent_nodes_to_thread = torch.sum( data__5bit[unique_adjacent_nodes_of_both_direction_edges_of_thread_node], dim = 0 )
               
               data_thread_node_all_unique_adjacent_nodes_5bit_dists = torch.cat(( data_thread_node_all_unique_adjacent_nodes_5bit_dists,
                                                                                          integrated_5bit_of_all_unique_adjacent_nodes_to_thread.unsqueeze(0) ), dim = 0)
               # --------------------------------------------------------------------------------------------------------------------------------------------------------------------------

               #-- Option-2 --------------------------------------------------------------------------------------------------------------------------------------------------------------
               # # # Already handled multi-graph case, but how about the bi-directional edge case?
               # # # For, T-->F and T<--F, information of F will be recorded twice. "Let it happen"
               # non_unique_adjacent_nodes_of_both_direction_edges_of_thread_node = torch.cat( [ target_nodes_of_outgoing_edges_from_thread_node, 
               #                                                                                 source_nodes_of_incoming_edges_to_thread_node ] )
               # integrated_5bit_of_all_non_unique_adjacent_nodes_to_thread = torch.sum( data__5bit[non_unique_adjacent_nodes_of_both_direction_edges_of_thread_node], dim = 0 )
               
               # data_thread_node_all_unique_adjacent_nodes_5bit_dists = torch.cat(( data_thread_node_all_unique_adjacent_nodes_5bit_dists,
               #                                                                      integrated_5bit_of_all_non_unique_adjacent_nodes_to_thread.unsqueeze(0) ), dim = 0)



               # --------------------------------------------------------------------------------------------------------------------------------------------------------------------------

            thread_eventdist_adjacent_5bit_dist = torch.cat( [data_thread_node_both_direction_edges_edge_attrs, 
                                                                  data_thread_node_all_unique_adjacent_nodes_5bit_dists], dim = 1)
            data_dict[ re.search(r'Processed_SUBGRAPH_P3_(.*?)\.pickle', data.name).group(1) ] = thread_eventdist_adjacent_5bit_dist.tolist()

            # data_dict[ data.name.lstrip("Processed_SUBGRAPH_P3_").rstrip(".pickle") ] = data_thread_node_both_direction_edges_edge_attrs.tolist()
            cnt+=1
      return data_dict





# Signal-Amplification Function (Thread-level Event-Dist. 1gram + Adjacent Node's adhoc_identifier)
def get_signal_amplified_thread_level_eventdist_adjacent_5bit_and_Adhoc_Identifier_dist_dict( dataset : list ):
      
      # A little different from "get_thread_level_eventdist_adjacentFRNpatterndist_dict"
      # Differencce is that "get_thread_level_eventdist_adjacentFRNpatterndist_dict" only considers adjacent-nodes Adhoc-pattern-node-attr
      # While this considers the 5Bit+Adhoc-pattern-node-attr

      file_node_tensor = torch.tensor([1, 0, 0, 0, 0])
      reg_node_tensor = torch.tensor([0, 1, 0, 0, 0])
      net_node_tensor = torch.tensor([0, 0, 1, 0, 0])
      proc_node_tensor = torch.tensor([0, 0, 0, 1, 0])
      thread_node_tensor = torch.tensor([0, 0, 0, 0, 1])

      data_dict = dict()

      cnt = 1
      for data in dataset:
            
            print(f"signal-amplifying: {data.name} ||  {cnt}/{len(dataset)}", flush=True)

            # Added by JY @ 2023-07-18 to handle node-attr dim > 5  
            if data.x.shape[1] != 5:
               data_x_first5 = data.x[:,:5]

            file_node_indices = torch.nonzero(torch.all(torch.eq( data_x_first5, file_node_tensor), dim=1), as_tuple=False).flatten().tolist()
            reg_node_indices = torch.nonzero(torch.all(torch.eq( data_x_first5, reg_node_tensor), dim=1), as_tuple=False).flatten().tolist()
            net_node_indices = torch.nonzero(torch.all(torch.eq( data_x_first5, net_node_tensor), dim=1), as_tuple=False).flatten().tolist()
            proc_node_indices = torch.nonzero(torch.all(torch.eq( data_x_first5, proc_node_tensor), dim=1), as_tuple=False).flatten().tolist()
            thread_node_indices = torch.nonzero(torch.all(torch.eq( data_x_first5, thread_node_tensor), dim=1), as_tuple=False).flatten().tolist()

            # which this node is a source-node (outgoing-edge w.r.t this node)
            
            data_thread_node_incoming_edges_edge_attrs = torch.tensor([])
            data_thread_node_outgoing_edges_edge_attrs = torch.tensor([])
            data_thread_node_both_direction_edges_edge_attrs = torch.tensor([])
            data_thread_node_all_unique_adjacent_nodes_5bit_and_Adhoc_Identifier_dists = torch.tensor([]) # Added by JY @ 2023-07-19

            for thread_node_idx in thread_node_indices:

               edge_src_indices = data.edge_index[0]
               edge_tar_indices = data.edge_index[1]

               # which this node is a target-node (outgoing-edge w.r.t this node)
               outgoing_edges_from_thread_node_idx = torch.nonzero( edge_src_indices == thread_node_idx ).flatten()
               # which this node is a target-node (incoming-edge w.r.t this node)
               incoming_edges_to_thread_node_idx = torch.nonzero( edge_tar_indices == thread_node_idx ).flatten()

               # Following is to deal with edge-attr (event-dist & time-scalar) -------------------------------------------------------------------------------------
               edge_attr_of_incoming_edges_from_thread_node_idx = data.edge_attr[incoming_edges_to_thread_node_idx]
               edge_attr_of_outgoing_edges_from_thread_node_idx = data.edge_attr[outgoing_edges_from_thread_node_idx]

               edge_attr_of_incoming_edges_from_thread_node_idx_sum = torch.sum(edge_attr_of_incoming_edges_from_thread_node_idx[:,:-1], dim = 0)
               edge_attr_of_outgoing_edges_from_thread_node_idx_sum = torch.sum(edge_attr_of_outgoing_edges_from_thread_node_idx[:,:-1], dim = 0)
               edge_attr_of_both_direction_edges_from_thread_node_idx_sum = torch.add(edge_attr_of_incoming_edges_from_thread_node_idx_sum, 
                                                                                      edge_attr_of_outgoing_edges_from_thread_node_idx_sum)
               
               data_thread_node_both_direction_edges_edge_attrs = torch.cat(( data_thread_node_both_direction_edges_edge_attrs,
                                                                              edge_attr_of_both_direction_edges_from_thread_node_idx_sum.unsqueeze(0) ), dim = 0)
               # -------------------------------------------------------------------------------------------------------------------------------------------------------

               # JY @ 2023-07-18: Now get all Adhoc identifier-pattern distributions of adjacent F/R/N t

               # But also need to consider the multi-graph aspect here. 
               # So here, do not count twice for duplicate adjacent nodes due to multigraph.
               # Just need to get the distribution.
               # Find unique column-wise pairs (dropping duplicates - src/dst pairs that come from multi-graph)
               unique_outgoing_edges_from_thread_node, _ = torch.unique( data.edge_index[:, outgoing_edges_from_thread_node_idx ], dim=1, return_inverse=True)

               # Find unique column-wise pairs (dropping duplicates - src/dst pairs that come from multi-graph)
               unique_incoming_edges_to_thread_node, _ = torch.unique( data.edge_index[:, incoming_edges_to_thread_node_idx ], dim=1, return_inverse=True)

               target_nodes_of_outgoing_edges_from_thread_node = unique_outgoing_edges_from_thread_node[1] # edge-target is index 1
               source_nodes_of_incoming_edges_to_thread_node = unique_incoming_edges_to_thread_node[0] # edge-src is index 0


               #-- Option-1 --------------------------------------------------------------------------------------------------------------------------------------------------------------
               # # Already handled multi-graph case, but how about the bi-directional edge case?
               # # For, T-->F and T<--F, information of F will be recorded twice."Dont Let it happen"
               unique_adjacent_nodes_of_both_direction_edges_of_thread_node = torch.unique( torch.cat( [ target_nodes_of_outgoing_edges_from_thread_node, 
                                                                                                         source_nodes_of_incoming_edges_to_thread_node ] ) )
               integrated_adjacent_5bit_and_Adhoc_Identifier_of_all_unique_adjacent_nodes_to_thread = torch.sum( data.x[unique_adjacent_nodes_of_both_direction_edges_of_thread_node], dim = 0 )

               data_thread_node_all_unique_adjacent_nodes_5bit_and_Adhoc_Identifier_dists = torch.cat(( data_thread_node_all_unique_adjacent_nodes_5bit_and_Adhoc_Identifier_dists,
                                                                                                         integrated_adjacent_5bit_and_Adhoc_Identifier_of_all_unique_adjacent_nodes_to_thread.unsqueeze(0) ), dim = 0)
               # --------------------------------------------------------------------------------------------------------------------------------------------------------------------------

               #-- Option-2 --------------------------------------------------------------------------------------------------------------------------------------------------------------
               # # # Already handled multi-graph case, but how about the bi-directional edge case?
               # # # For, T-->F and T<--F, information of F will be recorded twice. "Let it happen"
               # non_unique_adjacent_nodes_of_both_direction_edges_of_thread_node = torch.cat( [ target_nodes_of_outgoing_edges_from_thread_node, 
               #                                                                                 source_nodes_of_incoming_edges_to_thread_node ] )
               # integrated_5bit_of_all_non_unique_adjacent_nodes_to_thread = torch.sum( data__5bit[non_unique_adjacent_nodes_of_both_direction_edges_of_thread_node], dim = 0 )
               
               # data_thread_node_all_unique_adjacent_nodes_5bit_dists = torch.cat(( data_thread_node_all_unique_adjacent_nodes_5bit_dists,
               #                                                                      integrated_5bit_of_all_non_unique_adjacent_nodes_to_thread.unsqueeze(0) ), dim = 0)



               # --------------------------------------------------------------------------------------------------------------------------------------------------------------------------

            thread_eventdist_adjacent_5bit_and_Adhoc_Identifier_dist = torch.cat( [data_thread_node_both_direction_edges_edge_attrs, 
                                                                                   data_thread_node_all_unique_adjacent_nodes_5bit_and_Adhoc_Identifier_dists], dim = 1)
            data_dict[ re.search(r'Processed_SUBGRAPH_P3_(.*?)\.pickle', data.name).group(1) ] = thread_eventdist_adjacent_5bit_and_Adhoc_Identifier_dist.tolist()

            # data_dict[ data.name.lstrip("Processed_SUBGRAPH_P3_").rstrip(".pickle") ] = data_thread_node_both_direction_edges_edge_attrs.tolist()
            cnt+=1
      return data_dict








#**********************************************************************************************************************************************************************
# Signal-Amplification Function (Thread-level Event-Dist. 1gram)
def get_signal_amplified_thread_level_eventdist_dict( dataset : list ):
      
      file_node_tensor = torch.tensor([1, 0, 0, 0, 0])
      reg_node_tensor = torch.tensor([0, 1, 0, 0, 0])
      net_node_tensor = torch.tensor([0, 0, 1, 0, 0])
      proc_node_tensor = torch.tensor([0, 0, 0, 1, 0])
      thread_node_tensor = torch.tensor([0, 0, 0, 0, 1])

      data_dict = dict()

      for data in dataset:
            # Added by JY @ 2023-07-18 to handle node-attr dim > 5  
            if data.x.shape[1] != 5:
               data.x = data.x[:,:5]

            file_node_indices = torch.nonzero(torch.all(torch.eq( data.x, file_node_tensor), dim=1), as_tuple=False).flatten().tolist()
            reg_node_indices = torch.nonzero(torch.all(torch.eq( data.x, reg_node_tensor), dim=1), as_tuple=False).flatten().tolist()
            net_node_indices = torch.nonzero(torch.all(torch.eq( data.x, net_node_tensor), dim=1), as_tuple=False).flatten().tolist()
            proc_node_indices = torch.nonzero(torch.all(torch.eq( data.x, proc_node_tensor), dim=1), as_tuple=False).flatten().tolist()
            thread_node_indices = torch.nonzero(torch.all(torch.eq( data.x, thread_node_tensor), dim=1), as_tuple=False).flatten().tolist()

            # which this node is a source-node (outgoing-edge w.r.t this node)
            
            data_thread_node_incoming_edges_edge_attrs = torch.tensor([])
            data_thread_node_outgoing_edges_edge_attrs = torch.tensor([])
            data_thread_node_both_direction_edges_edge_attrs = torch.tensor([])

            for thread_node_idx in thread_node_indices:

               edge_src_indices = data.edge_index[0]
               edge_tar_indices = data.edge_index[1]

               # which this node is a target-node (outgoing-edge w.r.t this node)
               outgoing_edges_from_thread_node_idx = torch.nonzero( edge_src_indices == thread_node_idx ).flatten()
               # which this node is a target-node (incoming-edge w.r.t this node)
               incoming_edges_to_thread_node_idx = torch.nonzero( edge_tar_indices == thread_node_idx ).flatten()

               edge_attr_of_incoming_edges_from_thread_node_idx = data.edge_attr[incoming_edges_to_thread_node_idx]
               edge_attr_of_outgoing_edges_from_thread_node_idx = data.edge_attr[outgoing_edges_from_thread_node_idx]


               edge_attr_of_incoming_edges_from_thread_node_idx_sum = torch.sum(edge_attr_of_incoming_edges_from_thread_node_idx[:,:-1], dim = 0)
               edge_attr_of_outgoing_edges_from_thread_node_idx_sum = torch.sum(edge_attr_of_outgoing_edges_from_thread_node_idx[:,:-1], dim = 0)
               edge_attr_of_both_direction_edges_from_thread_node_idx_sum = torch.add(edge_attr_of_incoming_edges_from_thread_node_idx_sum, 
                                                                                      edge_attr_of_outgoing_edges_from_thread_node_idx_sum)

               data_thread_node_both_direction_edges_edge_attrs = torch.cat(( data_thread_node_both_direction_edges_edge_attrs,
                                                                              edge_attr_of_both_direction_edges_from_thread_node_idx_sum.unsqueeze(0) ), dim = 0)

            data_dict[ re.search(r'Processed_SUBGRAPH_P3_(.*)\.pickle', data.name).group(1) ] = data_thread_node_both_direction_edges_edge_attrs.tolist()

      return data_dict

def get_readout_applied_df( data_dict : dict,
                            readout_option : str, 
                            signal_amplification_option: str):    
   if signal_amplification_option == "signal_amplified__event_1gram_nodetype_5bit":
      nodetype_names = ["file", "registry", "network", "process", "thread"] # this is the correct-order
      feature_names = taskname_colnames + nodetype_names
   elif signal_amplification_option == "signal_amplified__event_1gram":
      feature_names = taskname_colnames

   elif signal_amplification_option == "signal_amplified__event_1gram_nodetype_5bit_and_Ahoc_Identifier":
      nodetype_names = ["file", "registry", "network", "process", "thread"] # this is the correct-order
      feature_names = taskname_colnames + nodetype_names + \
                      [f"adhoc_pattern_{i}" for i in range(len(data_dict[list(data_dict.keys())[0]][0]) -\
                                                               len(taskname_colnames) - len(nodetype_names))]


   else:
      ValueError(f"Invalid signal_amplification_option ({signal_amplification_option})")  

   entire_data_df = pd.DataFrame( columns = ["thread_is_from"] + feature_names )
   for data_name in data_dict:
      data_name_df = pd.DataFrame( data_dict[data_name], columns = feature_names )
      data_name_df['thread_is_from'] = data_name
      entire_data_df = pd.concat([entire_data_df, data_name_df], axis=0, ignore_index=True)
   
   if readout_option == "max":
       sample_grouby_df = entire_data_df.groupby("thread_is_from").max()
   elif readout_option == "mean":
       sample_grouby_df = entire_data_df.groupby("thread_is_from").mean()



   return sample_grouby_df

#**********************************************************************************************************************************************************************

def get_No_Graph_Structure_eventdist_nodetype5bit_dist_dict( dataset : list ):
      file_node_tensor = torch.tensor([1, 0, 0, 0, 0])
      reg_node_tensor = torch.tensor([0, 1, 0, 0, 0])
      net_node_tensor = torch.tensor([0, 0, 1, 0, 0])
      proc_node_tensor = torch.tensor([0, 0, 0, 1, 0])
      thread_node_tensor = torch.tensor([0, 0, 0, 0, 1])

      data_dict = dict()

      for data in dataset:

         data.x = data.x[:,:5]
         data.edge_attr = data.edge_attr[:,:-1] # drop time-scalar

         eventdist = torch.sum(data.edge_attr, dim = 0)
         nodetype_5bit_dist = torch.sum(data.x, axis = 0)

         event_nodetype5bit_dist = torch.cat(( eventdist, nodetype_5bit_dist ), dim = 0)
         data_dict[ re.search(r'Processed_SUBGRAPH_P3_(.*)\.pickle', data.name).group(1) ] = event_nodetype5bit_dist.tolist()

      return data_dict


def get_No_Graph_Structure_eventdist_nodetype5bit_adhoc_identifier_dist_dict( dataset : list ):
      file_node_tensor = torch.tensor([1, 0, 0, 0, 0])
      reg_node_tensor = torch.tensor([0, 1, 0, 0, 0])
      net_node_tensor = torch.tensor([0, 0, 1, 0, 0])
      proc_node_tensor = torch.tensor([0, 0, 0, 1, 0])
      thread_node_tensor = torch.tensor([0, 0, 0, 0, 1])

      data_dict = dict()

      for data in dataset:

         data.edge_attr = data.edge_attr[:,:-1] # drop time-scalar

         eventdist = torch.sum(data.edge_attr, dim = 0)
         nodetype5bit_adhoc_identifier_dist = torch.sum(data.x, axis = 0)

         event_nodetype5bit_adhoc_identifier_dist = torch.cat(( eventdist, nodetype5bit_adhoc_identifier_dist ), dim = 0)
         data_dict[ re.search(r'Processed_SUBGRAPH_P3_(.*)\.pickle', data.name).group(1) ] = event_nodetype5bit_adhoc_identifier_dist.tolist()

      return data_dict


def get_No_Graph_Structure_eventdist_dict( dataset : list ):

      file_node_tensor = torch.tensor([1, 0, 0, 0, 0])
      reg_node_tensor = torch.tensor([0, 1, 0, 0, 0])
      net_node_tensor = torch.tensor([0, 0, 1, 0, 0])
      proc_node_tensor = torch.tensor([0, 0, 0, 1, 0])
      thread_node_tensor = torch.tensor([0, 0, 0, 0, 1])

      data_dict = dict()

      for data in dataset:

         data.edge_attr = data.edge_attr[:,:-1] # drop time-scalar

         eventdist = torch.sum(data.edge_attr, dim = 0)

         data_dict[ re.search(r'Processed_SUBGRAPH_P3_(.*)\.pickle', data.name).group(1) ] = eventdist.tolist()

      return data_dict

#**********************************************************************************************************************************************************************

#**********************************************************************************************************************************************************************
#**********************************************************************************************************************************************************************
#**********************************************************************************************************************************************************************

#############################################################################################################################
# libraries for tradition-ML (2023-07-25)

import sklearn
from sklearn import metrics
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier # XGB

# for baseline
from sklearn.feature_extraction.text import CountVectorizer


##############################################################################################################################


# JY @ 2023-07-25: Here, not only tune using K-fold-CV, but also fit the best hyperparam on entire-train and test on final-test

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('-k', '--K', nargs = 1, type = int, default = [10])  

    parser.add_argument('-n', '--Top_N', nargs = 1, type = int, default = [10])      

    parser.add_argument('-data', '--dataset', 
                        choices= ['Dataset_1_B148_M148',
                                  'Dataset_2_B239_M239' # try
                                  ], 
                        default = ["Dataset_1_B148_M148"])


    model_cls_map = {"RandomForest": RandomForestClassifier, "XGBoost": GradientBoostingClassifier,
                     "LogisticRegression": LogisticRegression, "SVM": svm } 
    
    parser.add_argument('-mod_cls', '--trad_model_cls', nargs = 1, type = str, 
                        default = ["XGBoost"] )


    parser.add_argument('-sig_amp_opt', '--signal_amplification_option', 
                        choices= ['signal_amplified__event_1gram',
                                  'no_graph_structure__event_1gram',

                                  'signal_amplified__event_1gram_nodetype_5bit', 
                                  'no_graph_structure__event_1gram_nodetype_5bit', 

                                  'signal_amplified__event_1gram_nodetype_5bit_and_Ahoc_Identifier',
                                  'no_graph_structure__event_1gram_nodetype_5bit_and_Ahoc_Identifier',

                                  ], 
                                  default = ["signal_amplified__event_1gram_nodetype_5bit"])

    parser.add_argument('-readout_opt', '--readout_option', 
                        choices= ['max',
                                  'mean' # try
                                  ], 
                                  default = ["max"])

    parser.add_argument('-ss_opt', '--search_space_option', 
                        choices= [
                                  # Best-found @ 2023-07-26 based on tuning
                                  "XGBoost_Treatment_Best_20230726",
                                  "XGBoost_Control_Best_20230726",

                                  "RandomForest_Treatment_Best_20230726",
                                  "RandomForest_Control_Best_20230726",

                                  "LogisticRegression_Treatment_Best_20230726",
                                  "LogisticRegression_Contorl_Best_20230726",

                                  'XGBoost_Treatment_Best_Multiple',
                                  'XGBoost_Control_Best_Multiple',



                                  'XGBoost_Treatment_Best_20230726_TopN_from_df',
                                  'XGBoost_Control_Best_20230726_TopN_from_df',


                                  'XGBoost_Treatment_Best_Dataset_2_20230806_TopN_from_df',
                                  'XGBoost_Control_Best_Dataset_2_20230806_TopN_from_df',


                                  'XGBoost_Treatment_Best_Dataset_1_20230808_TopN_from_df',
                                  'XGBoost_Control_Best_Dataset_1_20230808_TopN_from_df',

                                  ], 
                                  default = ["XGBoost_Control_Best_Dataset_1_20230808_TopN_from_df"])

    parser.add_argument("--test_on_all_folds__or__final_test",  
                         choices= ["test_on_all_folds","final_test"],  
                         default = ["final_test"] )

    # cmd args
    K = parser.parse_args().K[0]
    Top_N = parser.parse_args().Top_N[0]
    model_cls = model_cls_map[ parser.parse_args().trad_model_cls[0] ]
    dataset_choice = parser.parse_args().dataset[0]

    signal_amplification_option = parser.parse_args().signal_amplification_option[0]
    readout_option = parser.parse_args().readout_option[0]
    search_space_option = parser.parse_args().search_space_option[0]
    test_on_all_folds__or__final_test = parser.parse_args().test_on_all_folds__or__final_test[0] 

    saved_models_dirpath = "/data/d1/jgwak1/stratkfold/saved_Traditional_ML_models"   
    model_cls_name = re.search(r"'(.*?)'", str(model_cls)).group(1)

    if test_on_all_folds__or__final_test in {"test_on_all_folds", "final_test"}:
       experiment_results_df_fpath = f"/data/d1/jgwak1/stratkfold/{model_cls_name}__{dataset_choice}__{search_space_option}__{K}_FoldCV__{test_on_all_folds__or__final_test}__{signal_amplification_option}__{readout_option}__{datetime.now().strftime('%Y-%m-%d_%H%M%S')}.csv"

    if test_on_all_folds__or__final_test == "final_test":
       final_test_results_df_fpath = f"/data/d1/jgwak1/stratkfold/{model_cls_name}__{dataset_choice}__{search_space_option}__{test_on_all_folds__or__final_test}__{signal_amplification_option}__{readout_option}__{datetime.now().strftime('%Y-%m-%d_%H%M%S')}.csv"


    trace_filename = f'traces_stratkfold_double_strat_{model_cls_name}__generated@'+str(datetime.now())+".txt" 

    ###############################################################################################################################################
    # Set data paths
    projection_datapath_Benign_Train_dict = {
      # Dataset-1 (B#148, M#148) ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
      "Dataset_1_B148_M148": \
         "/data/d1/jgwak1/tabby/BASELINE_COMPARISONS/Sequential/RF+Ngrams/RESULTS__RF_1gram_flatten_subgraph_psh__at_20230718_11-53-31__SupposedlyCleanerDataset__5BitPlusADHOC/GNN_TrainSet_same_as_RFexp/Processed_Benign_ONLY_TaskName_edgeattr",

      # ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
      "Dataset_2_B239_M239": \
         "/data/d1/jgwak1/tabby/OFFLINE_TRAINTEST_DATA/dataset_2/GNN_TrainSet_same_as_RFexp/Processed_Benign_ONLY_TaskName_edgeattr"
    
    }
    projection_datapath_Malware_Train_dict = {
      # Dataset-1 (B#148, M#148) ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
      "Dataset_1_B148_M148": \
         "/data/d1/jgwak1/tabby/BASELINE_COMPARISONS/Sequential/RF+Ngrams/RESULTS__RF_1gram_flatten_subgraph_psh__at_20230718_11-53-31__SupposedlyCleanerDataset__5BitPlusADHOC/GNN_TrainSet_same_as_RFexp/Processed_Malware_ONLY_TaskName_edgeattr",
      # ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
      "Dataset_2_B239_M239": \
         "/data/d1/jgwak1/tabby/OFFLINE_TRAINTEST_DATA/dataset_2/GNN_TrainSet_same_as_RFexp/Processed_Malware_ONLY_TaskName_edgeattr"
    }
    projection_datapath_Benign_Test_dict = {
      # Dataset-1 (B#148, M#148) ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
      "Dataset_1_B148_M148": \
         "/data/d1/jgwak1/tabby/BASELINE_COMPARISONS/Sequential/RF+Ngrams/RESULTS__RF_1gram_flatten_subgraph_psh__at_20230718_11-53-31__SupposedlyCleanerDataset__5BitPlusADHOC/GNN_TestSet_same_as_RFexp/Processed_Benign_ONLY_TaskName_edgeattr",
      # ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
      "Dataset_2_B239_M239": \
         "/data/d1/jgwak1/tabby/OFFLINE_TRAINTEST_DATA/dataset_2/GNN_TestSet_same_as_RFexp/Processed_Benign_ONLY_TaskName_edgeattr"

    }
    projection_datapath_Malware_Test_dict = {
      # Dataset-1 (B#148, M#148) ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
      "Dataset_1_B148_M148": \
         "/data/d1/jgwak1/tabby/BASELINE_COMPARISONS/Sequential/RF+Ngrams/RESULTS__RF_1gram_flatten_subgraph_psh__at_20230718_11-53-31__SupposedlyCleanerDataset__5BitPlusADHOC/GNN_TestSet_same_as_RFexp/Processed_Malware_ONLY_TaskName_edgeattr",
      # ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
      "Dataset_2_B239_M239": \
         "/data/d1/jgwak1/tabby/OFFLINE_TRAINTEST_DATA/dataset_2/GNN_TestSet_same_as_RFexp/Processed_Malware_ONLY_TaskName_edgeattr"


    }
    ###############################################################################################################################################

    _num_classes = 2  # number of class labels and always binary classification.


    _dim_node = 46   # num node features ; the #feats
    _dim_edge = 72    # (or edge_dim) ; num edge features


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


    ####################################################################################################################################################
    ####################################################################################################################################################
    ####################################################################################################################################################

    def XGBoost_Treatment_Best_20230726():   

         # https://docs.google.com/spreadsheets/d/1bsL35ChSkR8gvDGO7U5FN3wG1pNsIVfBu0XPuMRKCEw/edit#gid=571535633

         Top_1 =  {
                  'n_estimators': 100, 
                  'learning_rate': 0.05, 
                  'max_depth': 9, 
                  'min_samples_split': 2, 
                  'min_samples_leaf': 1, 
                  'subsample': 1.0, 
                  'max_features': "sqrt",
                  'loss': 'log_loss',
                  'random_state': 0,
                  'split_shuffle_seed': 100 # not needed
                  }
         Top_2 =  {
                  'n_estimators': 300, 
                  'learning_rate': 0.01, 
                  'max_depth': 9, 
                  'min_samples_split': 2, 
                  'min_samples_leaf': 1, 
                  'subsample': 0.75, 
                  'max_features': None,
                  'loss': 'log_loss',
                  'random_state': 0,
                  'split_shuffle_seed': 100 # not needed
                  }
         Top_3 =  {
                  'n_estimators': 200, 
                  'learning_rate': 0.01, 
                  'max_depth': 9, 
                  'min_samples_split': 2, 
                  'min_samples_leaf': 1, 
                  'subsample': 0.75, 
                  'max_features': None,
                  'loss': 'log_loss',
                  'random_state': 0,
                  'split_shuffle_seed': 100 # not needed
                  }
         Top_4 =  {
                  'n_estimators': 300, 
                  'learning_rate': 0.01, 
                  'max_depth': 6, 
                  'min_samples_split': 5, 
                  'min_samples_leaf': 1, 
                  'subsample': 0.75, 
                  'max_features': None,
                  'loss': 'log_loss',
                  'random_state': 0,
                  'split_shuffle_seed': 100 # not needed
                  }
         Top_5 =  {
                  'n_estimators': 100, 
                  'learning_rate': 0.05, 
                  'max_depth': 9, 
                  'min_samples_split': 5, 
                  'min_samples_leaf': 1, 
                  'subsample': 1.0, 
                  'max_features': "sqrt",
                  'loss': 'log_loss',
                  'random_state': 0,
                  'split_shuffle_seed': 100 # not needed
                  }                           

         Top_6 =  {
                  'n_estimators': 100, 
                  'learning_rate': 0.1, 
                  'max_depth': 6, 
                  'min_samples_split': 2, 
                  'min_samples_leaf': 3, 
                  'subsample': 1, 
                  'max_features': None,
                  'loss': 'log_loss',
                  'random_state': 99,
                  'split_shuffle_seed': 100 # not needed
                  }                           
         
         Top_7 =  {
                  'n_estimators': 100, 
                  'learning_rate': 0.1, 
                  'max_depth': 6, 
                  'min_samples_split': 5, 
                  'min_samples_leaf': 3, 
                  'subsample': 1, 
                  'max_features': None,
                  'loss': 'log_loss',
                  'random_state': 99,
                  'split_shuffle_seed': 100# not needed
                  }      
         Top_8 =  {
                  'n_estimators': 100, 
                  'learning_rate': 0.01, 
                  'max_depth': 9, 
                  'min_samples_split': 5, 
                  'min_samples_leaf': 1, 
                  'subsample': 0.75, 
                  'max_features': None,
                  'loss': 'log_loss',
                  'random_state': 0,
                  'split_shuffle_seed': 100 # not needed
                  }      
         Top_9 =  {
                  'n_estimators': 200, 
                  'learning_rate': 0.01, 
                  'max_depth': 9, 
                  'min_samples_split': 5, 
                  'min_samples_leaf': 1, 
                  'subsample': 0.75, 
                  'max_features': None,
                  'loss': 'log_loss',
                  'random_state': 0,
                  'split_shuffle_seed': 100  # not needed
                  }      
         
         Top_10 =  {
                  'n_estimators': 200, 
                  'learning_rate': 0.01, 
                  'max_depth': 6, 
                  'min_samples_split': 5, 
                  'min_samples_leaf': 1, 
                  'subsample': 0.75, 
                  'max_features': None,
                  'loss': 'log_loss',
                  'random_state': 99,
                  'split_shuffle_seed': 100  # not needed
                  }      

         Top_11 =  {
                  'n_estimators': 300, 
                  'learning_rate': 0.01, 
                  'max_depth': 9, 
                  'min_samples_split': 2, 
                  'min_samples_leaf': 1, 
                  'subsample': 0.75, 
                  'max_features': None,
                  'loss': 'log_loss',
                  'random_state': 99,
                  'split_shuffle_seed': 100 # not needed
                  }      

         manual_space = [ Top_1, Top_2, Top_3, Top_4, Top_5, Top_6, Top_7, Top_8, Top_9, Top_10, Top_11  ]
         return manual_space





    def XGBoost_Control_Best_20230726():   
         Top_1 =  {
                  'n_estimators': 200, 
                  'learning_rate': 0.01, 
                  'max_depth': 6, 
                  'min_samples_split': 2, 
                  'min_samples_leaf': 1, 
                  'subsample': 1.0, 
                  'max_features': None,
                  'loss': 'log_loss',
                  'random_state': 42,
                  'split_shuffle_seed': 100 # not needed
                  }
         Top_2 =  {
                  'n_estimators': 100, 
                  'learning_rate': 0.05, 
                  'max_depth': 3, 
                  'min_samples_split': 2, 
                  'min_samples_leaf': 1, 
                  'subsample': 0.75, 
                  'max_features': 'sqrt',
                  'loss': 'log_loss',
                  'random_state': 0,
                  'split_shuffle_seed': 100 # not needed
                  }
         Top_3 =  {
                  'n_estimators': 200, 
                  'learning_rate': 0.01, 
                  'max_depth': 9, 
                  'min_samples_split': 2, 
                  'min_samples_leaf': 3, 
                  'subsample': 0.75, 
                  'max_features': None,
                  'loss': 'log_loss',
                  'random_state': 0,
                  'split_shuffle_seed': 100 # not needed
                  }
         Top_4 =  {
                  'n_estimators': 200, 
                  'learning_rate': 0.01, 
                  'max_depth': 9, 
                  'min_samples_split': 5, 
                  'min_samples_leaf': 3, 
                  'subsample': 0.75, 
                  'max_features': None,
                  'loss': 'log_loss',
                  'random_state': 0,
                  'split_shuffle_seed': 100 # not needed
                  }
         Top_5 =  {
                  'n_estimators': 200, 
                  'learning_rate': 0.01, 
                  'max_depth': 6, 
                  'min_samples_split': 2, 
                  'min_samples_leaf': 3, 
                  'subsample': 0.75, 
                  'max_features': None,
                  'loss': 'log_loss',
                  'random_state': 99,
                  'split_shuffle_seed': 100 # not needed
                  }                           
         
         Top_6 =  {
                  'n_estimators': 200, 
                  'learning_rate': 0.01, 
                  'max_depth': 6, 
                  'min_samples_split': 5, 
                  'min_samples_leaf': 3, 
                  'subsample': 0.75, 
                  'max_features': None,
                  'loss': 'log_loss',
                  'random_state': 99,
                  'split_shuffle_seed': 100 # not needed
                  }

         Top_7 =  {
                  'n_estimators': 200, 
                  'learning_rate': 0.01, 
                  'max_depth': 6, 
                  'min_samples_split': 5, 
                  'min_samples_leaf': 1, 
                  'subsample': 1, 
                  'max_features': None,
                  'loss': 'log_loss',
                  'random_state': 99,
                  'split_shuffle_seed': 100 # not needed
                  }

         Top_8 =  {
                  'n_estimators': 100, 
                  'learning_rate': 0.01, 
                  'max_depth': 6, 
                  'min_samples_split': 2, 
                  'min_samples_leaf': 1, 
                  'subsample': 0.75, 
                  'max_features': None,
                  'loss': 'log_loss',
                  'random_state': 42,
                  'split_shuffle_seed': 100 # not needed
                  }

         Top_9 =  {
                  'n_estimators': 100, 
                  'learning_rate': 0.01, 
                  'max_depth': 9, 
                  'min_samples_split': 10, 
                  'min_samples_leaf': 1, 
                  'subsample': 0.75, 
                  'max_features': None,
                  'loss': 'log_loss',
                  'random_state': 42,
                  'split_shuffle_seed': 100 # not needed
                  }

         Top_10 =  {
                  'n_estimators': 100, 
                  'learning_rate': 0.1, 
                  'max_depth': 3, 
                  'min_samples_split': 10, 
                  'min_samples_leaf': 1, 
                  'subsample': 1, 
                  'max_features': 'sqrt',
                  'loss': 'log_loss',
                  'random_state': 0,
                  'split_shuffle_seed': 100 # not needed
                  }
         Top_11 =  {
                  'n_estimators': 100, 
                  'learning_rate': 0.1, 
                  'max_depth': 3, 
                  'min_samples_split': 2, 
                  'min_samples_leaf': 5, 
                  'subsample': 1, 
                  'max_features': 'sqrt',
                  'loss': 'log_loss',
                  'random_state': 0,
                  'split_shuffle_seed': 100 # not needed
                  }


         manual_space =  [ Top_1, Top_2, Top_3, Top_4, Top_5, Top_6, Top_7, Top_8, Top_9, Top_10, Top_11  ]
         return manual_space



    def XGBoost_Treatment_Best_Multiple():   

         # https://docs.google.com/spreadsheets/d/1bsL35ChSkR8gvDGO7U5FN3wG1pNsIVfBu0XPuMRKCEw/edit#gid=571535633
         Top_1 =  {
                  'n_estimators': 100, 
                  'learning_rate': 0.05, 
                  'max_depth': 9, 
                  'min_samples_split': 2, 
                  'min_samples_leaf': 1, 
                  'subsample': 1.0, 
                  'max_features': "sqrt",
                  'loss': 'log_loss',
                  'random_state': 0,
                  'split_shuffle_seed': 100 # not needed
                  }
         Top_2 =  {
                  'n_estimators': 300, 
                  'learning_rate': 0.01, 
                  'max_depth': 9, 
                  'min_samples_split': 2, 
                  'min_samples_leaf': 1, 
                  'subsample': 0.75, 
                  'max_features': None,
                  'loss': 'log_loss',
                  'random_state': 0,
                  'split_shuffle_seed': 100 # not needed
                  }
         Top_3 =  {
                  'n_estimators': 200, 
                  'learning_rate': 0.01, 
                  'max_depth': 9, 
                  'min_samples_split': 2, 
                  'min_samples_leaf': 1, 
                  'subsample': 0.75, 
                  'max_features': None,
                  'loss': 'log_loss',
                  'random_state': 0,
                  'split_shuffle_seed': 100 # not needed
                  }
         Top_4 =  {
                  'n_estimators': 300, 
                  'learning_rate': 0.01, 
                  'max_depth': 6, 
                  'min_samples_split': 5, 
                  'min_samples_leaf': 1, 
                  'subsample': 0.75, 
                  'max_features': None,
                  'loss': 'log_loss',
                  'random_state': 0,
                  'split_shuffle_seed': 100 # not needed
                  }
         Top_5 =  {
                  'n_estimators': 100, 
                  'learning_rate': 0.05, 
                  'max_depth': 9, 
                  'min_samples_split': 5, 
                  'min_samples_leaf': 1, 
                  'subsample': 1.0, 
                  'max_features': "sqrt",
                  'loss': 'log_loss',
                  'random_state': 0,
                  'split_shuffle_seed': 100 # not needed
                  }     
                  
         manual_space = [ Top_1 ] * 20 +  [ Top_2 ] * 20 +  [ Top_3 ] * 20 + [ Top_4 ] * 20 + [ Top_5 ] * 20
         return manual_space

    def XGBoost_Control_Best_Multiple():   
         Top_1 =  {
                  'n_estimators': 200, 
                  'learning_rate': 0.01, 
                  'max_depth': 6, 
                  'min_samples_split': 2, 
                  'min_samples_leaf': 1, 
                  'subsample': 1.0, 
                  'max_features': None,
                  'loss': 'log_loss',
                  'random_state': 42,
                  'split_shuffle_seed': 100 # not needed
                  }
         Top_2 =  {
                  'n_estimators': 100, 
                  'learning_rate': 0.05, 
                  'max_depth': 3, 
                  'min_samples_split': 2, 
                  'min_samples_leaf': 1, 
                  'subsample': 0.75, 
                  'max_features': 'sqrt',
                  'loss': 'log_loss',
                  'random_state': 0,
                  'split_shuffle_seed': 100 # not needed
                  }
         Top_3 =  {
                  'n_estimators': 200, 
                  'learning_rate': 0.01, 
                  'max_depth': 9, 
                  'min_samples_split': 2, 
                  'min_samples_leaf': 3, 
                  'subsample': 0.75, 
                  'max_features': None,
                  'loss': 'log_loss',
                  'random_state': 0,
                  'split_shuffle_seed': 100 # not needed
                  }
         Top_4 =  {
                  'n_estimators': 200, 
                  'learning_rate': 0.01, 
                  'max_depth': 9, 
                  'min_samples_split': 5, 
                  'min_samples_leaf': 3, 
                  'subsample': 0.75, 
                  'max_features': None,
                  'loss': 'log_loss',
                  'random_state': 0,
                  'split_shuffle_seed': 100 # not needed
                  }
         Top_5 =  {
                  'n_estimators': 200, 
                  'learning_rate': 0.01, 
                  'max_depth': 6, 
                  'min_samples_split': 2, 
                  'min_samples_leaf': 3, 
                  'subsample': 0.75, 
                  'max_features': None,
                  'loss': 'log_loss',
                  'random_state': 99,
                  'split_shuffle_seed': 100 # not needed
                  }                           

         manual_space = [ Top_1 ] * 20 +  [ Top_2 ] * 20 +  [ Top_3 ] * 20 + [ Top_4 ] * 20 + [ Top_5 ] * 20
         return manual_space

    def RandomForest_Treatment_Best_20230726():   
         Top_1 =  {
                  'n_estimators': "", 
                  'criterion': 'gini', 
                  'max_depth': "", 
                  'min_samples_split': "", 
                  'min_samples_leaf': "", 
                  'max_features': "",
                  'bootstrap': True,
                  'random_state': "",
                  'split_shuffle_seed': ""
                  }
         Top_2 =  {
                  'n_estimators': "", 
                  'criterion': 'gini', 
                  'max_depth': "", 
                  'min_samples_split': "", 
                  'min_samples_leaf': "", 
                  'max_features': "",
                  'bootstrap': True,
                  'random_state': "",
                  'split_shuffle_seed': ""
                  }
         Top_3 =  {
                  'n_estimators': "", 
                  'criterion': 'gini', 
                  'max_depth': "", 
                  'min_samples_split': "", 
                  'min_samples_leaf': "", 
                  'max_features': "",
                  'bootstrap': True,
                  'random_state': "",
                  'split_shuffle_seed': ""
                  }
         Top_4 =  {
                  'n_estimators': "", 
                  'criterion': 'gini', 
                  'max_depth': "", 
                  'min_samples_split': "", 
                  'min_samples_leaf': "", 
                  'max_features': "",
                  'bootstrap': True,
                  'random_state': "",
                  'split_shuffle_seed': ""
                  }
         Top_5 =  {
                  'n_estimators': "", 
                  'criterion': 'gini', 
                  'max_depth': "", 
                  'min_samples_split': "", 
                  'min_samples_leaf': "", 
                  'max_features': "",
                  'bootstrap': True,
                  'random_state': "",
                  'split_shuffle_seed': ""
                  }

         manual_space = []
         manual_space.append(

                  )
         return manual_space

    def RandomForest_Control_Best_20230726():   
         manual_space = []
         manual_space.append(
                  {
                  'n_estimators': "", 
                  'criterion': 'gini', 
                  'max_depth': "", 
                  'min_samples_split': "", 
                  'min_samples_leaf': "", 
                  'max_features': "",
                  'bootstrap': True,
                  'random_state': "",
                  'split_shuffle_seed': ""
                  }
                  )
         return manual_space


    def LogisticRegression_Treatment_Best_20230726():   
         manual_space = []
         manual_space.append(
                  {
                  'penalty': "", 
                  'C': "", 
                  'solver': "", 
                  'max_iter': "", 
                  'class_weight': "", 
                  'random_state': "",
                  'split_shuffle_seed': ""
                  }
                  )
         return manual_space

    def LogisticRegression_Control_Best_20230726():   
         manual_space = []
         manual_space.append(
                  {
                  'penalty': "", 
                  'C': "", 
                  'solver': "", 
                  'max_iter': "", 
                  'class_weight': "", 
                  'random_state': "",
                  'split_shuffle_seed': ""
                  }
                  )
         return manual_space

    ####################################################################################################################################################

    if search_space_option == "XGBoost_Treatment_Best_20230726":
       search_space = XGBoost_Treatment_Best_20230726()   
    elif search_space_option == "XGBoost_Control_Best_20230726":
       search_space = XGBoost_Control_Best_20230726()   

    elif search_space_option == "RandomForest_Treatment_Best_20230726":
       search_space = RandomForest_Treatment_Best_20230726()   
    elif search_space_option == "RandomForest_Control_Best_20230726":
       search_space = RandomForest_Control_Best_20230726()   

    elif search_space_option == "LogisticRegression_Treatment_Best_20230726":
       search_space = LogisticRegression_Treatment_Best_20230726()   
    elif search_space_option == "LogisticRegression_Control_Best_20230726":
       search_space = LogisticRegression_Control_Best_20230726()   

    elif search_space_option == "XGBoost_Treatment_Best_Multiple":
       search_space = XGBoost_Treatment_Best_Multiple()   
    elif search_space_option == "XGBoost_Control_Best_Multiple":
       search_space = XGBoost_Control_Best_Multiple()   



    elif search_space_option == "XGBoost_Treatment_Best_20230726_TopN_from_df":
         treatment_search_on_train_df = pd.read_csv("/data/d1/jgwak1/stratkfold/sklearn.ensemble._gb.GradientBoostingClassifier__XGBoost_searchspace_1__10_FoldCV__search_on_train__signal_amplified__event_1gram_nodetype_5bit__max__2023-07-26_220903.csv")
         sorted_treatment_search_on_train_df = treatment_search_on_train_df.sort_values(by='Avg_Val_F1', ascending = False)
         # user_input = input("Please enter N:")
         # # global N
         # N = int(user_input)
         manual_space = []
         for row, datapoint in sorted_treatment_search_on_train_df.head(Top_N).iterrows():
            
            max_features = datapoint['max_features']
            if str(max_features) == "nan":
                max_features = None

            Top_n_dict = { 'n_estimators': datapoint['n_estimators'], 
                           'learning_rate': datapoint['learning_rate'], 
                           'max_depth': datapoint['max_depth'], 
                           'min_samples_split': datapoint['min_samples_split'], 
                           'min_samples_leaf': datapoint['min_samples_leaf'], 
                           'subsample': datapoint['subsample'], 
                           'max_features': max_features,
                           'loss': datapoint['loss'],
                           'random_state': datapoint['random_state'],
                           'split_shuffle_seed': datapoint['split_shuffle_seed'] # not needed
                          }              

            manual_space.append(Top_n_dict)
         
         search_space = manual_space

    elif search_space_option == "XGBoost_Control_Best_20230726_TopN_from_df":
         no_treatment_search_on_train_df = pd.read_csv("/data/d1/jgwak1/stratkfold/sklearn.ensemble._gb.GradientBoostingClassifier__XGBoost_searchspace_1__10_FoldCV__search_on_train__no_graph_structure__event_1gram_nodetype_5bit__max__2023-07-26_220845.csv")
         sorted_no_treatment_search_on_train_df = no_treatment_search_on_train_df.sort_values(by='Avg_Val_F1', ascending = False)

         # user_input = input("Please enter N:")
         # N = int(user_input)
         manual_space = []
         for row, datapoint in sorted_no_treatment_search_on_train_df.head(Top_N).iterrows():
            
            max_features = datapoint['max_features']
            if str(max_features) == "nan":
                max_features = None

            Top_n_dict = { 'n_estimators': datapoint['n_estimators'], 
                           'learning_rate': datapoint['learning_rate'], 
                           'max_depth': datapoint['max_depth'], 
                           'min_samples_split': datapoint['min_samples_split'], 
                           'min_samples_leaf': datapoint['min_samples_leaf'], 
                           'subsample': datapoint['subsample'], 
                           'max_features': max_features,
                           'loss': datapoint['loss'],
                           'random_state': datapoint['random_state'],
                           'split_shuffle_seed': datapoint['split_shuffle_seed'] # not needed
                          }              

            manual_space.append(Top_n_dict)
         
         search_space = manual_space





    elif search_space_option == "XGBoost_Treatment_Best_Dataset_1_20230808_TopN_from_df":
         treatment_search_on_train_df = pd.read_csv("/data/d1/jgwak1/stratkfold/sklearn.ensemble._gb.GradientBoostingClassifier__Dataset_1_B148_M148__XGBoost_searchspace_1__10_FoldCV__search_on_train__signal_amplified__event_1gram_nodetype_5bit__max__2023-08-08_090647.csv")
         sorted_treatment_search_on_train_df = treatment_search_on_train_df.sort_values(by='Avg_Val_F1', ascending = False)
         # user_input = input("Please enter N:")
         # N = int(user_input)

         manual_space = []
         for row, datapoint in sorted_treatment_search_on_train_df.head(Top_N).iterrows():

         # for row, datapoint in sorted_treatment_search_on_train_df.iterrows():

            max_features = datapoint['max_features']
            if str(max_features) == "nan":
                max_features = None

            Top_n_dict = { 'n_estimators': datapoint['n_estimators'], 
                           'learning_rate': datapoint['learning_rate'], 
                           'max_depth': datapoint['max_depth'], 
                           'min_samples_split': datapoint['min_samples_split'], 
                           'min_samples_leaf': datapoint['min_samples_leaf'], 
                           'subsample': datapoint['subsample'], 
                           'max_features': max_features,
                           'loss': datapoint['loss'],
                           'random_state': datapoint['random_state'],
                           'split_shuffle_seed': datapoint['split_shuffle_seed'] # not needed
                          }              

            manual_space.append(Top_n_dict)
         
         search_space = manual_space

    elif search_space_option == "XGBoost_Control_Best_Dataset_1_20230808_TopN_from_df":
         no_treatment_search_on_train_df = pd.read_csv("/data/d1/jgwak1/stratkfold/sklearn.ensemble._gb.GradientBoostingClassifier__Dataset_1_B148_M148__XGBoost_searchspace_1__10_FoldCV__search_on_train__no_graph_structure__event_1gram_nodetype_5bit__max__2023-08-08_090452.csv")
         sorted_no_treatment_search_on_train_df = no_treatment_search_on_train_df.sort_values(by='Avg_Val_F1', ascending = False)

         # user_input = input("Please enter N:")
         # N = int(user_input)
         manual_space = []
         for row, datapoint in sorted_no_treatment_search_on_train_df.head(Top_N).iterrows():
         # for row, datapoint in sorted_no_treatment_search_on_train_df.iterrows():
            
            max_features = datapoint['max_features']
            if str(max_features) == "nan":
                max_features = None

            Top_n_dict = { 'n_estimators': datapoint['n_estimators'], 
                           'learning_rate': datapoint['learning_rate'], 
                           'max_depth': datapoint['max_depth'], 
                           'min_samples_split': datapoint['min_samples_split'], 
                           'min_samples_leaf': datapoint['min_samples_leaf'], 
                           'subsample': datapoint['subsample'], 
                           'max_features': max_features,
                           'loss': datapoint['loss'],
                           'random_state': datapoint['random_state'],
                           'split_shuffle_seed': datapoint['split_shuffle_seed'] # not needed
                          }              

            manual_space.append(Top_n_dict)
         
         search_space = manual_space




    elif search_space_option == "XGBoost_Treatment_Best_Dataset_2_20230806_TopN_from_df":
         treatment_search_on_train_df = pd.read_csv("/data/d1/jgwak1/stratkfold/sklearn.ensemble._gb.GradientBoostingClassifier__Dataset_2_B239_M239__XGBoost_searchspace_1__10_FoldCV__search_on_train__signal_amplified__event_1gram_nodetype_5bit__max__2023-08-06_235547.csv")
         sorted_treatment_search_on_train_df = treatment_search_on_train_df.sort_values(by='Avg_Val_F1', ascending = False)
         # user_input = input("Please enter N:")
         # N = int(user_input)

         manual_space = []
         for row, datapoint in sorted_treatment_search_on_train_df.head(Top_N).iterrows():

         # for row, datapoint in sorted_treatment_search_on_train_df.iterrows():

            max_features = datapoint['max_features']
            if str(max_features) == "nan":
                max_features = None

            Top_n_dict = { 'n_estimators': datapoint['n_estimators'], 
                           'learning_rate': datapoint['learning_rate'], 
                           'max_depth': datapoint['max_depth'], 
                           'min_samples_split': datapoint['min_samples_split'], 
                           'min_samples_leaf': datapoint['min_samples_leaf'], 
                           'subsample': datapoint['subsample'], 
                           'max_features': max_features,
                           'loss': datapoint['loss'],
                           'random_state': datapoint['random_state'],
                           'split_shuffle_seed': datapoint['split_shuffle_seed'] # not needed
                          }              

            manual_space.append(Top_n_dict)
         
         search_space = manual_space

    elif search_space_option == "XGBoost_Control_Best_Dataset_2_20230806_TopN_from_df":
         no_treatment_search_on_train_df = pd.read_csv("/data/d1/jgwak1/stratkfold/sklearn.ensemble._gb.GradientBoostingClassifier__Dataset_2_B239_M239__XGBoost_searchspace_1__10_FoldCV__search_on_train__no_graph_structure__event_1gram_nodetype_5bit__max__2023-08-06_235503.csv")
         sorted_no_treatment_search_on_train_df = no_treatment_search_on_train_df.sort_values(by='Avg_Val_F1', ascending = False)

         # user_input = input("Please enter N:")
         # N = int(user_input)
         manual_space = []
         for row, datapoint in sorted_no_treatment_search_on_train_df.head(Top_N).iterrows():
         # for row, datapoint in sorted_no_treatment_search_on_train_df.iterrows():
            
            max_features = datapoint['max_features']
            if str(max_features) == "nan":
                max_features = None

            Top_n_dict = { 'n_estimators': datapoint['n_estimators'], 
                           'learning_rate': datapoint['learning_rate'], 
                           'max_depth': datapoint['max_depth'], 
                           'min_samples_split': datapoint['min_samples_split'], 
                           'min_samples_leaf': datapoint['min_samples_leaf'], 
                           'subsample': datapoint['subsample'], 
                           'max_features': max_features,
                           'loss': datapoint['loss'],
                           'random_state': datapoint['random_state'],
                           'split_shuffle_seed': datapoint['split_shuffle_seed'] # not needed
                          }              

            manual_space.append(Top_n_dict)
         
         search_space = manual_space



    else:
        ValueError("Unavailable search-space option")





    ####################################################################################################################################################
    ####################################################################################################################################################
    ####################################################################################################################################################
    
   
    # trace file for preds and truth comparison for models during gridsearch
    
    f = open( trace_filename , 'w' )
    f.close()

    # Load both benign and malware graphs """
    dataprocessor = LoadGraphs()
    benign_train_dataset = dataprocessor.parse_all_data(_benign_train_data_path, _dim_node, _dim_edge, drop_dim_edge_timescalar = False)
    malware_train_dataset = dataprocessor.parse_all_data(_malware_train_data_path, _dim_node, _dim_edge, drop_dim_edge_timescalar = False)
    train_dataset = benign_train_dataset + malware_train_dataset
    print('+ train data loaded #Malware = {} | #Benign = {}'.format(len(malware_train_dataset), len(benign_train_dataset)), flush=True)

    # Load test benign and malware graphs """
    benign_test_dataset = dataprocessor.parse_all_data(_benign_final_test_data_path, _dim_node, _dim_edge, drop_dim_edge_timescalar = False)
    malware_test_dataset = dataprocessor.parse_all_data(_malware_final_test_data_path, _dim_node, _dim_edge, drop_dim_edge_timescalar = False)
    final_test_dataset = benign_test_dataset + malware_test_dataset
    print('+ final-test data loaded #Malware = {} | #Benign = {}'.format(len(malware_test_dataset), len(benign_test_dataset)), flush=True)





   # =================================================================================================================================================
   # Prepare for train_dataset (or all-dataset) 

    if test_on_all_folds__or__final_test == "test_on_all_folds":
      train_dataset = train_dataset + final_test_dataset



   # Now apply signal-amplification here (here least conflicts with existing code.)
    if signal_amplification_option == "signal_amplified__event_1gram_nodetype_5bit":
         train_dataset__signal_amplified_dict = get_signal_amplified_thread_level_eventdist_adjacent_5bit_dist_dict( dataset= train_dataset )

    elif signal_amplification_option == "signal_amplified__event_1gram":
         train_dataset__signal_amplified_dict = get_signal_amplified_thread_level_eventdist_dict( dataset= train_dataset )            

    elif signal_amplification_option == "signal_amplified__event_1gram_nodetype_5bit_and_Ahoc_Identifier":
         train_dataset__signal_amplified_dict = get_signal_amplified_thread_level_eventdist_adjacent_5bit_and_Adhoc_Identifier_dist_dict( dataset= train_dataset ) 


    elif signal_amplification_option == "no_graph_structure__event_1gram_nodetype_5bit":
         train_dataset__no_graph_structure_dict = get_No_Graph_Structure_eventdist_nodetype5bit_dist_dict( dataset= train_dataset )  

    elif signal_amplification_option == "no_graph_structure__event_1gram":
         train_dataset__no_graph_structure_dict = get_No_Graph_Structure_eventdist_dict( dataset= train_dataset )  

    elif signal_amplification_option == "no_graph_structure__event_1gram_nodetype_5bit_and_Ahoc_Identifier":
         train_dataset__no_graph_structure_dict = get_No_Graph_Structure_eventdist_nodetype5bit_adhoc_identifier_dist_dict( dataset= train_dataset ) 



    else:
         ValueError(f"Invalid signal_amplification_option ({signal_amplification_option})")                  
    #--------------------------------------------------------------------------
    # Now apply readout to the obtained thread-level vectors
    if "signal_amplified" in signal_amplification_option:
          train_dataset_signal_amplified_readout_df = get_readout_applied_df(data_dict = train_dataset__signal_amplified_dict, 
                                                                             readout_option= readout_option,
                                                                             signal_amplification_option= signal_amplification_option)
      

          X = train_dataset_signal_amplified_readout_df
          X.reset_index(inplace = True)
          X.rename(columns = {'thread_is_from':'data_name'}, inplace = True)

          y = [data.y for data in train_dataset]
    else:
         X = pd.DataFrame(train_dataset__no_graph_structure_dict).T
         if signal_amplification_option in {"no_graph_structure__event_1gram_nodetype_5bit"}:
               nodetype_names = ["file", "registry", "network", "process", "thread"] # this is the correct-order
               feature_names = taskname_colnames + nodetype_names

         if signal_amplification_option in {"no_graph_structure__event_1gram_nodetype_5bit_and_Ahoc_Identifier"}:
               nodetype_names = ["file", "registry", "network", "process", "thread"] # this is the correct-order
               feature_names = taskname_colnames + nodetype_names + [f"adhoc_pattern_{i}" for i in range(len(X.columns) - len(taskname_colnames) - len(nodetype_names))]
         if signal_amplification_option in {"no_graph_structure__event_1gram"} :
               feature_names = taskname_colnames
         X.columns = feature_names
         X.reset_index(inplace = True)
         X.rename(columns = {'index':'data_name'}, inplace = True)

         y = [data.y for data in train_dataset]
    # =================================================================================================================================================

    if test_on_all_folds__or__final_test == "final_test":
        # Also prepare for final-test dataset, to later test the best-fitted models on test-set
         # Now apply signal-amplification here (here least conflicts with existing code.)
         if signal_amplification_option == "signal_amplified__event_1gram_nodetype_5bit":
               final_test_dataset__signal_amplified_dict = get_signal_amplified_thread_level_eventdist_adjacent_5bit_dist_dict( dataset= final_test_dataset )

         elif signal_amplification_option == "signal_amplified__event_1gram":
               final_test_dataset__signal_amplified_dict = get_signal_amplified_thread_level_eventdist_dict( dataset= final_test_dataset )            
         
         elif signal_amplification_option == "no_graph_structure__event_1gram_nodetype_5bit":
               final_test_dataset__no_graph_structure_dict = get_No_Graph_Structure_eventdist_nodetype5bit_dist_dict( dataset= final_test_dataset )  

         elif signal_amplification_option == "no_graph_structure__event_1gram":
               final_test_dataset__no_graph_structure_dict = get_No_Graph_Structure_eventdist_dict( dataset= final_test_dataset )  
         else:
               ValueError(f"Invalid signal_amplification_option ({signal_amplification_option})")                  
         #--------------------------------------------------------------------------
         # Now apply readout to the obtained thread-level vectors
         if "signal_amplified" in signal_amplification_option:
               final_test_dataset_signal_amplified_readout_df = get_readout_applied_df(data_dict = final_test_dataset__signal_amplified_dict, 
                                                                                       readout_option= readout_option,
                                                                                       signal_amplification_option= signal_amplification_option)
               final_test_X = final_test_dataset_signal_amplified_readout_df
               final_test_X.reset_index(inplace = True)
               final_test_X.rename(columns = {'thread_is_from':'data_name'}, inplace = True)

               final_test_y = [data.y for data in final_test_dataset]
         else:
               final_test_X = pd.DataFrame(final_test_dataset__no_graph_structure_dict).T
               if signal_amplification_option in {"signal_amplified__event_1gram_nodetype_5bit", "no_graph_structure__event_1gram_nodetype_5bit"}:
                     nodetype_names = ["file", "registry", "network", "process", "thread"] # this is the correct-order
                     feature_names = taskname_colnames + nodetype_names

               if signal_amplification_option in {"signal_amplified__event_1gram_nodetype_5bit_and_Ahoc_Identifier", 
                                                "no_graph_structure__event_1gram_nodetype_5bit_and_Ahoc_Identifier"}:
                     nodetype_names = ["file", "registry", "network", "process", "thread"] # this is the correct-order
                     feature_names = taskname_colnames + nodetype_names + [f"adhoc_pattern_{i}" for i in range(len(X.columns) - len(taskname_colnames) - len(nodetype_names))]
               if signal_amplification_option in {"signal_amplified__event_1gram", "no_graph_structure__event_1gram"} :
                     feature_names = taskname_colnames
               final_test_X.columns = feature_names
               final_test_X.reset_index(inplace = True)
               final_test_X.rename(columns = {'index':'data_name'}, inplace = True)

               final_test_y = [data.y for data in train_dataset]


         
    #--------------------------------------------------------------------------------------------------------------------------------------------------------
    #--------------------------------------------------------------------------------------------------------------------------------------------------------
    #--------------------------------------------------------------------------------------------------------------------------------------------------------
    #--------------------------------------------------------------------------------------------------------------------------------------------------------
    # if test_on_all_folds__or__final_test in {"test_on_all_folds"}:
            
    #      # Instantiate Experiments Results Dataframe
    #     colnames= list(search_space[0].keys()) + [t+"_"+m for t,m in list(product(["Avg_Val"],["Accuracy","F1","Precision","Recall"]))] +\
    #                                                      ["Std_Val_F1", "Std_Val_Accuracy"] + \
    #                                                      ["K_Val_"+m for m in list(["Accuracy","Precision","Recall","F1"])]
    #     experiment_results_df = pd.DataFrame(columns=colnames)

    #     #  ####################################################################################################################################################

    #     #  """ try out different tunable parameter combinations """
         
    #     #  # test_results = []
    #     #  for hyperparam_set in search_space:

    #         # split_shuffle_seed = hyperparam_set['split_shuffle_seed']

    #     #     # ---------------------------------------------------------------------------------------
    #     #     if model_cls_name == 'sklearn.ensemble._gb.GradientBoostingClassifier'and\
    #     #        'xgboost' in search_space_option.lower():

    #     #        model = model_cls(
    #     #                          n_estimators= hyperparam_set['n_estimators'], 
    #     #                          learning_rate= hyperparam_set['learning_rate'], 
    #     #                          max_depth= hyperparam_set['max_depth'], 
    #     #                          min_samples_split= hyperparam_set['min_samples_split'], 
    #     #                          min_samples_leaf= hyperparam_set['min_samples_leaf'], 
    #     #                          subsample= hyperparam_set['subsample'], 
    #     #                          max_features= hyperparam_set['max_features'],
    #     #                          loss= hyperparam_set['loss'],
    #     #                          random_state= hyperparam_set['random_state']
    #     #                          )


    #     #     elif model_cls_name == 'sklearn.ensemble._forest.RandomForestClassifier'and\
    #     #        'randomforest' in search_space_option.lower():
    #     #        model = model_cls(
    #     #                          n_estimators= hyperparam_set['n_estimators'],
    #     #                          criterion= hyperparam_set['criterion'], 
    #     #                          max_depth= hyperparam_set['max_depth'],
    #     #                          min_samples_split= hyperparam_set['min_samples_split'], 
    #     #                          min_samples_leaf= hyperparam_set['min_samples_leaf'], 
    #     #                          max_features= hyperparam_set['max_features'],
    #     #                          bootstrap= hyperparam_set['bootstrap'],
    #     #                          random_state= hyperparam_set['random_state']
    #     #                          )


    #     #     elif model_cls_name == 'sklearn.linear_model._logistic.LogisticRegression'and\
    #     #        'logistic' in search_space_option.lower():
    #     #        model = model_cls()

    #     #     elif model_cls_name == 'sklearn.svm' and\
    #     #        'svm' in search_space_option:
    #     #        model = model_cls()

    #     #     else:
    #     #        ValueError(f"{model_cls_name} is not supported", flush = True)



    #         # ==================================================================================================================================================
    #         # [Double-Stratification] ===========================================================================================================================
    #         # JY @ 2023-06-27
    #         # Following are datasources for beign and malware 
    #         # exactly same as "/data/d1/jgwak1/tabby/BASELINE_COMPARISONS/Sequential/RF+Ngrams/RFSVM_ngram_flattened_subgraph_only_psh.py"
    #         #   BENIGN_PATTERNS_LIST_FOR_STRATIFIED_SPLIT = [ "fleschutz", "jhochwald", "devblackops", "farag2", "jimbrig", "jrussellfreelance", "nickrod518", 
    #         #                                                 "redttr", "sysadmin-survival-kit", "stevencohn", "ledrago"]
    #         #   MALWARE_PATTERNS_LIST_FOR_STRATIFIED_SPLIT = ["empire", "invoke_obfuscation", "nishang", "poshc2", "mafia",
    #         #                                                 "offsec", "powershellery", "psbits", "pt-toolkit", "randomps", "smallposh",
    #         #                                                 "bazaar"]

    #         # JY @ 2023-06-27
    #         #      Now we want to acheive stratified-K-Fold split based on not only label but also datasource.
    #         #      So in each split, we want the label+datasource combination to be consistent.
    #         #      e.g. Across folds, we want ratio of benign-fleschutz, benign-jhochwald, ... malware-empire, malware-nishang,.. to be consistent
    #         #      For this, first generate a group-list.
    #     X_grouplist = []

   
    #     data_names = X['data_name']
        
    #     split_shuffle_seed = search_space[0]['split_shuffle_seed']


    #     for data_name in data_names:
    #             # benign ------------------------------------------
    #             if "fleschutz" in data_name:
    #                 X_grouplist.append("benign_fleschutz")
    #             if "jhochwald" in data_name:
    #                 X_grouplist.append("benign_jhochwald")
    #             if "devblackops" in data_name:
    #                 X_grouplist.append("benign_devblackops")
    #             if "farag2" in data_name:
    #                 X_grouplist.append("benign_farag2")
    #             if "jimbrig" in data_name:
    #                 X_grouplist.append("benign_jimbrig")
    #             if "jrussellfreelance" in data_name:
    #                 X_grouplist.append("benign_jrussellfreelance")
    #             if "nickrod518" in data_name:
    #                 X_grouplist.append("benign_nickrod518")
    #             if "redttr" in data_name:
    #                 X_grouplist.append("benign_redttr")
    #             if "sysadmin-survival-kit" in data_name:
    #                 X_grouplist.append("benign_sysadmin-survival-kit")
    #             if "stevencohn" in data_name:
    #                 X_grouplist.append("benign_stevencohn")
    #             if "ledrago" in data_name:
    #                 X_grouplist.append("benign_ledrago")
    #             # malware ------------------------------------------
    #             if "empire" in data_name:
    #                 X_grouplist.append("malware_empire")
    #             if "invoke_obfuscation" in data_name:
    #                 X_grouplist.append("malware_invoke_obfuscation")
    #             if "nishang" in data_name:
    #                 X_grouplist.append("malware_nishang")
    #             if "poshc2" in data_name:
    #                 X_grouplist.append("malware_poshc2")
    #             if "mafia" in data_name:
    #                 X_grouplist.append("malware_mafia")
    #             if "offsec" in data_name:
    #                 X_grouplist.append("malware_offsec")
    #             if "powershellery" in data_name:
    #                 X_grouplist.append("malware_powershellery")
    #             if "psbits" in data_name:
    #                 X_grouplist.append("malware_psbits")
    #             if "pt-toolkit" in data_name:
    #                 X_grouplist.append("malware_pt-toolkit")
    #             if "randomps" in data_name:
    #                 X_grouplist.append("malware_randomps")
    #             if "smallposh" in data_name:
    #                 X_grouplist.append("malware_smallposh")
    #             if "bazaar" in data_name:
    #                 X_grouplist.append("malware_bazaar")
    #             if "recollected" in data_name:
    #                 X_grouplist.append("malware_recollected")

    #     # correctness of X_grouplist can be checked by following
    #     # list(zip(X, [data_name for data_name in X.index], y, X_grouplist))
    #     dataset_cv = StratifiedKFold(n_splits=K, 
    #                                     shuffle = True, 
    #                                     random_state=np.random.RandomState(seed=split_shuffle_seed))

    #     ''' JY @ 2023-08-26: Bootstrap here and train '''

    #     # Model-id Prefix

    #     #################################################################################################################
    #     k_validation_results = []
    #     k_results = defaultdict(list)
    #     avg_validation_results = {}


    #     for train_idx, validation_idx in dataset_cv.split(X, y = X_grouplist):  # JY @ 2023-06-27 : y = X_grouplist is crucial for double-stratification

    #         def convert_to_label(value):
    #             if "benign" in value.lower(): return 0
    #             elif "malware" in value.lower(): return 1
    #             else: ValueError("no benign or malware substr")

    #         X_train = X.iloc[train_idx].drop("data_name", axis = 1)
    #         y_train = X.iloc[train_idx]['data_name'].apply(convert_to_label)
    #         X_validation = X.iloc[validation_idx].drop("data_name", axis = 1)
    #         y_validation = X.iloc[validation_idx]['data_name'].apply(convert_to_label)

    #         print(f"Benign: #train:{y_train.value_counts()[0]}, #validation:{y_validation.value_counts()[0]},\nMalware: #train:{y_train.value_counts()[1]}, #validation:{y_validation.value_counts()[1]}",
    #             flush=True)
            
    #         ''' Bootstrap here and fit each bootstrap by that '''


    #         # Added by JY @ 2023-07-31
    #         # 
    #         # > Bootstrap here.
    #         # https://machinelearningmastery.com/a-gentle-introduction-to-the-bootstrap-method/
    #         # scikit-learn bootstrap
    #         from sklearn.utils import resample

    #         N = len(search_space)
    #         N_bootstrapped_datsaets = []
            
    #         for seed in range(N):
    #             boot = resample( X_train, 
    #                              replace=True, 
    #                              n_samples=len(train_dataset), 
    #                              random_state= seed 
    #                             )
    #             N_bootstrapped_datsaets.append( boot )

    #         N_bootstrapped_datsaets

    #         for




    #         # train / record training time
    #         train_start = datetime.now()
    #         model.fit(X = X_train, y = y_train)
    #         training_time =str( datetime.now() - train_start )
    
    #         # test / record test-time
    #         test_start = datetime.now()
    #         preds = model.predict(X = X_validation) # modified return-statement for trainer.train() for this.
            

    #         val_accuracy = sklearn.metrics.accuracy_score(y_true = y_validation, y_pred = preds)
    #         val_f1 = sklearn.metrics.f1_score(y_true = y_validation, y_pred = preds)
    #         val_precision = sklearn.metrics.precision_score(y_true = y_validation, y_pred = preds)
    #         val_recall = sklearn.metrics.recall_score(y_true = y_validation, y_pred = preds)
    #         tn, fp, fn, tp = sklearn.metrics.confusion_matrix(y_validation, preds).ravel()
            
    #         ## JY @ 2023-07-26: Imp almost done : Start from here
    
    #         validation_results = { 
    #             "val_acc": val_accuracy, 
    #             "val_F1": val_f1,
    #             "val_precision": val_precision, 
    #             "val_recall": val_recall, 
    #             "preds":list(preds), 
    #             "truths": list(y_validation),
    #         }
    #         k_validation_results.append(validation_results)
            
    #         for key in validation_results:
    #             k_results[key].append( validation_results[key] )



    #     print("hyparameter-set: ", hyperparam_set, flush=True)
    #     # k_results=temp_results.copy()
    #     print("Avg Validation Result: ", flush=True)
    #     for key in k_results:
    #             if key not in {'preds','truths'}:
    #                 avg_validation_results[key] = np.mean(k_results[key])
    #                 print(f"{key}: {avg_validation_results[key]}", flush=True)

    #     print("k-fold val results:", flush=True)
    #     for item in k_validation_results:
    #             print(item, flush=True)

    #     # write out 
    #     with open(trace_filename, 'a') as f:
    #             f.write("*"*120+"\n")
    #             f.write("param_info:{}\n".format(hyperparam_set))
    #             f.write("avg-validation_results: {}\n".format(avg_validation_results))
    #             f.write("[validation preds]:\n{}\n".format(k_results['preds']))
    #             f.write("[validation truths]:\n{}\n\n".format(k_results['truths']))
    #             #f.write("\n[training epochs preds and truths]:\n{}\n\n".format("\n".join("\n<{}>\n\n{}".format(k, "\n".join("\n[{}]]\n\n{}".format(k2, "\n".join("*{}:\n{}".format(k3, v3) for k3, v3 in v2.items()) ) for k2, v2 in v.items())) for k, v in train_results['trainingepochs_preds_truth'].items())))
    #             f.write("*"*120+"\n")

    #             #f.write("\n[training epochs preds and truths]:\n{}\n\n".format(train_results['trainingepochs_preds_truth']))

    #         # check how to make this simpler
    #     validation_info = {
    #                             "Avg_Val_Accuracy": avg_validation_results["val_acc"], 
    #                             "Std_Val_Accuracy": np.std(k_results["val_acc"]),
    #                             "Avg_Val_F1": avg_validation_results["val_F1"], 
    #                             "Std_Val_F1": np.std(k_results["val_F1"]),
    #                             "Avg_Val_Precision": avg_validation_results["val_precision"],
    #                             "Avg_Val_Recall": avg_validation_results["val_recall"] ,

    #                             "K_Val_Accuracy": k_results["val_acc"],
    #                             "K_Val_F1": k_results["val_F1"],
    #                             "K_Val_Precision": k_results["val_precision"],
    #                             "K_Val_Recall": k_results["val_recall"],
    #                             # "Train_Time": training_time, "Test_Time": test_time
    #                             }
    #     validation_info.update(hyperparam_set)


    #         #   tunable_param.update(train_test_info)

    #         # save collected info to experiments results dataframe
    #         # experiment_results_df = experiment_results_df.append( train_test_info , ignore_index= True )

    #         experiment_results_df = pd.concat([experiment_results_df, pd.DataFrame([validation_info])], axis=0)
    #         # write out csv file every iteration
    #         experiment_results_df.to_csv(path_or_buf=experiment_results_df_fpath)

    #--------------------------------------------------------------------------------------------------------------------------------------------------------
    #--------------------------------------------------------------------------------------------------------------------------------------------------------
    #--------------------------------------------------------------------------------------------------------------------------------------------------------
    #--------------------------------------------------------------------------------------------------------------------------------------------------------
    if test_on_all_folds__or__final_test == "final_test":

         # Instantiate Experiments Results Dataframe
        colnames= list(search_space[0].keys()) + [t+"_"+m for t,m in list(product(["Final_Test"],["Accuracy","F1","Precision","Recall"]))] +\
                                                         ["FN", "FP", "TN", "TP"] 
        final_test_results_df = pd.DataFrame(columns=colnames)

        def convert_to_label(value):
            if "benign" in value.lower(): return 0
            elif "malware" in value.lower(): return 1
            else: ValueError("no benign or malware substr")


        # X_ = X.drop("data_name", axis = 1)
        # y_ = X['data_name'].apply(convert_to_label)
        final_test_X_ = final_test_X.drop("data_name", axis = 1)
        final_test_y_ = final_test_X['data_name'].apply(convert_to_label)

        print(f"Benign: #train:{X['data_name'].apply(convert_to_label).value_counts()[0]}, #final-test:{final_test_y_.value_counts()[0]},\nMalware: #train:{X['data_name'].apply(convert_to_label).value_counts()[1]}, #final-test:{final_test_y_.value_counts()[1]}",
                flush=True)


        # Added by JY @ 2023-07-31
        # 
        # > Bootstrap here.
        # https://machinelearningmastery.com/a-gentle-introduction-to-the-bootstrap-method/
        # scikit-learn bootstrap
        from sklearn.utils import resample

        N = len(search_space)
        N_bootstrapped_Xs = []
        N_OOB_Xs = []

        for seed in range(N):
            bootstrapped_X = resample( X, 
                             replace=True, 
                             n_samples=len(X), 
                             random_state= seed 
                            )
            N_bootstrapped_Xs.append( bootstrapped_X ) # list of pd.datafrmaes

            OOB_indices = np.setdiff1d(X.index, bootstrapped_X.index)
            OOB_X = X.loc[OOB_indices]
            N_OOB_Xs.append(OOB_X)


        Ensemble_of_models = []
        OOB_infos = []
        for hyperparam_set, bootstrapped_X, OOB_X in zip(search_space, N_bootstrapped_Xs, N_OOB_Xs):

            # ---------------------------------------------------------------------------------------
            if model_cls_name == 'sklearn.ensemble._gb.GradientBoostingClassifier'and\
               'xgboost' in search_space_option.lower():

               model = model_cls(
                                 n_estimators= hyperparam_set['n_estimators'], 
                                 learning_rate= hyperparam_set['learning_rate'], 
                                 max_depth= hyperparam_set['max_depth'], 
                                 min_samples_split= hyperparam_set['min_samples_split'], 
                                 min_samples_leaf= hyperparam_set['min_samples_leaf'], 
                                 subsample= hyperparam_set['subsample'], 
                                 max_features= hyperparam_set['max_features'],
                                 loss= hyperparam_set['loss'],
                                 random_state= hyperparam_set['random_state']
                                 )


            elif model_cls_name == 'sklearn.ensemble._forest.RandomForestClassifier'and\
               'randomforest' in search_space_option.lower():
               model = model_cls(
                                 n_estimators= hyperparam_set['n_estimators'],
                                 criterion= hyperparam_set['criterion'], 
                                 max_depth= hyperparam_set['max_depth'],
                                 min_samples_split= hyperparam_set['min_samples_split'], 
                                 min_samples_leaf= hyperparam_set['min_samples_leaf'], 
                                 max_features= hyperparam_set['max_features'],
                                 bootstrap= hyperparam_set['bootstrap'],
                                 random_state= hyperparam_set['random_state']
                                 )


            elif model_cls_name == 'sklearn.linear_model._logistic.LogisticRegression'and\
               'logistic' in search_space_option.lower():
               model = model_cls()

            elif model_cls_name == 'sklearn.svm' and\
               'svm' in search_space_option:
               model = model_cls()

            else:
               ValueError(f"{model_cls_name} is not supported", flush = True)

            bootstrapped_X_ = bootstrapped_X.drop("data_name", axis = 1)
            bootstrapped_y_ = bootstrapped_X['data_name'].apply(convert_to_label)

            model.fit(X = bootstrapped_X_, y = bootstrapped_y_)


            OOB_X_ = OOB_X.drop("data_name", axis = 1)
            OOB_y_ = OOB_X['data_name'].apply(convert_to_label)

            OOB_preds = model.predict(X = OOB_X_)
            OOB_accuracy = sklearn.metrics.accuracy_score(y_true = OOB_y_, y_pred = OOB_preds)
            OOB_f1 = sklearn.metrics.f1_score(y_true = OOB_y_, y_pred = OOB_preds)
            OOB_precision = sklearn.metrics.precision_score(y_true = OOB_y_, y_pred = OOB_preds)
            OOB_recall = sklearn.metrics.recall_score(y_true = OOB_y_, y_pred = OOB_preds)
            OOB_tn, OOB_fp, OOB_fn, OOB_tp = sklearn.metrics.confusion_matrix(OOB_y_, OOB_preds).ravel()
            OOB_info = {
                        "OOB_Test_Accuracy": OOB_accuracy, 
                        "OOB_Test_F1": OOB_f1, 
                        "OOB_Test_Precision": OOB_precision,
                        "OOB_Test_Recall": OOB_recall ,

                        "OOB_FN": OOB_fn,
                        "OOB_FP": OOB_fp,
                        "OOB_TN": OOB_tn,
                        "OOB_TP": OOB_tp,
                                    # "Train_Time": training_time, "Test_Time": test_time
                        }

            # could test-on OOB-data of bootstrap               
            print(OOB_info, flush=True)
            OOB_infos.append(OOB_info)
            
            # gather the ensembles
            Ensemble_of_models.append(model)

         #
        # Now using the Ensemble-of-models, perform the ensemble-final-test 

        def find_matching_key(key, dictionary):
            for dict_key in dictionary.keys():
               if key in dict_key:
                     return dict_key
            return None

        def most_common_element(lst):
            counts = Counter(lst)
            return max(counts, key=counts.get)


        majority_voting_dict = defaultdict(list)

        for index, final_test_datapoint in final_test_X.iterrows():
           
           final_test_datapoint_x = pd.DataFrame( final_test_datapoint.drop("data_name") ).T
           final_test_datapoint_y = convert_to_label(final_test_datapoint['data_name'])        
           final_test_datapoint_dataname = final_test_datapoint['data_name']
      
           for model in Ensemble_of_models:

               pred = model.predict(X = final_test_datapoint_x ) # modified return-statement for trainer.train() for this.

               majority_voting_dict[final_test_datapoint_dataname].append(int(pred))

        majority_voted_dict = dict()
        for key in majority_voting_dict:
            majority_voted_dict[key] = most_common_element(majority_voting_dict[key])

        truth_list = [ 0 if "benign" in k.lower() else 1 for k,v in majority_voted_dict.items() ] 
        majority_votes_list = [v for k,v in majority_voted_dict.items()]


        ensemble_final_test_accuracy = sklearn.metrics.accuracy_score(y_true = truth_list, y_pred = majority_votes_list)
        ensemble_final_test_f1 = sklearn.metrics.f1_score(y_true = truth_list, y_pred = majority_votes_list)
        ensemble_final_test_precision = sklearn.metrics.precision_score(y_true = truth_list, y_pred = majority_votes_list)
        ensemble_final_test_recall = sklearn.metrics.recall_score(y_true = truth_list, y_pred = majority_votes_list)

        ensemble_final_tn, ensemble_final_fp, ensemble_final_fn, ensemble_final_tp = sklearn.metrics.confusion_matrix(truth_list, majority_votes_list).ravel()


        # Now using the Ensemble-of-models, perform the final-test SEPARATELY (for comparison-purpose) -------------------------------------------------------------------------------

        
        separate_final_tests_info_df = pd.DataFrame(columns=['model',
                                                          'indiv_final_test_accuracy', 'indiv_final_test_f1', 'indiv_final_test_precision', 'indiv_final_test_recall',
                                                          'indiv_final_test_tn', 'indiv_final_test_fp', 'indiv_final_test_fn', 'indiv_final_test_tp'])
        for model in Ensemble_of_models:

            preds = model.predict(X = final_test_X_ ) # modified return-statement for trainer.train() for this.

               # final_test_y_ = final_test_X['data_name'].apply(convert_to_label)

            individual_final_test_accuracy = sklearn.metrics.accuracy_score(y_true = final_test_y_, y_pred = preds)
            individual_final_test_f1 = sklearn.metrics.f1_score(y_true = final_test_y_, y_pred = preds)
            individual_final_test_precision = sklearn.metrics.precision_score(y_true = final_test_y_, y_pred = preds)
            individual_final_test_recall = sklearn.metrics.recall_score(y_true = final_test_y_, y_pred = preds)
            individual_final_tn, individual_final_fp, individual_final_fn, individual_final_tp = sklearn.metrics.confusion_matrix(final_test_y_, preds).ravel()
            
            separate_final_test_info = {  "model": str(model),
                                          "indiv_final_test_accuracy": individual_final_test_accuracy,
                                          "indiv_final_test_f1": individual_final_test_f1,    
                                          "indiv_final_test_precision": individual_final_test_precision,    
                                          "indiv_final_test_recall": individual_final_test_recall,    
                                          "indiv_final_test_tn": individual_final_tn,    
                                          "indiv_final_test_fp": individual_final_fp,    
                                          "indiv_final_test_fn": individual_final_fn,
                                          "indiv_final_test_tp": individual_final_tp,                                                     
                                       }
            separate_final_tests_info_df = pd.concat([separate_final_tests_info_df, pd.DataFrame([separate_final_test_info])], ignore_index = True)
        
        print(separate_final_tests_info_df, flush = True)
        print(f"separate_final_tests_info_df['indiv_final_test_accuracy'].mean(): {separate_final_tests_info_df['indiv_final_test_accuracy'].mean()}" )
        print(f"separate_final_tests_info_df['indiv_final_test_f1'].mean(): {separate_final_tests_info_df['indiv_final_test_f1'].mean()}" )
        print(f"separate_final_tests_info_df['indiv_final_test_accuracy'].max(): {separate_final_tests_info_df['indiv_final_test_accuracy'].max()}" )
        print(f"separate_final_tests_info_df['indiv_final_test_f1'].max(): {separate_final_tests_info_df['indiv_final_test_f1'].max()}" )
        print("", flush=True)
        print(f"search_space_option: {search_space_option}", flush=True)
        print(f"signal_amplification_option: {signal_amplification_option}", flush=True)
        print(f"N:{N}", flush=True)
        print(f"ensemble_final_test_accuracy: {ensemble_final_test_accuracy}", flush=True)
        print(f"ensemble_final_test_f1: {ensemble_final_test_f1}", flush=True)
        print(f"ensemble_final_fn: {ensemble_final_fn}", flush=True)
        print(f"ensemble_final_fp: {ensemble_final_fp}", flush=True)
        print(f"ensemble_final_tn: {ensemble_final_tn}", flush=True)
        print(f"ensemble_final_tp: {ensemble_final_tp}", flush=True)
        print()


         #       final_test_accuracy = sklearn.metrics.accuracy_score(y_true = final_test_y_, y_pred = preds)
         #       final_test_f1 = sklearn.metrics.f1_score(y_true = final_test_y_, y_pred = preds)
         #       final_test_precision = sklearn.metrics.precision_score(y_true = final_test_y_, y_pred = preds)
         #       final_test_recall = sklearn.metrics.recall_score(y_true = final_test_y_, y_pred = preds)
         #       tn, fp, fn, tp = sklearn.metrics.confusion_matrix(final_test_y_, preds).ravel()
                                 
         #       final_test_info = {
         #                                  "Final_Test_Accuracy": final_test_accuracy, 
         #                                  "Final_Test_F1": final_test_f1, 
         #                                  "Final_Test_Precision": final_test_precision,
         #                                  "Final_Test_Recall": final_test_recall ,

         #                                  "FN": fn,
         #                                  "FP": fp,
         #                                  "TN": tn,
         #                                  "TP": tp,
         #                                  # "Train_Time": training_time, "Test_Time": test_time
         #                                  }
         #   final_test_info.update(hyperparam_set)


        #   tunable_param.update(train_test_info)

        # save collected info to experiments results dataframe
        # experiment_results_df = experiment_results_df.append( train_test_info , ignore_index= True )

      #   final_test_results_df = pd.concat([final_test_results_df, pd.DataFrame([final_test_info])], axis=0)
      #   # write out csv file every iteration
      #   time.sleep(0.05)
      #   final_test_results_df.to_csv(path_or_buf=final_test_results_df_fpath)



