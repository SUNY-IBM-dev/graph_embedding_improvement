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
from source.model import GNNBasic, LocalMeanPool,  SumPool, GlobalMeanPool, GlobalMaxPool

sys.path.append("/data/d1/jgwak1/stratkfold/source")
from source.gnn_signal_amplification import GNN_Signal_Amplification__ver1,  GNN_Signal_Amplification__ver2

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

##############################################################################################################################

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('-data', '--dataset_choice', nargs = 1, type = str, default = ['Dataset_1_B148_M148'])  # Dataset_2_B239_M239  # Dataset_1_B148_M148
    parser.add_argument('-k', '--K', nargs = 1, type = int, default = [10])  
    parser.add_argument('-ne', '--num_epochs', nargs = 1, type = int, default = [1000])
    parser.add_argument('-bs', '--batch_size', nargs = 1, type = int, default = [32])
    parser.add_argument('-d', '--device', nargs = 1, type = str, default = ['cuda:1'])
    parser.add_argument('-dsr', '--data_split_ratio', nargs = 3, type = float, default = [1.0, 0, 0] ) 
    parser.add_argument('-bc', '--best_criteria', nargs = 1, type = str, default = ['train_loss'] ) 

    model_cls_map = {
                    "GNN_Signal_Amplification__ver1": GNN_Signal_Amplification__ver1,
                    "GNN_Signal_Amplification__ver2": GNN_Signal_Amplification__ver2,
                    } 
    parser.add_argument('-mod_cls', '--model_cls',
                        choices= ['GNN_Signal_Amplification__ver1',
                                  'GNN_Signal_Amplification__ver2', 
                                  ], 
                                  default = ["GNN_Signal_Amplification__ver2"])
                        
    parser.add_argument('-sig_amp_opt', 
                        '--signal_amplification_option', 
                        choices= ['signal_amplified__event_1gram',
                                  'signal_amplified__event_1gram_nodetype_5bit', 
                                  'signal_amplified__event_1gram_nodetype_5bit_and_Ahoc_Identifier',
                                  ], 
                                  default = ["signal_amplified__event_1gram"])

    parser.add_argument('-save_bestmodel', '--save_bestmodel_fitted_on_whole_data', nargs = 1, type = strtobool, default = [False] )

    # cmd args
    dataset_choice = parser.parse_args().dataset_choice[0]
    K = parser.parse_args().K[0]
    num_epochs = parser.parse_args().num_epochs[0]
    batch_size = parser.parse_args().batch_size[0]
    device = parser.parse_args().device[0]
    data_split_ratio = list(parser.parse_args().data_split_ratio)
    best_criteria = parser.parse_args().best_criteria[0]

    # ------------------------------------------------------------------------------------------------------------------
    if type(parser.parse_args().model_cls) == list:
       # run from python directly
       model_cls = model_cls_map[ parser.parse_args().model_cls[0] ]
    else: 
       # from a bash file
       model_cls = model_cls_map[ parser.parse_args().model_cls ]

    
    signal_amplification_option = parser.parse_args().signal_amplification_option[0]
    # ------------------------------------------------------------------------------------------------------------------

    save_bestmodel_fitted_on_whole_data = parser.parse_args().save_bestmodel_fitted_on_whole_data[0]

    saved_models_dirpath = "/data/d1/jgwak1/stratkfold/saved_models"
    saved_temp_models_dirpath = "/data/d1/jgwak1/stratkfold/saved_temp_models"
   
    model_cls_name = re.search(r"'(.*?)'", str(model_cls)).group(1).lstrip('.source.model')

    if save_bestmodel_fitted_on_whole_data == False:
       experiment_results_df_fpath = f"/data/d1/jgwak1/stratkfold/{dataset_choice}__{model_cls_name}__{signal_amplification_option}__{K}foldCV_{num_epochs}epochs_{device}_{datetime.now().strftime('%Y-%m-%d_%H%M%S')}.csv"

    trace_filename = 'traces_stratkfold_double_strat__SignalAmp_GNN___generated@'+str(datetime.now())+".txt" 

    ###############################################################################################################################################
    # Set data paths
   

    projection_benign_datapath_dict = {
      # all node and edge attrs
      
      # all node attrs, only task-name edge attr      
      # "prj_3_node_all_node_attrs_edge_attr_only_TaskName_OLDdata" : "/home/jgwak1/tabby/OFFLINE_TRAINTEST_OLD_PROCESSED_DATA_FOR_PRETRAINED_GAT/TRAIN_DATA/Processed_Benign_ONLY_TaskName_edgeattr"
      "RESULTS__RF_4gram_flatten_subgraph_psh__at_20230510_23-00-46": "/data/d1/jgwak1/tabby/BASELINE_COMPARISONS/Sequential/RF+Ngrams/RESULTS__RF_4gram_flatten_subgraph_psh__at_20230510_23-00-46/GAT_TrainSet_same_as_RFexp/Processed_Benign_ONLY_TaskName_edgeattr",
      "RESULTS__RF_1gram_flatten_subgraph_psh__at_20230609_13-17-38_EdgeFeatMigrated": "/data/d1/jgwak1/tabby/BASELINE_COMPARISONS/Sequential/RF+Ngrams/RESULTS__RF_1gram_flatten_subgraph_psh__at_20230609_13-17-38_EdgeFeatMigrated/GNN_TrainSet_same_as_RFexp/Processed_Benign_ONLY_TaskName_edgeattr",
      
      "RESULTS__RF_1gram_flatten_subgraph_psh__at_20230612_21-47-21": "/data/d1/jgwak1/tabby/BASELINE_COMPARISONS/Sequential/RF+Ngrams/RESULTS__RF_1gram_flatten_subgraph_psh__at_20230612_21-47-21/GNN_TrainSet_same_as_RFexp/Processed_Benign_ONLY_TaskName_edgeattr",
      "RESULTS__RF_1gram_flatten_subgraph_psh__at_20230612_22-19-52_EdgeFeatMigrated": "/data/d1/jgwak1/tabby/BASELINE_COMPARISONS/Sequential/RF+Ngrams/RESULTS__RF_1gram_flatten_subgraph_psh__at_20230612_22-19-52_EdgeFeatMigrated/GNN_TrainSet_same_as_RFexp/Processed_Benign_ONLY_TaskName_edgeattr",

      "RESULTS__RF_1gram_flatten_subgraph_psh__at_20230706_14-46-10__5BitPlusADHOC": "/data/d1/jgwak1/tabby/BASELINE_COMPARISONS/Sequential/RF+Ngrams/RESULTS__RF_1gram_flatten_subgraph_psh__at_20230706_14-46-10__5BitPlusADHOC/GNN_TrainSet_same_as_RFexp/Processed_Benign_ONLY_TaskName_edgeattr",
      "RESULTS__RF_1gram_flatten_subgraph_psh__at_20230706_14-57-13__5BitPlusADHOC_EdgeFeatMigrated": "/data/d1/jgwak1/tabby/BASELINE_COMPARISONS/Sequential/RF+Ngrams/RESULTS__RF_1gram_flatten_subgraph_psh__at_20230706_14-57-13__5BitPlusADHOC_EdgeFeatMigrated/GNN_TrainSet_same_as_RFexp/Processed_Benign_ONLY_TaskName_edgeattr",

      "RESULTS__RF_1gram_flatten_subgraph_psh__at_20230717_17-00-27__SupposedlyCleanerDataset__5BitPlusADHOC_EdgeFeatMigrated": \
         "/data/d1/jgwak1/tabby/BASELINE_COMPARISONS/Sequential/RF+Ngrams/RESULTS__RF_1gram_flatten_subgraph_psh__at_20230717_17-00-27__SupposedlyCleanerDataset__5BitPlusADHOC_EdgeFeatMigrated/GNN_TrainSet_same_as_RFexp/Processed_Benign_ONLY_TaskName_edgeattr",

      "RESULTS__RF_1gram_flatten_subgraph_psh__at_20230718_11-53-31__SupposedlyCleanerDataset__5BitPlusADHOC": \
         "/data/d1/jgwak1/tabby/BASELINE_COMPARISONS/Sequential/RF+Ngrams/RESULTS__RF_1gram_flatten_subgraph_psh__at_20230718_11-53-31__SupposedlyCleanerDataset__5BitPlusADHOC/GNN_TrainSet_same_as_RFexp/Processed_Benign_ONLY_TaskName_edgeattr",

      "Dataset_1_B148_M148": \
         "/data/d1/jgwak1/tabby/BASELINE_COMPARISONS/Sequential/RF+Ngrams/RESULTS__RF_1gram_flatten_subgraph_psh__at_20230718_11-53-31__SupposedlyCleanerDataset__5BitPlusADHOC/GNN_TrainSet_same_as_RFexp/Processed_Benign_ONLY_TaskName_edgeattr",
      # ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
      "Dataset_2_B239_M239": \
         "/data/d1/jgwak1/tabby/OFFLINE_TRAINTEST_DATA/dataset_2/GNN_TrainSet_same_as_RFexp/Processed_Benign_ONLY_TaskName_edgeattr",

    }

    projection_malware_datapath_dict = {
      # all node-attrs
      
      # all node attrs, only task-name edge attr      
      # "prj_3_node_all_node_attrs_edge_attr_only_TaskName_OLDdata" : "/home/jgwak1/tabby/OFFLINE_TRAINTEST_OLD_PROCESSED_DATA_FOR_PRETRAINED_GAT/TRAIN_DATA/Processed_Malware_ONLY_TaskName_edgeattr", # 
      "RESULTS__RF_4gram_flatten_subgraph_psh__at_20230510_23-00-46": "/data/d1/jgwak1/tabby/BASELINE_COMPARISONS/Sequential/RF+Ngrams/RESULTS__RF_4gram_flatten_subgraph_psh__at_20230510_23-00-46/GAT_TrainSet_same_as_RFexp/Processed_Malware_ONLY_TaskName_edgeattr",
      "RESULTS__RF_1gram_flatten_subgraph_psh__at_20230609_13-17-38_EdgeFeatMigrated": "/data/d1/jgwak1/tabby/BASELINE_COMPARISONS/Sequential/RF+Ngrams/RESULTS__RF_1gram_flatten_subgraph_psh__at_20230609_13-17-38_EdgeFeatMigrated/GNN_TrainSet_same_as_RFexp/Processed_Malware_ONLY_TaskName_edgeattr",
      
      "RESULTS__RF_1gram_flatten_subgraph_psh__at_20230612_21-47-21": "/data/d1/jgwak1/tabby/BASELINE_COMPARISONS/Sequential/RF+Ngrams/RESULTS__RF_1gram_flatten_subgraph_psh__at_20230612_21-47-21/GNN_TrainSet_same_as_RFexp/Processed_Malware_ONLY_TaskName_edgeattr",      
      "RESULTS__RF_1gram_flatten_subgraph_psh__at_20230612_22-19-52_EdgeFeatMigrated": "/data/d1/jgwak1/tabby/BASELINE_COMPARISONS/Sequential/RF+Ngrams/RESULTS__RF_1gram_flatten_subgraph_psh__at_20230612_22-19-52_EdgeFeatMigrated/GNN_TrainSet_same_as_RFexp/Processed_Malware_ONLY_TaskName_edgeattr",
    
      "RESULTS__RF_1gram_flatten_subgraph_psh__at_20230706_14-46-10__5BitPlusADHOC": "/data/d1/jgwak1/tabby/BASELINE_COMPARISONS/Sequential/RF+Ngrams/RESULTS__RF_1gram_flatten_subgraph_psh__at_20230706_14-46-10__5BitPlusADHOC/GNN_TrainSet_same_as_RFexp/Processed_Malware_ONLY_TaskName_edgeattr",
      "RESULTS__RF_1gram_flatten_subgraph_psh__at_20230706_14-57-13__5BitPlusADHOC_EdgeFeatMigrated": "/data/d1/jgwak1/tabby/BASELINE_COMPARISONS/Sequential/RF+Ngrams/RESULTS__RF_1gram_flatten_subgraph_psh__at_20230706_14-57-13__5BitPlusADHOC_EdgeFeatMigrated/GNN_TrainSet_same_as_RFexp/Processed_Malware_ONLY_TaskName_edgeattr",
    
      "RESULTS__RF_1gram_flatten_subgraph_psh__at_20230717_17-00-27__SupposedlyCleanerDataset__5BitPlusADHOC_EdgeFeatMigrated": \
         "/data/d1/jgwak1/tabby/BASELINE_COMPARISONS/Sequential/RF+Ngrams/RESULTS__RF_1gram_flatten_subgraph_psh__at_20230717_17-00-27__SupposedlyCleanerDataset__5BitPlusADHOC_EdgeFeatMigrated/GNN_TrainSet_same_as_RFexp/Processed_Malware_ONLY_TaskName_edgeattr",

      "RESULTS__RF_1gram_flatten_subgraph_psh__at_20230718_11-53-31__SupposedlyCleanerDataset__5BitPlusADHOC": \
         "/data/d1/jgwak1/tabby/BASELINE_COMPARISONS/Sequential/RF+Ngrams/RESULTS__RF_1gram_flatten_subgraph_psh__at_20230718_11-53-31__SupposedlyCleanerDataset__5BitPlusADHOC/GNN_TrainSet_same_as_RFexp/Processed_Malware_ONLY_TaskName_edgeattr",


      "Dataset_1_B148_M148": \
         "/data/d1/jgwak1/tabby/BASELINE_COMPARISONS/Sequential/RF+Ngrams/RESULTS__RF_1gram_flatten_subgraph_psh__at_20230718_11-53-31__SupposedlyCleanerDataset__5BitPlusADHOC/GNN_TrainSet_same_as_RFexp/Processed_Malware_ONLY_TaskName_edgeattr",
      # ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
      "Dataset_2_B239_M239": \
         "/data/d1/jgwak1/tabby/OFFLINE_TRAINTEST_DATA/dataset_2/GNN_TrainSet_same_as_RFexp/Processed_Malware_ONLY_TaskName_edgeattr",

    }


    _num_classes = 2  # number of class labels and always binary classification.
   #  _dim_node = 5+71    # num node features ; the #feats
   #  _dim_edge = None    # (or edge_dim) ; num edge features
   #  _dim_node = 5    # num node features ; the #feats
   #  _dim_edge = 72    # (or edge_dim) ; num edge features


   #  dataset_choice = "Dataset_2_B239_M239"
    _dim_node = 46   # num node features ; the #feats
    _dim_edge = 72    # (or edge_dim) ; num edge features


    if signal_amplification_option == "signal_amplified__event_1gram":
        dim_node__expanded_for_compatibility_with_signal_amplification = _dim_edge - 1
    elif signal_amplification_option == "signal_amplified__event_1gram_nodetype_5bit":
        dim_node__expanded_for_compatibility_with_signal_amplification = 5 + (_dim_edge - 1)     
    elif signal_amplification_option == "signal_amplified__event_1gram_nodetype_5bit_and_Ahoc_Identifier":
        dim_node__expanded_for_compatibility_with_signal_amplification = _dim_node + (_dim_edge - 1)     
    else:
        ValueError("Invalid signal-amplification-option")
    model_dim_node = dim_node__expanded_for_compatibility_with_signal_amplification
    model_dim_edge = _dim_edge
    drop_dim_edge_timescalar = False





    _benign_train_data_path = projection_benign_datapath_dict[dataset_choice]
    _malware_train_data_path = projection_malware_datapath_dict[dataset_choice]
    print("data-paths:", flush=True)
    print(f"_benign_train_data_path: {_benign_train_data_path}", flush=True)
    print(f"_malware_train_data_path: {_malware_train_data_path}", flush=True)
    print(f"\n_dim_node: {_dim_node}", flush=True)
    print(f"_dim_edge: {_dim_edge}", flush=True)
    print(f"model_dim_node: {model_dim_node}", flush=True)
    print(f"model_dim_edge: {model_dim_edge}\n", flush=True)


   
    # trace file for preds and truth comparison for models during gridsearch
    
    f = open( trace_filename , 'w' )
    f.close()

    # Load both benign and malware graphs """
    dataprocessor = LoadGraphs()
    benign_train_dataset = dataprocessor.parse_all_data__expand_dim_node_compatible_with_SignalAmplification(_benign_train_data_path, _dim_node, _dim_edge,
                                                                                                             signal_amplification_option= signal_amplification_option,
                                                                                                             )
    malware_train_dataset = dataprocessor.parse_all_data__expand_dim_node_compatible_with_SignalAmplification(_malware_train_data_path, _dim_node, _dim_edge,
                                                                                                             signal_amplification_option= signal_amplification_option)
    train_dataset = benign_train_dataset + malware_train_dataset
   
    print('+ data loaded #Malware = {} | #Benign = {}'.format(len(malware_train_dataset), len(benign_train_dataset)), flush=True)




    ######################################################################################################################################################

    # "Memory-Inefficient BUT Torch-Geometric-Conv-Layer Friendly Data Modificiation"
    # *   Goal: Make our Directed Subgraph into Undirected Subgraph.
    # * Reason: 
    #
    # * References
    #           https://github.com/pyg-team/pytorch_geometric/discussions/3043
    #           https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/conv/gat_conv.html#GATConv
    #           https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/transforms/to_undirected.html


   #  from torch_geometric.utils import to_undirected


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



    ####################################################################################################################################################
    ####################################################################################################################################################
    ####################################################################################################################################################




    def GNN_Signal_Amplification__ver1__RandomGridSearch() -> dict :

      search_space = dict()
      search_space['embedding_dim'] = [ 
                                             # 2 hops
                                             [32],
                                       ]

      search_space['ffn_dims'] = [ 
                                             # 2 hops
                                             [16, 8],
                                             [16, 8, 4],
                                             [16],


                                       ]


      search_space['activation_fn'] = [ 
                                             # 2 hops
                                             nn.ReLU, 
                                       ]

      search_space['neighborhood_aggr'] = [ 
                                             "add"
                                       ]


      search_space['pool'] = [ 
                                             GlobalMaxPool,
                                             GlobalMeanPool
                                       ]




      # seed is not gnn-model hyperparam, but can be viewed part of model as it can control initial-model-weights.
      # search_space['lr'] = [ 5e-09, 5e-08, 5e-07, 1e-06, 1e-05, 1e-04 ]             # learning rate

      search_space['lr'] = [  1e-03, 3e-03, 5e-03, 7e-03 ]             # learning rate set2
      # try model_dropout == 0?
      search_space['dropout_level'] = [ 0, 0.01, 0.1, 0.2, 0.5 ]                  # dropout : dropout level ;  the dropout probability
      search_space['num_heads'] = [1]                         # num heads : number of attention heads (unique to GAT) ; multi-headed attentions

      search_space["split_shuffle_seed"] = [ 100 ]
      search_space["weight_init_seed"] = [ 200, 1200, 2200, 3200, 4200 ]

      #-----------------------------------------------------------------------------------------------
      # TO_SKIP tuples set --format is [dim_hidden, lr, dropout, pool, act_fn, weight_init_seed, split_shuffle_seed]  --- In log, look for "[CV 1/5]"
      to_skip_tuples_set = \
      [
      ]


      print("-"*100)
      print(f"to_skip_tuples_set:", flush = True )
      print(*to_skip_tuples_set, sep= "\n", flush=True)
      print("-"*100)

      #-----------------------------------------------------------------------------------------------
      manual_space = []
      # Change order of these for-loops as the
      for embedding_dim in search_space['embedding_dim']:
         for ffn_dims in search_space['ffn_dims']:
                for activation_fn in search_space['activation_fn']:
                   for neighborhood_aggr in search_space['neighborhood_aggr']:
                        for pool in search_space['pool']:
                           for lr in search_space['lr']:
                                 for dropout_level in search_space['dropout_level']:
                                    for weight_init_seed in search_space["weight_init_seed"]:
                                       for split_shuffle_seed in search_space["split_shuffle_seed"]:
                                       
                                             print(f"appending {[embedding_dim, ffn_dims, activation_fn, neighborhood_aggr, pool, lr, dropout_level, weight_init_seed, split_shuffle_seed]}", flush = True )
                                             manual_space.append(
                                                   {
                                                   'embedding_dim': embedding_dim,
                                                   'ffn_dims': ffn_dims,
                                                   'activation_fn': activation_fn,
                                                   'neighborhood_aggr': neighborhood_aggr,
                                                   'pool': pool,
                                                   'lr': lr,
                                                   'dropout_level': dropout_level,
                                                   'weight_init_seed': weight_init_seed, 
                                                   'split_shuffle_seed': split_shuffle_seed,                    
                                                   }
                                             )
      random.shuffle(manual_space)
      return manual_space



    def GNN_Signal_Amplification__ver1__for_Debugging() -> dict :

      search_space = dict()
      search_space['embedding_dim'] = [ 
                                             # 2 hops
                                             [32],
                                       ]

      search_space['ffn_dims'] = [ 
                                             # 2 hops
                                             [16, 8],
                                       ]

      search_space['conv_activation_fn'] = [ 
                                             # 2 hops
                                             nn.ReLU, 
                                       ]

      search_space['activation_fn'] = [ 
                                             # 2 hops
                                             nn.ReLU, 
                                       ]

      search_space['neighborhood_aggr'] = [ 
                                             "add"
                                       ]


      search_space['pool'] = [ 
                                             GlobalMaxPool,
                                       ]




      # seed is not gnn-model hyperparam, but can be viewed part of model as it can control initial-model-weights.
      # search_space['lr'] = [ 5e-09, 5e-08, 5e-07, 1e-06, 1e-05, 1e-04 ]             # learning rate

      search_space['lr'] = [  1e-03]             # learning rate set2
      # try model_dropout == 0?
      search_space['dropout_level'] = [ 0]                  # dropout : dropout level ;  the dropout probability
      search_space['num_heads'] = [1]                         # num heads : number of attention heads (unique to GAT) ; multi-headed attentions

      search_space["split_shuffle_seed"] = [ 100 ]
      search_space["weight_init_seed"] = [ 200 ]

      #-----------------------------------------------------------------------------------------------
      # TO_SKIP tuples set --format is [dim_hidden, lr, dropout, pool, act_fn, weight_init_seed, split_shuffle_seed]  --- In log, look for "[CV 1/5]"
      to_skip_tuples_set = \
      [
      ]


      print("-"*100)
      print(f"to_skip_tuples_set:", flush = True )
      print(*to_skip_tuples_set, sep= "\n", flush=True)
      print("-"*100)

      #-----------------------------------------------------------------------------------------------
      manual_space = []
      # Change order of these for-loops as the
      for embedding_dim in search_space['embedding_dim']:
         for ffn_dims in search_space['ffn_dims']:
                for activation_fn in search_space['activation_fn']:
                   for neighborhood_aggr in search_space['neighborhood_aggr']:
                        for pool in search_space['pool']:
                           for lr in search_space['lr']:
                                 for dropout_level in search_space['dropout_level']:
                                    for weight_init_seed in search_space["weight_init_seed"]:
                                       for split_shuffle_seed in search_space["split_shuffle_seed"]:
                                       
                                             print(f"appending {[embedding_dim, ffn_dims, activation_fn, neighborhood_aggr, pool, lr, dropout_level, weight_init_seed, split_shuffle_seed]}", flush = True )
                                             manual_space.append(
                                                   {
                                                   'embedding_dim': embedding_dim,
                                                   'ffn_dims': ffn_dims,
                                                   'activation_fn': activation_fn,
                                                   'neighborhood_aggr': neighborhood_aggr,
                                                   'pool': pool,
                                                   'lr': lr,
                                                   'dropout_level': dropout_level,
                                                   'weight_init_seed': weight_init_seed, 
                                                   'split_shuffle_seed': split_shuffle_seed,                    
                                                   }
                                             )
      random.shuffle(manual_space)
      return manual_space




    def GNN_Signal_Amplification__ver2__RandomGridSearch() -> dict :

      search_space = dict()

      search_space['ffn_dims'] = [ 
                                             # 2 hops
                                             [36, 18],
                                             [36, 16],
                                             [36, 16, 8],                                             
                                             [32, 16],
                                             [32, 16 , 8],
                                             [32, 16 , 8, 4],
                                       ]


      search_space['activation_fn'] = [ 
                                             # 2 hops
                                             nn.ReLU, 
                                       ]

      search_space['pool'] = [ 
                                             GlobalMaxPool,
                                             GlobalMeanPool
                                       ]




      # seed is not gnn-model hyperparam, but can be viewed part of model as it can control initial-model-weights.
      # search_space['lr'] = [ 5e-09, 5e-08, 5e-07, 1e-06, 1e-05, 1e-04 ]             # learning rate

      search_space['lr'] = [  1e-03, 3e-03, 5e-03, 7e-03 ]             # learning rate set2
      # try model_dropout == 0?
      search_space['dropout_level'] = [ 0, 0.01, 0.1, 0.2, 0.5 ]                  # dropout : dropout level ;  the dropout probability

      search_space["split_shuffle_seed"] = [ 100 ]
      search_space["weight_init_seed"] = [ 200, 1200, 2200, 3200, 4200 ]

      #-----------------------------------------------------------------------------------------------
      # TO_SKIP tuples set --format is [dim_hidden, lr, dropout, pool, act_fn, weight_init_seed, split_shuffle_seed]  --- In log, look for "[CV 1/5]"
      to_skip_tuples_set = \
      [
      ]


      print("-"*100)
      print(f"to_skip_tuples_set:", flush = True )
      print(*to_skip_tuples_set, sep= "\n", flush=True)
      print("-"*100)

      #-----------------------------------------------------------------------------------------------
      manual_space = []
      # Change order of these for-loops as the
      for ffn_dims in search_space['ffn_dims']:
               for activation_fn in search_space['activation_fn']:
                      for pool in search_space['pool']:
                           for lr in search_space['lr']:
                                 for dropout_level in search_space['dropout_level']:
                                    for weight_init_seed in search_space["weight_init_seed"]:
                                       for split_shuffle_seed in search_space["split_shuffle_seed"]:
                                       
                                             print(f"appending {[ffn_dims, activation_fn, pool, lr, dropout_level, weight_init_seed, split_shuffle_seed]}", flush = True )
                                             manual_space.append(
                                                   {
                                                   'ffn_dims': ffn_dims,
                                                   'activation_fn': activation_fn,
                                                   'pool': pool,
                                                   'lr': lr,
                                                   'dropout_level': dropout_level,
                                                   'weight_init_seed': weight_init_seed, 
                                                   'split_shuffle_seed': split_shuffle_seed,                    
                                                   }
                                             )
      random.shuffle(manual_space)
      return manual_space



    ####################################################################################################################################################
 


    search_space = GNN_Signal_Amplification__ver1__RandomGridSearch()

   #  search_space = GNN_Signal_Amplification__ver1__for_Debugging()

   #  search_space = GNN_Signal_Amplification__ver2__RandomGridSearch()

    if save_bestmodel_fitted_on_whole_data == True:

       best_model_id = "GAT_mlp_fed_1gram_EdgeFeatMigrate_Top5_20230629_weightseed1200_doublestrat_splitseed100" # done


       print(f"best_model_id: {best_model_id}", flush=True)
       best_model_savedirpath = os.path.join(saved_models_dirpath, best_model_id)
       if not os.path.exists( best_model_savedirpath):
          os.makedirs(best_model_savedirpath) 



    ####################################################################################################################################################
    ####################################################################################################################################################
    ####################################################################################################################################################
    



    #--------------------------------------------------------------------------------------------------------------------------------------------------------
    if save_bestmodel_fitted_on_whole_data == False:
            
         # Instantiate Experiments Results Dataframe
         colnames= list(search_space[0].keys()) + [t+"_"+m for t,m in list(product(["Train","Mean_Test"],["Loss","Accuracy","Precision","Recall","F1"]))] +\
                                                         ["Std_Test_F1", "Std_Test_Accuracy"] + \
                                                         ["K_Test_"+m for m in list(["Loss","Accuracy","Precision","Recall","F1"])] + [""] + ["model_id"]
         experiment_results_df = pd.DataFrame(columns=colnames)

         ####################################################################################################################################################



         """ try out different tunable parameter combinations """
         for hyperparam_set in search_space:


            weight_init_seed = hyperparam_set['weight_init_seed']
            split_shuffle_seed = hyperparam_set['split_shuffle_seed']
            lr = hyperparam_set['lr']
            
            if "GNN_Signal_Amplification__ver1" in model_cls_name:
               model_param_dict = dict()
               model_param_dict["embedding_dim"] = hyperparam_set['embedding_dim']

               model_param_dict["signal_amplification_option"] = signal_amplification_option

               model_param_dict["ffn_dims"] = hyperparam_set['ffn_dims']
               model_param_dict["activation_fn"] = hyperparam_set['activation_fn']

               model_param_dict["neighborhood_aggr"] = hyperparam_set['neighborhood_aggr']
               model_param_dict["pool"] = hyperparam_set['pool']
               model_param_dict["dropout_level"] = hyperparam_set['dropout_level']

            

            if "GNN_Signal_Amplification__ver2" in model_cls_name:
               model_param_dict = dict()

               model_param_dict["signal_amplification_option"] = signal_amplification_option
               model_param_dict["ffn_dims"] = hyperparam_set['ffn_dims']
               model_param_dict["activation_fn"] = hyperparam_set['activation_fn']

               model_param_dict["pool"] = hyperparam_set['pool']
               model_param_dict["dropout_level"] = hyperparam_set['dropout_level']






            print(hyperparam_set, flush = True)



            # get this iteration's tunable_param
            X = train_dataset
            y = [data.y for data in train_dataset]


            # [Double-Stratification] ===========================================================================================================================
            # JY @ 2023-06-27
            # Following are datasources for beign and malware 
            # exactly same as "/data/d1/jgwak1/tabby/BASELINE_COMPARISONS/Sequential/RF+Ngrams/RFSVM_ngram_flattened_subgraph_only_psh.py"
            #   BENIGN_PATTERNS_LIST_FOR_STRATIFIED_SPLIT = [ "fleschutz", "jhochwald", "devblackops", "farag2", "jimbrig", "jrussellfreelance", "nickrod518", 
            #                                                 "redttr", "sysadmin-survival-kit", "stevencohn", "ledrago"]
            #   MALWARE_PATTERNS_LIST_FOR_STRATIFIED_SPLIT = ["empire", "invoke_obfuscation", "nishang", "poshc2", "mafia",
            #                                                 "offsec", "powershellery", "psbits", "pt-toolkit", "randomps", "smallposh",
            #                                                 "bazaar"]

            # JY @ 2023-06-27
            #      Now we want to acheive stratified-K-Fold split based on not only label but also datasource.
            #      So in each split, we want the label+datasource combination to be consistent.
            #      e.g. Across folds, we want ratio of benign-fleschutz, benign-jhochwald, ... malware-empire, malware-nishang,.. to be consistent
            #      For this, first generate a group-list.
            X_grouplist = []
            for data in X:
                  # benign ------------------------------------------
                  if "fleschutz" in data.name:
                     X_grouplist.append("benign_fleschutz")
                  if "jhochwald" in data.name:
                     X_grouplist.append("benign_jhochwald")
                  if "devblackops" in data.name:
                     X_grouplist.append("benign_devblackops")
                  if "farag2" in data.name:
                     X_grouplist.append("benign_farag2")
                  if "jimbrig" in data.name:
                     X_grouplist.append("benign_jimbrig")
                  if "jrussellfreelance" in data.name:
                     X_grouplist.append("benign_jrussellfreelance")
                  if "nickrod518" in data.name:
                     X_grouplist.append("benign_nickrod518")
                  if "redttr" in data.name:
                     X_grouplist.append("benign_redttr")
                  if "sysadmin-survival-kit" in data.name:
                     X_grouplist.append("benign_sysadmin-survival-kit")
                  if "stevencohn" in data.name:
                     X_grouplist.append("benign_stevencohn")
                  if "ledrago" in data.name:
                     X_grouplist.append("benign_ledrago")
                  # malware ------------------------------------------
                  if "empire" in data.name:
                     X_grouplist.append("malware_empire")
                  if "invoke_obfuscation" in data.name:
                     X_grouplist.append("malware_invoke_obfuscation")
                  if "nishang" in data.name:
                     X_grouplist.append("malware_nishang")
                  if "poshc2" in data.name:
                     X_grouplist.append("malware_poshc2")
                  if "mafia" in data.name:
                     X_grouplist.append("malware_mafia")
                  if "offsec" in data.name:
                     X_grouplist.append("malware_offsec")
                  if "powershellery" in data.name:
                     X_grouplist.append("malware_powershellery")
                  if "psbits" in data.name:
                     X_grouplist.append("malware_psbits")
                  if "pt-toolkit" in data.name:
                     X_grouplist.append("malware_pt-toolkit")
                  if "randomps" in data.name:
                     X_grouplist.append("malware_randomps")
                  if "smallposh" in data.name:
                     X_grouplist.append("malware_smallposh")
                  if "bazaar" in data.name:
                     X_grouplist.append("malware_bazaar")
                  if "recollected" in data.name:
                     X_grouplist.append("malware_recollected")



            # correctness of X_grouplist can be checked by following
            list(zip(X, [data.name.lstrip("Processed_SUBGRAPH_P3_") for data in X], y, X_grouplist))

            dataset_cv = StratifiedKFold(n_splits=K, 
                                          shuffle = True, 
                                          random_state=np.random.RandomState(seed=split_shuffle_seed))

            # ===========================================================================================================================

            gc.collect()
            torch.cuda.empty_cache()
            # Model-id Prefix

            #################################################################################################################
            ktest_results = []
            k_results = temp_results = mean_test_results = {}

            # id of Temp Model file that will be overwritten for this hyperparameter-set.
            model_id = "Temp_Model_for_KfoldCV_Test_Not_Fitted_on_Entire_Trainset_" + datetime.now().strftime('%Y-%m-%d_%H%M%S')

            iter_n = 0
            for train_idx, test_idx in dataset_cv.split(X, y = X_grouplist):  # JY @ 2023-06-27 : y = X_grouplist is crucial for double-stratification
                  X_train, X_test = list(map(X.__getitem__, train_idx)), list(map(X.__getitem__, test_idx))
                  y_train, y_test = np.array(y)[train_idx], np.array(y)[test_idx]

                  num_0 = np.sum(y_train==0)
                  num_1 = np.sum(y_train==1)
                  num_b_test = np.sum(y_test==0)
                  num_m_test = np.sum(y_test==1)
                  print(f"Benign: #train:{num_0}, #test:{num_b_test},\nMalware: #train:{num_1}, #test:{num_m_test}", flush=True)
                  # set this iteration's model-id

                  # print("="*100, flush=True)
                  # print(f"[Split-{iter_n} TRAIN]:\n", flush=True)
                  # print(*X_train, sep="\n", flush=True)
                  # print(*[x.name.lstrip("Processed_SUBGRAPH_P3_").rstrip(".pickle") for x in X_train], sep="\n",flush=True)
                  # print("\n", flush=True)
                  # print(f"[Split-{iter_n} TEST]:\n", flush=True)
                  # print(*X_test, sep="\n", flush=True)
                  # print(*[x.name.lstrip("Processed_SUBGRAPH_P3_").rstrip(".pickle") for x in X_test], sep="\n", flush=True)

                  #Setting the seed for model
                  # pl.seed_everything(tunable_param['seed'])

                  #Setting the seed for model
                  torch.manual_seed( weight_init_seed )   
                  if 'cuda' in device:
                     torch.cuda.manual_seed_all( weight_init_seed )

                  # instanitate classifer
                  classifier = model_cls(
                     num_classes=_num_classes, 
                     expanded_dim_node = model_dim_node,
                     edge_dim=model_dim_edge,
                     **model_param_dict
                  )



                  # set dataloader_parms
                  dataloader_params = {
                                       'batch_size' : batch_size,
                                       # 'data_split_ratio': data_split_ratio,  # train : 100% / eval: 0% / test: 0%  (test should be 0%, since we are just 'fitting' here.)
                                       'seed': split_shuffle_seed
                                    }

                  # create the trainer class
                  trainer = TrainModel(
                     model=classifier,
                     # dataset=[X_train[0:num_0-1], X_train[-num_1:]],
                     traindataset=X_train,
                     testdataset = X_test,
                     dataloader_params=dataloader_params,
                     device=device,
                     save_dir=saved_temp_models_dirpath, # needs to save model for test;
                     save_name= model_id, # it might be possible to overwrite to the save model instead of generating multiple
                     verbose= 0, # Needs to be 0 here to prevent invoking SummaryWriter in trainer.py ; 
                              # verbose { 0: prints output as text ,  -1: does not print anything on screen }
                  )

                  # set train_params
                  train_params = { 
                                 'num_epochs': num_epochs, 
                                 'num_early_stop': num_epochs,
                                 'milestones': None,
                                 'gamma': None
                                 }

                  # set optimizer_params
                  optimizer_params = { 
                                 'lr': lr, 
                                 'weight_decay': 5e-8
                              }
         
                  # train / record training time
                  train_start = datetime.now()
                  train_results = trainer.train( train_params=train_params, 
                                                optimizer_params=optimizer_params) # added return-statement for trainer.train() for this.
                  training_time =str( datetime.now() - train_start )
         
                  # test / record test-time
                  test_start = datetime.now()
                  test_results = trainer.test() # modified return-statement for trainer.train() for this.
                  ktest_results.append(test_results)
                  #print("Meng: iter_n = ", iter_n, test_results)

                  if iter_n == 0:
                     for key in test_results:
                        temp_results[key] = [test_results[key]]
                  else:
                     for key in test_results:
                        #print("Meng: ", key, test_results[key])
                        temp_results[key].append(test_results[key])
                  test_time = str( datetime.now() - test_start )
                  iter_n +=1

            print("hyparameter-set: ", hyperparam_set, flush=True)
            k_results=temp_results.copy()
            print("mean test result: ", flush=True)
            for key in temp_results:
                  if key != 'preds' and key != 'truths':
                     mean_test_results[key] = np.mean(temp_results[key])
                     print(key,":", mean_test_results[key], flush=True)
                  else:
                     mean_test_results[key] = temp_results[key]
            print("k-fold test results:", flush=True)
            for item in ktest_results:
                  print(item, flush=True)

            # write out 
            with open(trace_filename, 'a') as f:
                  f.write("*"*120+"\n")
                  f.write("model-id: {}\n\n".format(model_id))
                  f.write("param_info:{}\n".format(hyperparam_set))
                  #f.write("train_results:{}\n".format({x: train_results[x] for x in train_results if x not in {"trainingepochs_preds_truth"}}))
                  f.write("test_results:{}\n".format({x: test_results[x] for x in mean_test_results if x not in {"preds","truths"}}))
                  f.write("[test preds]:\n{}\n".format(mean_test_results['preds']))
                  f.write("[test truths]:\n{}\n\n".format(mean_test_results['truths']))
                  #f.write("\n[training epochs preds and truths]:\n{}\n\n".format("\n".join("\n<{}>\n\n{}".format(k, "\n".join("\n[{}]]\n\n{}".format(k2, "\n".join("*{}:\n{}".format(k3, v3) for k3, v3 in v2.items()) ) for k2, v2 in v.items())) for k, v in train_results['trainingepochs_preds_truth'].items())))
                  f.write("*"*120+"\n")

                  #f.write("\n[training epochs preds and truths]:\n{}\n\n".format(train_results['trainingepochs_preds_truth']))

               # check how to make this simpler
            train_test_info = {
                                    "model_id": model_id,
                                    "Train_Loss": train_results["bestmodel_train_loss"],
                                    "Train_Accuracy": train_results["bestmodel_train_acc"], "Train_Precision":train_results["bestmodel_train_precision"],
                                    "Train_Recall": train_results["bestmodel_train_recall"], "Train_F1": train_results["bestmodel_train_F1"],
                                    "Mean_Test_Loss": mean_test_results["test_loss"],

                                    "Mean_Test_Accuracy": mean_test_results["test_acc"], 
                                    "Std_Test_Accuracy": np.std(k_results["test_acc"]),

                                    "Mean_Test_F1": mean_test_results["test_F1"], 
                                    "Std_Test_F1": np.std(k_results["test_F1"]),

                                    "Mean_Test_Precision": mean_test_results["test_precision"],
                                    "Mean_Test_Recall": mean_test_results["test_recall"] ,
                                    "K_Test_Loss": k_results["test_loss"],
                                    "K_Test_Accuracy": k_results["test_acc"],
                                    "K_Test_Precision": k_results["test_precision"],
                                    "K_Test_Recall": k_results["test_recall"],
                                    "K_Test_F1": k_results["test_F1"],
                                    # "Train_Time": training_time, "Test_Time": test_time
                                 }
            train_test_info.update(model_param_dict)
            train_test_info.update({"weight_init_seed": weight_init_seed,
                                    "split_shuffle_seed":split_shuffle_seed,
                                    "lr":lr})

            #   tunable_param.update(train_test_info)

            # save collected info to experiments results dataframe
            # experiment_results_df = experiment_results_df.append( train_test_info , ignore_index= True )

            experiment_results_df = pd.concat([experiment_results_df, pd.DataFrame([train_test_info])], axis=0)
            # write out csv file every iteration
            experiment_results_df.to_csv(path_or_buf=experiment_results_df_fpath)

    #--------------------------------------------------------------------------------------------------------------------------------------------------------
    if save_bestmodel_fitted_on_whole_data == True:
         """ try out different tunable parameter combinations """

         print(search_space,flush=True)

         for hyperparam_set in search_space:

            gc.collect()
            torch.cuda.empty_cache()


            weight_init_seed = hyperparam_set['weight_init_seed']
            split_shuffle_seed = hyperparam_set['split_shuffle_seed']
            lr = hyperparam_set['lr']
            

            model_param_dict = dict()
            model_param_dict["embedding_dim"] = hyperparam_set['embedding_dim']
            model_param_dict["ffn_dims"] = hyperparam_set['ffn_dims']
            model_param_dict["conv_activation_fn"] = hyperparam_set['conv_activation_fn']
            model_param_dict["activation_fn"] = hyperparam_set['activation_fn']

            model_param_dict["neighborhood_aggr"] = hyperparam_set['neighborhood_aggr']
            model_param_dict["pool"] = hyperparam_set['pool']
            model_param_dict["dropout_level"] = hyperparam_set['dropout_level']

            model_param_dict["signal_amplification_option"] = signal_amplification_option,


            # get this iteration's tunable_param
            X = train_dataset
            y = [data.y for data in train_dataset]

            counter = Counter([d.item() for d in y])

            print(f"Benign: #train:{counter[0]},\nMalware: #train:{counter[1]}")
            # set this iteration's model-id

            #Setting the seed for model
            # pl.seed_everything(tunable_param['seed'])


            torch.manual_seed( weight_init_seed )   
            if 'cuda' in device:
               torch.cuda.manual_seed_all( weight_init_seed )

            # instanitate classifer
            classifier = model_cls(
               num_classes=_num_classes, 
               dim_node= model_dim_node,
               edge_dim=model_dim_edge,
               **model_param_dict
            )



            # set dataloader_parms
            dataloader_params = {
                                 'batch_size' : batch_size,
                                 # 'data_split_ratio': data_split_ratio,  # train : 100% / eval: 0% / test: 0%  (test should be 0%, since we are just 'fitting' here.)
                                 'seed': split_shuffle_seed
                              }

            # create the trainer class
            trainer = TrainModel(
               model=classifier,
               # dataset=[X_train[0:num_0-1], X_train[-num_1:]],
               traindataset=X,
               testdataset = None,
               dataloader_params=dataloader_params,
               device=device,
               # **
               save_dir=best_model_savedirpath, # needs to save model for test;
               save_name= best_model_id, # it might be possible to overwrite to the save model instead of generating multiple
               
               verbose= 0, # Needs to be 0 here to prevent invoking SummaryWriter in trainer.py ; 
                        # verbose { 0: prints output as text ,  -1: does not print anything on screen }
            )

            # set train_params
            train_params = { 
                           'num_epochs': num_epochs, 
                           'num_early_stop': num_epochs,
                           'milestones': None,
                           'gamma': None
                           }

            # set optimizer_params
            optimizer_params = { 
                           'lr': lr, 
                           'weight_decay': 5e-8
                        }
         
            # train / record training time
            train_start = datetime.now()
            train_results = trainer.train( train_params=train_params, 
                                           optimizer_params=optimizer_params) # added return-statement for trainer.train() for this.
            training_time =str( datetime.now() - train_start )


            best_hyperparams = dict() # for hyperparams
            best_hyperparams['model_weight_init_seed'] = weight_init_seed  # no need for split_shuffle_seed as refit with whole-data.
            best_hyperparams["model__activation_fn"] = hyperparam_set['activation_fn']
            best_hyperparams["model__dim_hidden"] = hyperparam_set['dim_hidden']
            best_hyperparams["model__dropout"] = hyperparam_set['dropout_level']
            best_hyperparams["model__pool"] = hyperparam_set['pool']
            best_hyperparams["lr"] = hyperparam_set['lr']
            if "GAT" in model_cls_name:
               best_hyperparams["model__num_heads"] = hyperparam_set['num_heads']
         
            # JY @ 2023-06-27 : Don't save the best-model dirpath here!! 
            #                   It will not save the best model, but save the last model!!
            #                   Based on Dinal's ver of trainer file, it is saved during the training based on train-loss
            #                   , unlike my modified ver.
            #                   So, here, just save the hyperapmarameter json file here!
            #                   Jonly save hyperparams as json
            # bestmodel_saved_dirpath = os.path.join("saved_models", best_model_id)
            with open(os.path.join(best_model_savedirpath, f'{best_model_id}.json'), 'w') as fp:
               best_hyperparams['model__activation_fn'] = str(best_hyperparams['model__activation_fn'])
               best_hyperparams['model__pool'] = str(best_hyperparams['model__pool'])
               json.dump(best_hyperparams, fp)

            print(f"best model's weights and json file saved on {best_model_savedirpath}", flush=True)
            print(f"best model hyperparams:", flush=True)
            pprint.pprint(best_hyperparams)

      #   #################################################################################################################
      #   [Double-Stratification] Correctness-Check
      #   folds = []

      #   # https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html#sklearn.model_selection.StratifiedKFold

      #   #   for train_idx, test_idx in dataset_cv.split(X, y, groups= X_grouplist):
      #   for train_idx, test_idx in dataset_cv.split(X, y = X_grouplist):

      #       X_train, X_test = list(map(X.__getitem__, train_idx)), list(map(X.__getitem__, test_idx))
      #       y_train, y_test = np.array(y)[train_idx], np.array(y)[test_idx]

      #       folds.append({"X": X_test,"y": y_test}) # each fold corresponds to test-set here.

      #   fold_id = 0
      #   for fold in folds:
      #       print("="*100, flush=True)            
      #       print(f"Fold: {fold_id}", flush=True)
      #       # check if split is stratified in terms of 'y'
      #       print(f"label-distribution",flush=True)
      #       unique_elements, counts = np.unique(fold['y'], return_counts=True)
      #       for element, count in zip(unique_elements, counts):
      #          print(f"{element}: {count}", flush=True)
      #       print("-"*100, flush=True)
      #       # check if split is stratified in terms of 'y'
      #       # fold_X_names = [ x.name for x in fold['X'] ]
      #       fold_X_grouplist = []
      #       for data in fold['X']:
      #             # benign ------------------------------------------
      #             if "fleschutz" in data.name:
      #                fold_X_grouplist.append("benign_fleschutz")
      #             if "jhochwald" in data.name:
      #                fold_X_grouplist.append("benign_jhochwald")
      #             if "devblackops" in data.name:
      #                fold_X_grouplist.append("benign_devblackops")
      #             if "farag2" in data.name:
      #                fold_X_grouplist.append("benign_farag2")
      #             if "jimbrig" in data.name:
      #                fold_X_grouplist.append("benign_jimbrig")
      #             if "jrussellfreelance" in data.name:
      #                fold_X_grouplist.append("benign_jrussellfreelance")
      #             if "nickrod518" in data.name:
      #                fold_X_grouplist.append("benign_nickrod518")
      #             if "redttr" in data.name:
      #                fold_X_grouplist.append("benign_redttr")
      #             if "sysadmin-survival-kit" in data.name:
      #                fold_X_grouplist.append("benign_sysadmin-survival-kit")
      #             if "stevencohn" in data.name:
      #                fold_X_grouplist.append("benign_stevencohn")
      #             if "ledrago" in data.name:
      #                fold_X_grouplist.append("benign_ledrago")
      #             # malware ------------------------------------------
      #             if "empire" in data.name:
      #                fold_X_grouplist.append("malware_empire")
      #             if "invoke_obfuscation" in data.name:
      #                fold_X_grouplist.append("malware_invoke_obfuscation")
      #             if "nishang" in data.name:
      #                fold_X_grouplist.append("malware_nishang")
      #             if "poshc2" in data.name:
      #                fold_X_grouplist.append("malware_poshc2")
      #             if "mafia" in data.name:
      #                fold_X_grouplist.append("malware_mafia")
      #             if "offsec" in data.name:
      #                fold_X_grouplist.append("malware_offsec")
      #             if "powershellery" in data.name:
      #                fold_X_grouplist.append("malware_powershellery")
      #             if "psbits" in data.name:
      #                fold_X_grouplist.append("malware_psbits")
      #             if "pt-toolkit" in data.name:
      #                fold_X_grouplist.append("malware_pt-toolkit")
      #             if "randomps" in data.name:
      #                fold_X_grouplist.append("malware_randomps")
      #             if "smallposh" in data.name:
      #                fold_X_grouplist.append("malware_smallposh")
      #             if "bazaar" in data.name:
      #                fold_X_grouplist.append("malware_bazaar")
      #       unique_elements, counts = np.unique(fold_X_grouplist, return_counts=True)
      #       for element, count in zip(unique_elements, counts):
      #          print(f"{element}: {count}", flush=True)

      #       print("-"*100, flush=True)
      #       # check if there is no data overlap between folds
      #       print([x.name.lstrip("Processed_SUBGRAPH_P3_").rstrip(".pickle") for x in fold['X']], flush=True)

      #       fold_id+=1

      #   exit 
      #   #################################################################################################################