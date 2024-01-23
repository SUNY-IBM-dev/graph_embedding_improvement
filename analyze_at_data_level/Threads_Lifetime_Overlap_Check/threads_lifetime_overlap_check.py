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

sys.path.append("/data/d1/jgwak1/tabby/graph_embedding_improvement_JY_git/analyze_at_data_level/Threads_Lifetime_Overlap_Check/source")

from source.dataprocessor_graphs import LoadGraphs
from source.model import GIN
from source.trainer_meng_ver import TrainModel

from itertools import product
import pandas as pd
from datetime import datetime

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold


#GNN-TRAINING
from source.dataprocessor_graphs import LoadGraphs
#from source.model import GAT, GIN, GIN_no_edgefeat, GIN_no_edgefeat_simplified, GCN, GAT_He, GAT_mlp_fed_1gram 
#from source.model import GNNBasic, LocalMeanPool,  SumPool, GlobalMeanPool

#from source.gnn_dynamicgraph import GNN_DynamicGraph__ImplScheme_1

# from source.trainer import TrainModel, get_dataloader

#ETC
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader #PW nood to install torch_geometric
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

# JY @ 2024-1-21 -- to save
import matplotlib.pyplot as plt 


# TODO -- make sure explanations are produced by this
EventID_to_RegEventName_dict =\
{
"EventID(1)":"CreateKey", 
"EventID(2)":"OpenKey",
"EventID(3)":"DeleteKey", 
"EventID(4)":"QueryKey", 
"EventID(5)":"SetValueKey", 
"EventID(6)":"DeleteValueKey", 
"EventID(7)":"QueryValueKey",  
"EventID(8)":"EnumerateKey", 
"EventID(9)":"EnumerateValueKey", 
"EventID(10)":"QueryMultipleValueKey",
"EventID(11)":"SetinformationKey", 
"EventID(13)":"CloseKey", 
"EventID(14)":"QuerySecurityKey",
"EventID(15)":"SetSecurityKey", 
"Thisgroupofeventstrackstheperformanceofflushinghives": "RegPerfOpHiveFlushWroteLogFile",
}

##############################################################################################################################

#**********************************************************************************************************************************************************************


taskname_colnames = [
    'None_or_empty', #0 (index 0) 
    'Cleanup', #1
    'Close', #2
    'Create', #3
    'CreateNewFile', #4
    'DeletePath',#5
    'DirEnum',#6
    'DirNotify',#7
    'Flush',#8
    'FSCTL',#9
    'NameCreate',#10
    'NameDelete',#11
    'OperationEnd',#12
    #'QueINFO',#13
    'QueryInformation',#13
    'QueryEA',#14
    'QuerySecurity',#15
    'Read',#16
    'Write',#17
    'SetDelete',#18
    'SetInformation', #19
    'PagePriorityChange',#20
    'IoPriorityChange',#21
    'CpuBasePriorityChange',#22
    #'IMAGEPriorityChange',#24
    'CpuPriorityChange',#23
    'ImageLoad',#24
    'ImageUnload',#25
    'ProcessStop/Stop',#26
    'ProcessStart/Start',#27
    'ProcessFreeze/Start',#28--------
    #'PSDISKIOATTRIBUTE',#31
    #'PSIORATECONTROL',#32 
    'ThreadStart/Start',#29
    'ThreadStop/Stop',#30
    'ThreadWorkOnBehalfUpdate', #31
    'JobStart/Start',#32--------
    'JobTerminate/Stop',#33--------
    #'LOSTEVENT',#38
    #'PSDISKIOATTRIBUTION',#39
    'Rename',#34
    'Renamepath',#35
    'RegPerfOpHiveFlushWroteLogFile',#36------- "Thisgroupofeventstrackstheperformanceofflushinghives": "RegPerfOpHiveFlushWroteLogFile",
    'CreateKey',#37 "EventID(1)":"CreateKey", 
    'OpenKey',#38 "EventID(2)":"OpenKey",
    'DeleteKey',#39 "EventID(3)":"DeleteKey", 
    'QueryKey',#40 "EventID(4)":"QueryKey", 
    'SetValueKey',#41 "EventID(5)":"SetValueKey", 
    'DeleteValueKey',#42 "EventID(6)":"DeleteValueKey", 
    'QueryValueKey',#43 "EventID(7)":"QueryValueKey",  
    'EnumerateKey',#44 "EventID(8)":"EnumerateKey",  
    'EnumerateValueKey',#45 "EventID(9)":"EnumerateValueKey", 
    'QueryMultipleValueKey',#46 "EventID(10)":"QueryMultipleValueKey",
    'SetinformationKey',#47 "EventID(11)":"SetinformationKey", 
    'CloseKeys',#48 "EventID(13)":"CloseKey", 
    'QuerySecurityKey',#49 "EventID(14)":"QuerySecurityKey",
    'SetSecurityKey',#50 "EventID(15)":"SetSecurityKey", 
    'KERNEL_NETWORK_TASK_TCPIP/Datasent.', #51
    'KERNEL_NETWORK_TASK_TCPIP/Datareceived.',#52
    'KERNEL_NETWORK_TASK_TCPIP/Connectionattempted.',#53
    'KERNEL_NETWORK_TASK_TCPIP/Disconnectissued.', #54
    'KERNEL_NETWORK_TASK_TCPIP/Dataretransmitted.',#55
    'KERNEL_NETWORK_TASK_TCPIP/connectionaccepted.' , #56----
    'KERNEL_NETWORK_TASK_TCPIP/Protocolcopieddataonbehalfofuser.', #57
    'KERNEL_NETWORK_TASK_UDPIP/DatareceivedoverUDPprotocol.',#58
    'KERNEL_NETWORK_TASK_UDPIP/DatasentoverUDPprotocol.', #59
    # 'NULL', #remove this entry once second step updations used for subgra
    'Unseen', # 60
    #PW: below events are now different in silketw
    # all below 3 task are combined with opcode and having index 43 onwards for all of them in the function TN2int()
    # 'KERNEL_NETWORK_TASK_UDPIP'#index 43 # 42(opcode value) 43,49(https://github.com/repnz/etw-providers-docs/blob/master/Manifests-Win7-7600/Microsoft-Windows-Kernel-Network.xml)
    # 'KERNEL_NETWORK_TASK_TCPIP', # 10-18 (https://github.com/repnz/etw-providers-docs/blob/master/Manifests-Win7-7600/Microsoft-Windows-Kernel-Network.xml)
    # 'MICROSOFT-WINDOWS-KERNEL-REGISTRY', # 32- 46 (https://github.com/repnz/etw-providers-docs/blob/master/Manifests-Win7-7600/Microsoft-Windows-Kernel-Registry.xml)

]



##########################################################################################################################################################

# JY @ 2024-1-20: thread-level N>1 gram events + nodetype-5bits

def threads_lifetime_overlap_check( dataset : list,
                                    results_dirpath : str  ):
      
      # ------------------------------------------------------------
      # JY @ 2024-1-21: To save out .

      thread_lifetime_horizontal_dotplots_dirpath = os.path.join(results_dirpath, "thread_lifetime_horizontal_dotplots")
      thread_taskname_timestamp_tuple_lists_dirpath = os.path.join(results_dirpath, "thread_taskname_timestamp_tuple_lists")

      if not os.path.exists(thread_lifetime_horizontal_dotplots_dirpath):
         os.makedirs(thread_lifetime_horizontal_dotplots_dirpath)

         # results_dirpath 
         # --> thread_lifetime_horizontal_dotplots_dirpath
         #     ---> <subgraph-1 thread_lifetime_horizontal_dotplot>.png
         #     ---> <subgraph-2 thread_lifetime_horizontal_dotplot>.png
         #                 ....
         # 


      if not os.path.exists(thread_taskname_timestamp_tuple_lists_dirpath):
         os.makedirs(thread_taskname_timestamp_tuple_lists_dirpath)

         # results_dirpath 
         # --> thread_taskname_timestamp_tuple_lists_dirpath
         #     ---> <subgraph-1 thread_taskname_timestamp_tuple_lists>.txt
         #     ---> <subgraph-2 thread_taskname_timestamp_tuple_lists>.txt
         #                 ....
         # 


      # ------------------------------------------------------------

      thread_node_tensor = torch.tensor([0, 0, 0, 0, 1])

      data_dict = dict()

      cnt = 1
      for data in dataset:

            # ----------------------------------------------------------------------------------            
            if int(data.y) == 0:
                label = "benign"
            else:
                label = "malware"

            data_name = re.search(r'Processed_SUBGRAPH_P3_(.*?)\.pickle', data.name).group(1)
            # ---------------------------------------------------------------------------------- 
            
            print(f"{cnt} / {len(dataset)}: {data.name} -- generate graph-embedding", flush = True)

            # Added by JY @ 2023-07-18 to handle node-attr dim > 5  
            # if data.x.shape[1] != 5:
            data_x_first5 = data.x[:,:5]
            thread_node_indices = torch.nonzero(torch.all(torch.eq( data_x_first5, thread_node_tensor), dim=1), as_tuple=False).flatten().tolist()

            # which this node is a source-node (outgoing-edge w.r.t this node)

            data__threads__taskname_timestamp__tuples = []

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



               ''' JY @ 2024-1-20: Get thread-level event-sequence sorted by timestamps '''
               # Refered to : https://github.com/SUNY-IBM-dev/graph_embedding_improvement/blob/20627016d59466d3dad191ff208efce97b15d35e/graph_embedding_improvement_efforts/Trial_7__Thread_level_N_grams__N_gt_than_1__Similar_to_PriorGraphEmbedding/thread_level_n_gram__n_gt_1__similar_to_asiaccs_graph_embedding.py#L483C1-L491C95

               edge_attr_of_both_direction_edges_from_thread_node_idx = torch.cat([edge_attr_of_incoming_edges_from_thread_node_idx, 
                                                                                   edge_attr_of_outgoing_edges_from_thread_node_idx],
                                                                                   dim = 0)


               timestamp_sorted_indices = torch.argsort(edge_attr_of_both_direction_edges_from_thread_node_idx[:, -1], descending=False)

               edge_attr_of_both_direction_edges_from_thread_node_idx__sorted = edge_attr_of_both_direction_edges_from_thread_node_idx[ timestamp_sorted_indices ]

               taskname_indices = torch.nonzero(edge_attr_of_both_direction_edges_from_thread_node_idx__sorted[:,:-1], as_tuple=False)[:, -1]

               thread_sorted_event_list = [taskname_colnames[i] for i in taskname_indices]
   
               thread__taskname_timestamp__tuple = list(zip(thread_sorted_event_list, edge_attr_of_both_direction_edges_from_thread_node_idx__sorted[:,-1].tolist()))



               data__threads__taskname_timestamp__tuples.append( thread__taskname_timestamp__tuple )

            # -------------------------------------------------------------------
            # 1. save the list "thread__taskname_timestamp__tuple"

            data_thread_taskname_timestamp_tuple_lists_fpath = os.path.join( thread_taskname_timestamp_tuple_lists_dirpath, f"{label}_{data_name}.txt" )

            # Sorting by the length of each sublist in ascending order for easier reading of output
            data__threads__taskname_timestamp__tuples = sorted(data__threads__taskname_timestamp__tuples, key=len)
            data__threads__taskname_timestamp__tuples__strings = [ str(x) for x in data__threads__taskname_timestamp__tuples ]

            with open(data_thread_taskname_timestamp_tuple_lists_fpath, 'w') as f:
               #json.dump(data__threads__taskname_timestamp__tuples__strings, jsonf, indent=1)
               for tuple_string in data__threads__taskname_timestamp__tuples__strings:
                  f.write(f"{tuple_string}\n\n")
            # -------------------------------------------------------------------
            # 2. save duration-horizontal bar-plots to a dir
            data__threads__timestamps = [[tup[1] for tup in thread__taskname_timestamp__tuples] for thread__taskname_timestamp__tuples in data__threads__taskname_timestamp__tuples]


            # Create a scatter plot
            fig, ax = plt.subplots(figsize=(10, 6))

            # Plot dots
            for i, data__thread__timestamp in enumerate(data__threads__timestamps):
               ax.scatter(data__thread__timestamp, 
                          [i] * len(data__thread__timestamp), 
                          label=f'Thread {i+1}', 
                          s=1)

            # Set y-axis ticks and labels
            ax.set_yticks(range(len(data__threads__timestamps)))
            ax.set_yticklabels([f'Thread {i+1}' for i in range(len(data__threads__timestamps))])

            # Set x-axis label
            ax.set_xlabel('Norm. Timestamps')

            # Add legend
            # ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

            plt.tight_layout()

            # Show the plot
            plt.savefig(os.path.join(thread_lifetime_horizontal_dotplots_dirpath, f"{label}_{data_name}.png"))




            # -------------------------------------------------------------------
            cnt+=1
      return 
##########################################################################################################################################################
##########################################################################################################################################################


#**********************************************************************************************************************************************************************

#**********************************************************************************************************************************************************************
#**********************************************************************************************************************************************************************
#**********************************************************************************************************************************************************************



##############################################################################################################################


# JY @ 2023-07-25: Here, not only tune using K-fold-CV, but also fit the best hyperparam on entire-train and test on final-test

if __name__ == '__main__':

    parser = ArgumentParser()

    parser.add_argument('-data', '--dataset', 
                        choices= ['Dataset-Case-1',
                                  'Dataset-Case-2' # try
                                  ], 
                        default = ['Dataset-Case-2'])




   # ==================================================================================================================================

    # cmd args

    dataset_choice = parser.parse_args().dataset[0]


    ###############################################################################################################################################
    # Set data paths
    projection_datapath_Benign_Train_dict = {
      # Dataset-1 (B#288, M#248) ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
      #PW: Dataset-Case-1 
      "Dataset-Case-1": \
         {"5": "/data/d1/jgwak1/tabby/SILKETW_DATASET_NEW/Silketw_benign_train_test_data_case1/offline_train/Processed_Benign_ONLY_TaskName_edgeattr", # dim-node == 5
         "35": "/data/d1/jgwak1/tabby/SILKETW_DATASET_NEW/Benign_case1/train"}, # dim-node == 35 (adhoc)

      # ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
      # Dataset-2 (B#662, M#628)
      "Dataset-Case-2": \
        {"5": "/data/d1/jgwak1/tabby/SILKETW_DATASET_NEW/Silketw_benign_train_test_data_case1_case2/offline_train/Processed_Benign_ONLY_TaskName_edgeattr",
         "35": "/data/d1/jgwak1/tabby/SILKETW_DATASET_NEW/Benign_case2/train"},

    }
    projection_datapath_Malware_Train_dict = {
      # Dataset-1 (B#288, M#248) ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
      "Dataset-Case-1": \
      {"5":"/data/d1/jgwak1/tabby/SILKETW_DATASET_NEW/Silketw_malware_train_test_data_case1/offline_train/Processed_Malware_ONLY_TaskName_edgeattr",
       "35":"/data/d1/jgwak1/tabby/SILKETW_DATASET_NEW/Malware_case1/train"},

      # ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
      # Dataset-2 (B#662, M#628)
      "Dataset-Case-2": \
      {"5": "/data/d1/jgwak1/tabby/SILKETW_DATASET_NEW/Silketw_malware_train_test_data_case1_case2/offline_train/Processed_Malware_ONLY_TaskName_edgeattr",
       "35": "/data/d1/jgwak1/tabby/SILKETW_DATASET_NEW/Malware_case2/train"},


    }
    projection_datapath_Benign_Test_dict = {
      # Dataset-1 (B#73, M#62) ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
      "Dataset-Case-1": \
      {"5": "/data/d1/jgwak1/tabby/SILKETW_DATASET_NEW/Silketw_benign_train_test_data_case1/offline_test/Processed_Benign_ONLY_TaskName_edgeattr",
       "35": "/data/d1/jgwak1/tabby/SILKETW_DATASET_NEW/Benign_case1/test"},

      # ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
      # Dataset-2 (B#167, M#158)
      "Dataset-Case-2": \
         {"5": "/data/d1/jgwak1/tabby/SILKETW_DATASET_NEW/Silketw_benign_train_test_data_case1_case2/offline_test/Processed_Benign_ONLY_TaskName_edgeattr",
          "35": "/data/d1/jgwak1/tabby/SILKETW_DATASET_NEW/Benign_case2/test"},

    }
    projection_datapath_Malware_Test_dict = {
      # Dataset-1 (B#73, M#62) ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
      "Dataset-Case-1": \
         {"5": "/data/d1/jgwak1/tabby/SILKETW_DATASET_NEW/Silketw_malware_train_test_data_case1/offline_test/Processed_Malware_ONLY_TaskName_edgeattr",
          "35": "/data/d1/jgwak1/tabby/SILKETW_DATASET_NEW/Malware_case1/test"},

      # ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
      # Dataset-2 (B#167, M#158)
      "Dataset-Case-2": \
         {"5":"/data/d1/jgwak1/tabby/SILKETW_DATASET_NEW/Silketw_malware_train_test_data_case1_case2/offline_test/Processed_Malware_ONLY_TaskName_edgeattr",
          "35": "/data/d1/jgwak1/tabby/SILKETW_DATASET_NEW/Malware_case2/test"},

    }

    _num_classes = 2  # number of class labels and always binary classification.

    _dim_node = 5 #46   # num node features ; the #feats  # -- JY @ 2024-1-21: Doesn't matter whether it's 5 or 35, for this file.


    #PW: 5 and 62 (61 taskname + 1 timestamp) based on new silketw
    _dim_edge = 62 #72    # (or edge_dim) ; num edge features

    _benign_train_data_path = projection_datapath_Benign_Train_dict[dataset_choice][str(_dim_node)]
    _malware_train_data_path = projection_datapath_Malware_Train_dict[dataset_choice][str(_dim_node)]
    _benign_final_test_data_path = projection_datapath_Benign_Test_dict[dataset_choice][str(_dim_node)]
    _malware_final_test_data_path = projection_datapath_Malware_Test_dict[dataset_choice][str(_dim_node)]


    ####################################################################################################################################################
    
   

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


    entire_dataset = train_dataset + final_test_dataset


    # =================================================================================================================================================
    run_identifier = f"Threads_Lifetime_Overlap_Check__{dataset_choice}__{datetime.now().strftime('%Y-%m-%d_%H%M%S')}"
    this_results_dirpath = f"/data/d1/jgwak1/tabby/graph_embedding_improvement_JY_git/analyze_at_data_level/Threads_Lifetime_Overlap_Check/RESULTS/{run_identifier}"
    if not os.path.exists(this_results_dirpath):
      os.makedirs(this_results_dirpath)

    threads_lifetime_overlap_check( dataset= entire_dataset,
                                    results_dirpath = this_results_dirpath)
