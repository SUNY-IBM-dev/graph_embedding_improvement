


'''

JY @ 2024-1-23: Might need to do this analysis directly from the elastic-search indices
                by adjusting tabby/SUNYIBM_ExplainableAI_2nd_Year_JY/Task_1__Behavior_identification_and_intention_learning/1_0__Identify_Behavioral_Events/group_log_entries_by_processThreads.py



'''


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
import matplotlib.dates as mdates

from itertools import cycle
from elasticsearch import Elasticsearch, helpers
from itertools import product
import pandas as pd
from datetime import datetime

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
def event_and_entity_level_artifactual_thread_investigation( dataset : list,
                                                             results_dirpath : str  ,

                                                             processThread_grouped_events = True,
                                                             processThread_grouped_events_EasyRead = True,
                                                             thread_lifetime_horizontal_dotplots__with_More_Detail = True ):
      # ------------------------------------------------------------
      # JY @ 2024-1-21: To save out .

      processThread_grouped_events_dirpath = os.path.join(results_dirpath, "processThread_grouped_events")

      processThread_grouped_events_EasyRead_dirpath = os.path.join(results_dirpath, "processThread_grouped_events_EasyRead")


      thread_lifetime_horizontal_dotplots__with_More_Detail__dirpath = os.path.join(results_dirpath, "thread_lifetime_horizontal_dotplots__with_More_Detail")


      if processThread_grouped_events:
         if not os.path.exists(processThread_grouped_events_dirpath):
            os.makedirs(processThread_grouped_events_dirpath)

            # results_dirpath 
            # --> processThread_grouped_events_dirpath
            #     ---> <subgraph-1 processThread_grouped_events_dirpath>.json
            #     ---> <subgraph-2 processThread_grouped_events_dirpath>.json
            #                 ....
            # 

      if processThread_grouped_events_EasyRead:
         if not os.path.exists(processThread_grouped_events_EasyRead_dirpath):
            os.makedirs(processThread_grouped_events_EasyRead_dirpath)

            # results_dirpath 
            # --> processThread_grouped_events_EasyRead_dirpath
            #     ---> <subgraph-1 processThread_grouped_events_dirpath>.txt
            #     ---> <subgraph-2 processThread_grouped_events_dirpath>.txt
            #                 ....
            


      if thread_lifetime_horizontal_dotplots__with_More_Detail:
         if not os.path.exists(thread_lifetime_horizontal_dotplots__with_More_Detail__dirpath):
            os.makedirs(thread_lifetime_horizontal_dotplots__with_More_Detail__dirpath)

            # results_dirpath 
            # --> thread_lifetime_horizontal_dotplots__with_More_Detail__dirpath
            #     ---> <subgraph-1 thread_lifetime_horizontal_dotplot>.png
            #     ---> <subgraph-2 thread_lifetime_horizontal_dotplot>.png
            #                 ....
            # 

      # ------------------------------------------------------------


      # 1. Iterate through each sample
      # 2.   read all log-entries of the sample from elastic-search 
      # 3.   get log-entries that belong to subgraph-root-process(powershell process) and its descendent processes.
      # 4.   for each process, group log-entries by threads, and save those out in a format good for analysis (json? )
      # 
      # Then, analyze whether there are artifactual threads (first 3?) with same 'event & entity sequence' across samples   

      skipped_indices = []

      for data in dataset:

            # ---------------------------------------------------------------------------------- 
            # No harm in getting label 
            if int(data.y) == 0:
                label = "benign"
            else:
                label = "malware"
            # 'data_name', which is index, is needed to read in log-entries from elastic-search           
            index = re.search(r'Processed_SUBGRAPH_P3_(.*?)\.pickle', data.name).group(1)
            # ----------------------------------------------------------------------------------          
            # Read all log-entries from elastic-search
            try:
               es = Elasticsearch(['http://ocelot.cs.binghamton.edu:9200'],timeout = 300)   
               es.indices.put_settings(index = index, body={'index':{'max_result_window':99999999}})
               result = es.search(index = index, size = 99999999)
               index__all_log_entries = result['hits']['hits']    # list of dicts

               subgraph_process_tree_pids = get_subgraph_process_tree_pids( es_index__all_log_entries = index__all_log_entries, 
                                                                            subgraph_root_pid = all_index_to_root_pid_mappings[index])

            except:
               skipped_indices.append(index)
               print(f"\n{len(skipped_indices)}:  {index}  is skipped as Elasticsearch doesn't contain it\n", flush = True)

            # Get all log-entries of subgraph-root and its descendents(if any)
            subgraph_process_tree_all_log_entries = get_log_entries_of_process_of_interest_and_descendents(index__all_log_entries,
                                                                                                           subgraph_process_tree_pids) 

            # Now group the caldera-technique-process and its descednents(if any)'s log-entries by process-Threads
            subgraph_process_tree_all_log_entries__with_EntityInfo = \
                              get_log_entries_with_entity_info( subgraph_process_tree_all_log_entries )

            processThread_to_logentries_dict = group_log_entries_by_processThreads( subgraph_process_tree_all_log_entries__with_EntityInfo )

            if thread_lifetime_horizontal_dotplots__with_More_Detail:
                processThread_to_logentries__with_events_order_information__dict = get__processThread_to_logentries__with_events_order_information( processThread_to_logentries_dict )

            # save it in an analysis friendly format (may be cross-check with threads-lifetime overlap check output?)

            if processThread_grouped_events :
               # save out a python dictionary (key : str, value :list ) to a json file, but dictionary to be sorted by key with smaller number lengths first, 
               # so that when I write out the json file and view it, I see the ones with smaller length first 
               for key, inner_dict in processThread_to_logentries_dict.items():
                  sorted_inner_keys = sorted(inner_dict, key=lambda k: len(inner_dict[k]))
                  processThread_to_logentries_dict[key] = {k: inner_dict[k] for k in sorted_inner_keys}
               processThread_to_logentries_dict__fpath = os.path.join( processThread_grouped_events_dirpath, f"{label}_{index}.json" )
               with open( processThread_to_logentries_dict__fpath , "w") as json_file:
                  json.dump(processThread_to_logentries_dict, json_file) 

            if processThread_grouped_events_EasyRead :
               # Save out EasyRead ver. of processThread_to_logentries_dict
               processThread_grouped_events_EasyRead__fpath = os.path.join( processThread_grouped_events_EasyRead_dirpath, f"{label}_{index}.txt" )
               with open( processThread_grouped_events_EasyRead__fpath , "w") as file:
                  
                  processes = list(processThread_to_logentries_dict.keys())

                  file.write(f"{len(processes)} processes : {processes} \n")
                  
                  for pid in processes:
                     file.write("==================================================\n")
                     file.write(f"{pid}  ({len(processThread_to_logentries_dict[pid])} threads) :\n")
                     file.write("-----------------------------------------------\n")
                     for tid in processThread_to_logentries_dict[pid]: # already sorted by list-length in ascending order.
                           file.write(f"     {pid}_{tid}  ({len(processThread_to_logentries_dict[pid][tid])} events) :\n")
                           for thread_log_entry_info in processThread_to_logentries_dict[pid][tid]:
                              file.write(f"       {thread_log_entry_info}\n")
                           file.write("-----------------------------------------------\n")
                  file.write("==================================================\n")

            if thread_lifetime_horizontal_dotplots__with_More_Detail :
                  
                  # Might need to work with normailzed ones ; might need to handle it in "group_log_entries_by_processThreads"


                  #processThread_to_logentries__with_events_order_information__dict


                  # data__threads__timestamps = [[tup[1] for tup in thread__taskname_timestamp__tuples] for thread__taskname_timestamp__tuples in data__threads__taskname_timestamp__tuples]




                  # Save out thread_lifetime_horizontal_dotplots_with_ThreadID
                  fig, ax = plt.subplots(figsize=(10, 6))
                  #plt.xlabel('Datetime')
                  color_cycle = cycle(['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'])
                  y_coord = 0
                  for_ylabels = []
                  for pid in processThread_to_logentries__with_events_order_information__dict.keys():

                     for tid in processThread_to_logentries__with_events_order_information__dict[pid]: # already sorted by list-length in ascending order.

                           # get all sorted timestamps for this thread
                           
                           thread_descriptor = f"{pid}__{tid}"
                           thread_sorted_event_timestamps = [ log_entry["normalized__event_order"] for log_entry in processThread_to_logentries__with_events_order_information__dict[pid][tid]]
                           thread_dots_color = next(color_cycle)
                           # Plot dots
                           ax.scatter( x = thread_sorted_event_timestamps ,  y = [y_coord] * len(thread_sorted_event_timestamps), 
                                          c = thread_dots_color,
                                       label= thread_descriptor, 
                                       s=1)
                           print(thread_descriptor)

                           y_coord += 1 # for horizontal 
                           for_ylabels.append(thread_descriptor) # confirmed to be correct with 'ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')'
                  ax.set_xlabel('Norm. TimeStamp')

                     # # Set y-axis ticks and labels
                  ax.set_yticks(range(len(for_ylabels)))
                  ax.set_yticklabels(for_ylabels)

                     # # Set x-axis label

                  #Add legend
                  # ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

                  plt.tight_layout()

                           # Show the plot
                  plt.savefig(os.path.join(thread_lifetime_horizontal_dotplots__with_More_Detail__dirpath, f"{label}_{index}.png")) # this does save out





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

    running_from_machine = "panther"

    # ---------------------------------------------------
    if running_from_machine == "ocelot":
      abs_path_to_tabby = "/data/d1/jgwak1/tabby"
    else: # ocelot
      abs_path_to_tabby = "/home/jgwak1/tabby"



    # -----------------------------------------------------------------------------------------------------------------------------------------
    # Added by JY @ 2024-1-23:
    sys.path.append(f"{abs_path_to_tabby}/graph_embedding_improvement_JY_git/analyze_at_data_level/Event_and_Entity_Level_Artifactual_Thread_Check")
    from case1_case2_data_processID_mapping import *
    all_index_to_root_pid_mappings = case1_benign_logs | case2_benign_logs | case1_malware_logs | case2_malware_logs
    from investigation_helper_functions import *

    sys.path.append(f"{abs_path_to_tabby}/graph_embedding_improvement_JY_git/analyze_at_data_level/Event_and_Entity_Level_Artifactual_Thread_Check/source")
    from source.dataprocessor_graphs import LoadGraphs
    from source.model import GIN
    from source.trainer_meng_ver import TrainModel


   # ==================================================================================================================================

    # cmd args

    dataset_choice = parser.parse_args().dataset[0]


    ###############################################################################################################################################
    # Set data paths
    projection_datapath_Benign_Train_dict = {
      # Dataset-1 (B#288, M#248) ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
      #PW: Dataset-Case-1 
      "Dataset-Case-1": \
         {"5": f"{abs_path_to_tabby}/SILKETW_DATASET_NEW/Silketw_benign_train_test_data_case1/offline_train/Processed_Benign_ONLY_TaskName_edgeattr", # dim-node == 5
         "35": f"{abs_path_to_tabby}/SILKETW_DATASET_NEW/Benign_case1/train"}, # dim-node == 35 (adhoc)

      # ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
      # Dataset-2 (B#662, M#628)
      "Dataset-Case-2": \
        {"5": f"{abs_path_to_tabby}/SILKETW_DATASET_NEW/Silketw_benign_train_test_data_case1_case2/offline_train/Processed_Benign_ONLY_TaskName_edgeattr",
         "35": f"{abs_path_to_tabby}/SILKETW_DATASET_NEW/Benign_case2/train"},

    }
    projection_datapath_Malware_Train_dict = {
      # Dataset-1 (B#288, M#248) ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
      "Dataset-Case-1": \
      {"5":f"{abs_path_to_tabby}/SILKETW_DATASET_NEW/Silketw_malware_train_test_data_case1/offline_train/Processed_Malware_ONLY_TaskName_edgeattr",
       "35":f"{abs_path_to_tabby}/SILKETW_DATASET_NEW/Malware_case1/train"},

      # ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
      # Dataset-2 (B#662, M#628)
      "Dataset-Case-2": \
      {"5": f"{abs_path_to_tabby}/SILKETW_DATASET_NEW/Silketw_malware_train_test_data_case1_case2/offline_train/Processed_Malware_ONLY_TaskName_edgeattr",
       "35": f"{abs_path_to_tabby}/SILKETW_DATASET_NEW/Malware_case2/train"},


    }
    projection_datapath_Benign_Test_dict = {
      # Dataset-1 (B#73, M#62) ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
      "Dataset-Case-1": \
      {"5": f"{abs_path_to_tabby}/SILKETW_DATASET_NEW/Silketw_benign_train_test_data_case1/offline_test/Processed_Benign_ONLY_TaskName_edgeattr",
       "35": f"{abs_path_to_tabby}/SILKETW_DATASET_NEW/Benign_case1/test"},

      # ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
      # Dataset-2 (B#167, M#158)
      "Dataset-Case-2": \
         {"5": f"{abs_path_to_tabby}/SILKETW_DATASET_NEW/Silketw_benign_train_test_data_case1_case2/offline_test/Processed_Benign_ONLY_TaskName_edgeattr",
          "35": f"{abs_path_to_tabby}/SILKETW_DATASET_NEW/Benign_case2/test"},

    }
    projection_datapath_Malware_Test_dict = {
      # Dataset-1 (B#73, M#62) ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
      "Dataset-Case-1": \
         {"5": f"{abs_path_to_tabby}/SILKETW_DATASET_NEW/Silketw_malware_train_test_data_case1/offline_test/Processed_Malware_ONLY_TaskName_edgeattr",
          "35": f"{abs_path_to_tabby}/SILKETW_DATASET_NEW/Malware_case1/test"},

      # ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
      # Dataset-2 (B#167, M#158)
      "Dataset-Case-2": \
         {"5": f"{abs_path_to_tabby}/SILKETW_DATASET_NEW/Silketw_malware_train_test_data_case1_case2/offline_test/Processed_Malware_ONLY_TaskName_edgeattr",
          "35": f"{abs_path_to_tabby}/SILKETW_DATASET_NEW/Malware_case2/test"},

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
    run_identifier = f"Event_and_Entity_Level_Artifactual_Thread_Check__{dataset_choice}__{datetime.now().strftime('%Y-%m-%d_%H%M%S')}"
    this_results_dirpath = f"{abs_path_to_tabby}/graph_embedding_improvement_JY_git/analyze_at_data_level/Event_and_Entity_Level_Artifactual_Thread_Check/RESULTS/{run_identifier}"
    if not os.path.exists(this_results_dirpath):
      os.makedirs(this_results_dirpath)

    event_and_entity_level_artifactual_thread_investigation( dataset= entire_dataset, 
                                                             results_dirpath = this_results_dirpath,
                                                             processThread_grouped_events = True,
                                                             processThread_grouped_events_EasyRead = True,
                                                             thread_lifetime_horizontal_dotplots__with_More_Detail = False)
