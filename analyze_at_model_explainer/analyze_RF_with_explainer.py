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

sys.path.append("/data/d1/jgwak1/tabby/graph_embedding_improvement_JY_git/analyze_at_model_explainer/source")

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
# from grpc import stream_unary_rpc_method_handler
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


#############################################################################################################################
# libraries for tradition-ML (2023-07-25)

import sklearn
from sklearn import metrics
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier # XGB

# https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMClassifier.html#lightgbm.LGBMClassifier
#from lightgbm import LGBMClassifier # pip install lightgbm -- added by JY @ 2023-08-10

# for baseline
from sklearn.feature_extraction.text import CountVectorizer

##############################################################################################################################
# Explainer related (2023-12-20)

import shap
import matplotlib.pyplot as plt    


# TODO -- make sure explanations are produced by this
EventID_to_RegEventName_dict =\
{
"EventID(1)":"Createkey", 
"EventID(2)":"Openkey",
"EventID(3)":"Deletekey", 
"EventID(4)":"Querykey", 
"EventID(5)":"SetValueKey", 
"EventID(6)":"DeleteValueKey", 
"EventID(7)":"QueryValueKey",  
"EventID(8)":"EnumerateKey", 
"EventID(9)":"EnumerateValuekey", 
"EventID(10)":"QueryMultipleValuekey",
"EventID(11)":"Setinformationkey", 
"EventID(13)":"Closekey", 
"EventID(14)":"QuerySecuritykey",
"EventID(15)":"SetSecuritykey", 
"Thisgroupofeventstrackstheperformanceofflushinghives": "RegPerfOpHiveFlushWroteLogFile",
}


def produce_SHAP_explanations(classification_model, 
                              Test_dataset : pd.DataFrame,
                              Train_dataset : pd.DataFrame,
                              Explanation_Results_save_dirpath : str, 
                              model_cls_name : str, 
                              Test_SG_names : list,
                              misprediction_subgraph_names : list,
                              N : int = 1,
                              ):

      # JY @ 2023-12-20: Integrate SHAP into this file based on:
      #                  /data/d1/jgwak1/tabby/BASELINE_COMPARISONS/Sequential/RF+Ngrams/RFSVM_1gram_events_flattened_subgraph_only_psh.py
      #                  /data/d1/jgwak1/tabby/CXAI_2023_Experiments/Run_Explainers/SHAP_LIME__Ngram.py (*)   

      # =============================================================================================================================
      # SHAP-Global 
      if N == 2: check_additivity = False
      else: check_additivity = True # default

      # https://shap-lrjball.readthedocs.io/en/latest/generated/shap.TreeExplainer.html
      shap_explainer = shap.TreeExplainer(classification_model)
 
      shap_values = shap_explainer.shap_values( np.array(Test_dataset.drop(columns=['data_name']).values), 
                                                check_additivity=check_additivity) 
      f = plt.figure()

      if "RandomForestClassifier" in model_cls_name:
            shap.summary_plot(shap_values = shap_values[1], # class-1 (positive class ; malware)
                              features = Test_dataset.drop(columns=['data_name']).values, 
                              feature_names = list(Test_dataset.drop(columns=['data_name']).columns),      # pipeline[0] == our countvectorizer (n-gram)
                              plot_type = "bar")
            model_resultX = pd.DataFrame(shap_values[1], columns = list(Test_dataset.drop(columns=['data_name']).columns))

      elif "GradientBoostingClassifier" in model_cls_name:
            shap.summary_plot(shap_values = shap_values, # class-? (positive class ; ?)
                           features = Test_dataset.drop(columns=['data_name']).values, 
                           feature_names = list(Test_dataset.drop(columns=['data_name']).columns),      # pipeline[0] == our countvectorizer (n-gram)
                           plot_type = "bar")                
            model_resultX = pd.DataFrame(shap_values, columns = list(Test_dataset.drop(columns=['data_name']).columns))

      f.savefig( os.path.join(Explanation_Results_save_dirpath, 
                              f"{N}_gram_SHAP_Global_Interpretability_Summary_BarPlot_Push_Towards_Malware_Features.png"), 
                              bbox_inches='tight', dpi=600)


      # TODO: Important: Get a feature-importance from shap-values
      # https://stackoverflow.com/questions/65534163/get-a-feature-importance-from-shap-values

      vals = np.abs(model_resultX.values).mean(0)
      shap_importance = pd.DataFrame(list(zip(list(Test_dataset.drop(columns=['data_name']).columns), vals)),
                                     columns=['col_name','feature_importance_vals']) # Later could make use of the "feature_importance_vals" if needed.
      shap_importance.sort_values(by=['feature_importance_vals'], ascending=False, inplace=True)
      shap_importance.to_csv(os.path.join(Explanation_Results_save_dirpath, f"{model_cls_name} {N}-gram Global-SHAP Importance.csv"))

      Global_Important_featureIndex_featureName = dict(zip(shap_importance.reset_index()['index'], shap_importance.reset_index()['col_name']))
      Global_Important_featureNames = [ v for k,v in Global_Important_featureIndex_featureName.items() ]
      # Save Global-First20Important Features "Train dataset"
      Global_Important_Features_Train_dataset = Train_dataset[ Global_Important_featureNames ]
      Global_Important_Features_Train_dataset = Global_Important_Features_Train_dataset.assign(SUM = Global_Important_Features_Train_dataset.sum(axis=1)) 
      Global_Important_Features_Train_dataset.to_csv(os.path.join(Explanation_Results_save_dirpath, f"{model_cls_name} {N}-gram Global-SHAP Important FeatureNames Train-Dataset.csv"))

      # Save Global-First20Important Features "Test dataset"
      Global_Important_Features_Test_dataset = Test_dataset[ Global_Important_featureNames ] #  여기가 문제인듯
      Global_Important_Features_Test_dataset = Global_Important_Features_Test_dataset.assign(SUM = Global_Important_Features_Test_dataset.sum(axis=1)) 
      # Global_Important_Features_Test_dataset["predict_proba"] = pd.Series(Predict_proba_dict)
      Global_Important_Features_Test_dataset.to_csv(os.path.join(Explanation_Results_save_dirpath, f"{model_cls_name} {N}-gram Global-SHAP Important FeatureNames Test-Dataset.csv"))

      # # Save Global-First20Important Features "ALL dataset" (After Integrating Train and Test)
      # Global_Important_Features_All_dataset = pd.concat( [ Global_Important_Features_Train_dataset, Global_Important_Features_Test_dataset ] , axis = 0 )
      # Global_Important_Features_All_dataset.sort_values(by="subgraph", inplace= True)
      # Global_Important_Features_All_dataset.to_csv(os.path.join(Explanation_Results_save_dirpath, f"{model_cls_name} {N}-gram Global-SHAP Important FeatureNames ALL-Dataset.csv"))



      # =============================================================================================================================
      # SHAP-Local (Waterfall-Plots)

      WATERFALL_PLOTS_Local_Explanation_dirpath = os.path.join(Explanation_Results_save_dirpath, f"WATERFALL_PLOTS_Local-Explanation_{N}gram")
      Correct_Predictions_WaterfallPlots_subdirpath = os.path.join(WATERFALL_PLOTS_Local_Explanation_dirpath, "Correct_Predictions")
      Mispredictions_WaterfallPlots_subdirpath = os.path.join(WATERFALL_PLOTS_Local_Explanation_dirpath, "Mispredictions")

      if not os.path.exists( WATERFALL_PLOTS_Local_Explanation_dirpath ): os.makedirs(WATERFALL_PLOTS_Local_Explanation_dirpath)
      if not os.path.exists( Correct_Predictions_WaterfallPlots_subdirpath ): os.makedirs( Correct_Predictions_WaterfallPlots_subdirpath )
      if not os.path.exists( Mispredictions_WaterfallPlots_subdirpath ): os.makedirs( Mispredictions_WaterfallPlots_subdirpath )


      # Iterate through all tested-subgraphs
      Test_dataset.set_index('data_name', inplace= True) # added by JY @ 2023-12-20

      for Test_SG_name in Test_SG_names:

            shap_explainer = shap.TreeExplainer(classification_model)

            shap_values = shap_explainer.shap_values(X = Test_dataset.loc[Test_SG_name].values, 
                                                     check_additivity=check_additivity)

            # shap_values = shap_explainer(Test_dataset.loc[Test_SG_name])


            # https://stackoverflow.com/questions/71751251/get-waterfall-plot-values-of-a-feature-in-a-dataframe-using-shap-package
            # shap_values = shap_explainer(Test_dataset)
            # exp = shap.Explanation(shap_values.values[:,:,1], 
            #                         shap_values.base_values[:,1], 
            #                    data=Test_dataset.values, 
            #                 feature_names=Test_dataset.columns)            

            if "RandomForestClassifier" in model_cls_name:
               shap_values = shap_values[1]
               base_value = shap_explainer.expected_value[1]
            
            elif "GradientBoostingClassifier" in model_cls_name:
               shap_values = shap_values 
               base_value = shap_explainer.expected_value # this is possibility in terms of benign


            # https://shap.readthedocs.io/en/latest/generated/shap.Explanation.html
            exp = shap.Explanation(values = shap_values, 
                                   base_values = base_value, 
                                   data=Test_dataset.loc[Test_SG_name].values, 
                                   feature_names=Test_dataset.columns)    
            plt.close()
            # plt.xticks(fontsize=8)  # Adjust the fontsize to fit within the bars
            # plt.rcParams['figure.constrained_layout.use'] = True
            # plt.figure(figsize=(300, 300), dpi=80)
            # plt.figure(layout="constrained")
            # plt.figure(layout="tight")
            waterfallplot_out = shap.plots.waterfall(exp, max_display=20, show=False) # go inside here # https://github.com/shap/shap/issues/3213
            plt.tight_layout()

            # https://shap.readthedocs.io/en/latest/example_notebooks/api_examples/plots/waterfall.html
            # https://medium.com/dataman-in-ai/the-shap-with-more-elegant-charts-bc3e73fa1c0c
            # https://github.com/shap/shap/issues/1420
            # https://github.com/shap/shap/issues/2470 <-- waterfall_legacy (X)
            # https://github.com/shap/shap/issues/1420 <-- waterfall_legacy (X)
            # https://stackoverflow.com/questions/71751251/get-waterfall-plot-values-of-a-feature-in-a-dataframe-using-shap-package <-- waterfall_legacy (X)
            # shap.plots.waterfall(shap_values[0])
            # https://github.com/shap/shap/blob/2262893cf441478418abac5fd8cdd38e436a867b/shap/plots/_waterfall.py#L321C107-L321C117

            if Test_SG_name in misprediction_subgraph_names:
               waterfallplot_out.savefig( os.path.join(Mispredictions_WaterfallPlots_subdirpath, f"{N}-gram SHAP_local_interpretability_waterfall_plot_{Test_SG_name}.png") )
            else:
               waterfallplot_out.savefig( os.path.join(Correct_Predictions_WaterfallPlots_subdirpath, f"{N}-gram SHAP_local_interpretability_waterfall_plot_{Test_SG_name}.png") )






      return 



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
    'Thisgroupofeventstrackstheperformanceofflushinghives',#36-------
    'EventID(1)',#37
    'EventID(2)',#38
    'EventID(3)',#39
    'EventID(4)',#40
    'EventID(5)',#41
    'EventID(6)',#42
    'EventID(7)',#43
    'EventID(8)',#44
    'EventID(9)',#45
    'EventID(10)',#46
    'EventID(11)',#47
    'EventID(13)',#48
    'EventID(14)',#49
    'EventID(15)',#50
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
##########################################################################################################################################################
# Signal-Amplification Function (Thread-level Event-Dist. 1gram + Adjacent Node's Node-Type 5Bit)
#PW: Thread node embedding by aggregating one hop neighbour nodes
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
            # if data.x.shape[1] != 5:
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
##########################################################################################################################################################
##########################################################################################################################################################
#PW: just experiments
def get_signal_amplified_thread_level_eventdist_adjacent_5bit_dist__INCOMING_AND_OUTGOING_CONCATENATED_20230821_PROF_GUANHUA__dict( dataset : list ):
      
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
            

            prof_guanhua__suggested_features = torch.tensor([]) # Added by JY @ 2023-07-19

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
               # edge_attr_of_both_direction_edges_from_thread_node_idx_sum = torch.add(edge_attr_of_incoming_edges_from_thread_node_idx_sum, 
               #                                                                        edge_attr_of_outgoing_edges_from_thread_node_idx_sum)
               
               # data_thread_node_both_direction_edges_edge_attrs = torch.cat(( data_thread_node_both_direction_edges_edge_attrs,
               #                                                                edge_attr_of_both_direction_edges_from_thread_node_idx_cat.unsqueeze(0) ), dim = 0)
               # -------------------------------------------------------------------------------------------------------------------------------------------------------

               # JY @ 2023-08-21: Do as Prof. Guanhua advised

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
               # unique_adjacent_nodes_of_both_direction_edges_of_thread_node = torch.unique( torch.cat( [ target_nodes_of_outgoing_edges_from_thread_node, 
               #                                                                                           source_nodes_of_incoming_edges_to_thread_node ] ) )

               unique_adjacent_nodes_of_outgoing_edges_of_thread_node = torch.unique( target_nodes_of_outgoing_edges_from_thread_node ) 
               unique_adjacent_nodes_of_incoming_edges_of_thread_node = torch.unique( source_nodes_of_incoming_edges_to_thread_node ) 



               distribution_of_unique_outgoing_edge_adjacent_nodes_of_thread = torch.sum( data__5bit[unique_adjacent_nodes_of_outgoing_edges_of_thread_node], dim = 0 )
               distribution_of_unique_incoming_edge_adjacent_nodes_of_thread = torch.sum( data__5bit[unique_adjacent_nodes_of_incoming_edges_of_thread_node], dim = 0 )               





               # JY @ 20230821
               prof_guanhua__suggested_feature = torch.cat([edge_attr_of_incoming_edges_from_thread_node_idx_sum,
                                                             distribution_of_unique_incoming_edge_adjacent_nodes_of_thread,

                                                             edge_attr_of_outgoing_edges_from_thread_node_idx_sum,
                                                             distribution_of_unique_outgoing_edge_adjacent_nodes_of_thread])


               prof_guanhua__suggested_features = torch.cat(( prof_guanhua__suggested_features,
                                                                              prof_guanhua__suggested_feature.unsqueeze(0) ), dim = 0)


               # data_thread_node_all_unique_adjacent_nodes_5bit_dists = torch.cat(( data_thread_node_all_unique_adjacent_nodes_5bit_dists,
               #                                                                            integrated_5bit_of_all_unique_adjacent_nodes_to_thread.unsqueeze(0) ), dim = 0)




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

            data_dict[ re.search(r'Processed_SUBGRAPH_P3_(.*?)\.pickle', data.name).group(1) ] = prof_guanhua__suggested_features.tolist()

            # data_dict[ data.name.lstrip("Processed_SUBGRAPH_P3_").rstrip(".pickle") ] = data_thread_node_both_direction_edges_edge_attrs.tolist()
            cnt+=1
      return data_dict


##########################################################################################################################################################
##########################################################################################################################################################
# Signal-Amplification Function (Thread-level Event-Dist. 1gram + Adjacent Node's adhoc_identifier)
#PW: just experiments
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
            #if data.x.shape[1] != 5:
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

   elif signal_amplification_option == "signal_amplified__event_1gram_nodetype_5bit_INCOMING_OUTGOING_CONCATENATED_PROFGUANHUA_20230821":
      nodetype_names = ["file", "registry", "network", "process", "thread"] # this is the correct-order
      feature_names = [ f"{x}___incoming_edge" for x in taskname_colnames] +  [ f"{x}___incoming_edge" for x in nodetype_names] +\
                      [ f"{x}___outgoing_edge" for x in taskname_colnames] +  [ f"{x}___outgoing_edge" for x in nodetype_names]
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


   # entire_data_df.to_csv("concat_separate_incoming_outgoing_node_edge_featvectors__prof_guanhua.csv")
   return sample_grouby_df

#**********************************************************************************************************************************************************************
#PW: baseline similar to RF flatten code
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

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('-k', '--K', nargs = 1, type = int, default = [10])  

    parser.add_argument('-data', '--dataset', 
                        choices= ['Dataset-Case-1', 'Dataset-Case-2'], 
                        default = ["Dataset-Case-1"])


    model_cls_map = {"RandomForest": RandomForestClassifier, "XGBoost": GradientBoostingClassifier,
                     "LogisticRegression": LogisticRegression, "SVM": svm } 
    
    parser.add_argument('-mod_cls', '--trad_model_cls', nargs = 1, type = str, 
                        default = ["RandomForest"] )

    parser.add_argument('-ss_opt', '--search_space_option', 
                        choices= [ 
                                 # defaults
                                 "XGBoost_default_hyperparam",
                                 "RandomForest_default_hyperparam",

                                 #PW: RF and XGboost ----------------------------------------------------------
                                 # JY: non-ahoc seems to be '5bit eventdist' + final-test
                                 "RandomForest_best_hyperparameter_max_case1",
                                 "RandomForest_best_hyperparameter_max_case1_nograph",

                                 "RandomForest_best_hyperparameter_max_case2",
                                 "RandomForest_best_hyperparameter_max_case2_nograph",   

                                 "RandomForest_best_hyperparameter_max_case1_ahoc",   
                                 "RandomForest_best_hyperparameter_max_case1_ahoc_nograph",  

                                 "RandomForest_best_hyperparameter_max_case2_ahoc",   
                                 "RandomForest_best_hyperparameter_max_case2_ahoc_nograph",

                                 "XGBoost_best_hyperparameter_max_case1",   
                                 "XGBoost_best_hyperparameter_max_case1_nograph",

                                 "XGBoost_best_hyperparameter_max_case2",
                                 "XGBoost_best_hyperparameter_max_case2_nograph",

                                 "XGBoost_best_hyperparameter_max_case1_ahoc",
                                 "XGBoost_best_hyperparameter_max_case1_ahoc_nograph",

                                 "XGBoost_best_hyperparameter_max_case2_ahoc",
                                 "XGBoost_best_hyperparameter_max_case2_ahoc_nograph",

                                 # -----------------Mean------------------------------------
                                 "RandomForest_best_hyperparameter_mean_case1",
                                 "RandomForest_best_hyperparameter_mean_case1_nograph",

                                 "RandomForest_best_hyperparameter_mean_case2",
                                 "RandomForest_best_hyperparameter_mean_case2_nograph",

                                 "RandomForest_best_hyperparameter_mean_case1_ahoc",
                                 "RandomForest_best_hyperparameter_mean_case1_ahoc_nograph",

                                 "RandomForest_best_hyperparameter_mean_case2_ahoc",
                                 "RandomForest_best_hyperparameter_mean_case2_ahoc_nograph",
                                  
                                  ], 
                                  default = ["RandomForest_best_hyperparameter_max_case1"])
   
    parser.add_argument('-sig_amp_opt', '--signal_amplification_option', 
                        
                        choices= ['signal_amplified__event_1gram', #PW: this is a pair, want to see the effect of signal amplication vs no signal amplification
                                  'no_graph_structure__event_1gram', #PW: signal amplification means graph embedding

                                  'signal_amplified__event_1gram_nodetype_5bit_INCOMING_OUTGOING_CONCATENATED_PROFGUANHUA_20230821',
                                  
                                  'signal_amplified__event_1gram_nodetype_5bit', 
                                  'no_graph_structure__event_1gram_nodetype_5bit', 

                                  'signal_amplified__event_1gram_nodetype_5bit_and_Ahoc_Identifier',
                                  'no_graph_structure__event_1gram_nodetype_5bit_and_Ahoc_Identifier',
                                  ], 

                                  default = ["signal_amplified__event_1gram_nodetype_5bit"])
    
    parser.add_argument('-readout_opt', '--readout_option', 
                        choices= ['max', 'mean' ], 
                        default = ["max"])

    parser.add_argument("--search_on_train__or__final_test", 
                                 
                         choices= ["search_on_train", "final_test", "search_on_all"],  # TODO PW:use "final_test" on test dataset
                         #PW: serach on all- more robust, --> next to run
                                  
                         default = ["final_test"] )

    # cmd args
    K = parser.parse_args().K[0]
    model_cls = model_cls_map[ parser.parse_args().trad_model_cls[0] ]
    dataset_choice = parser.parse_args().dataset[0]

    signal_amplification_option = parser.parse_args().signal_amplification_option[0]
    readout_option = parser.parse_args().readout_option[0]
    search_space_option = parser.parse_args().search_space_option[0]
    search_on_train__or__final_test = parser.parse_args().search_on_train__or__final_test[0] 
    # saved_models_dirpath = "/home/pwakodi1/tabby/Graph_embedding_aka_signal_amplification_files/Stratkfold_Priti/saved_Traditional_ML_models"   #PW:to change
    #saved_models_dirpath = "/home/pwakodi1/tabby/Graph_embedding_aka_signal_amplification_files/Stratkfold_mean_Priti/saved_Traditional_ML_models"   #PW:to change
    model_cls_name = re.search(r"'(.*?)'", str(model_cls)).group(1)


    if search_on_train__or__final_test in {"search_on_train", "search_on_all"}:
       run_identifier = f"{model_cls_name}__{dataset_choice}__{search_space_option}__{K}_FoldCV__{search_on_train__or__final_test}__{signal_amplification_option}__{readout_option}__{datetime.now().strftime('%Y-%m-%d_%H%M%S')}"
       experiment_results_df_fpath = os.path.join(this_results_dirpath, f"{run_identifier}.csv")
       if not os.path.exists(this_results_dirpath):
           os.makedirs(this_results_dirpath)


    if search_on_train__or__final_test == "final_test":
       run_identifier = f"{model_cls_name}__{dataset_choice}__{search_space_option}__{search_on_train__or__final_test}__{signal_amplification_option}__{readout_option}__{datetime.now().strftime('%Y-%m-%d_%H%M%S')}"
       this_results_dirpath = f"/data/d1/jgwak1/tabby/graph_embedding_improvement_JY_git/analyze_at_model_explainer/RESULTS/{run_identifier}"
       final_test_results_df_fpath = os.path.join(this_results_dirpath, f"{run_identifier}.csv")
       if not os.path.exists(this_results_dirpath):
           os.makedirs(this_results_dirpath)

    trace_filename = f'traces_stratkfold_double_strat_{model_cls_name}__generated@'+str(datetime.now())+".txt" 

    ###############################################################################################################################################
    # Set data paths
    ###############################################################################################################################################
    # Set data paths
    projection_datapath_Benign_Train_dict = {
      # Dataset-1 (B#288, M#248) ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
      #PW: Dataset-Case-1 
      "Dataset-Case-1": \
         "/data/d1/jgwak1/tabby/SILKETW_benign_train_test_data_case1/offline_train/Processed_Benign_ONLY_TaskName_edgeattr",
      # ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
      # Dataset-2 (B#662, M#628)
      "Dataset-Case-2": \
         "/data/d1/jgwak1/tabby/SILKETW_benign_train_test_data_case1_case2/offline_train/Processed_Benign_ONLY_TaskName_edgeattr"
    }
    projection_datapath_Malware_Train_dict = {
      # Dataset-1 (B#288, M#248) ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
      "Dataset-Case-1": \
         "/data/d1/jgwak1/tabby/SILKETW_malware_train_test_data_case1/offline_train/Processed_Malware_ONLY_TaskName_edgeattr",
      # ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
      # Dataset-2 (B#662, M#628)
      "Dataset-Case-2": \
         "/data/d1/jgwak1/tabby/SILKETW_malware_train_test_data_case1_case2/offline_train/Processed_Malware_ONLY_TaskName_edgeattr"
    }
    projection_datapath_Benign_Test_dict = {
      # Dataset-1 (B#73, M#62) ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
      "Dataset-Case-1": \
         "/data/d1/jgwak1/tabby/SILKETW_benign_train_test_data_case1/offline_test/Processed_Benign_ONLY_TaskName_edgeattr",
      # ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
      # Dataset-2 (B#167, M#158)
      "Dataset-Case-2": \
         "/data/d1/jgwak1/tabby/SILKETW_benign_train_test_data_case1_case2/offline_test/Processed_Benign_ONLY_TaskName_edgeattr"
    }
    projection_datapath_Malware_Test_dict = {
      # Dataset-1 (B#73, M#62) ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
      "Dataset-Case-1": \
         "/data/d1/jgwak1/tabby//SILKETW_malware_train_test_data_case1/offline_test/Processed_Malware_ONLY_TaskName_edgeattr",
      # ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
      # Dataset-2 (B#167, M#158)
      "Dataset-Case-2": \
         "/data/d1/jgwak1/tabby/SILKETW_malware_train_test_data_case1_case2/offline_test/Processed_Malware_ONLY_TaskName_edgeattr"
    }
    ###############################################################################################################################################

    _num_classes = 2  # number of class labels and always binary classification.

   #PW: 5 and 62 (61 taskname + 1 timestamp) based on new silketw 
    _dim_node = 5 #46   # num node features ; the #feats
    _dim_edge = 62 #72    # (or edge_dim) ; num edge features


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

    def RandomForest_default_hyperparam() -> dict :
      # https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier


      search_space = dict()
      search_space['n_estimators'] = [ 100 ] # The number of decision trees in the random forest. Typical range: 50 to 500 or more. More trees generally lead to better performance, but there are diminishing returns beyond a certain point.
      search_space['criterion'] = [ 'gini' ] # The function used to measure the quality of a split. Typical options: 'gini' for Gini impurity and 'entropy' for information gain. Gini is the default and works well in most cases.
      search_space['max_depth'] = [ None  ]  # The maximum depth of each decision tree. Typical range: 3 to 20. Setting it to None allows nodes to expand until all leaves are pure or contain less than 
      search_space['min_samples_split'] = [ 2 ] # The minimum number of samples required to split an internal node. Typical range: 2 to 20.
      search_space['min_samples_leaf'] = [ 1 ]  # The minimum number of samples required to be at a leaf node. Typical range: 1 to 10.
      search_space['max_features'] = [ 'sqrt' ]  # The number of features to consider when looking for the best split. Typical options: 'sqrt' (square root of the total number of features) or 'log2' (log base 2 of the total number of features).
      search_space['bootstrap'] = [ True ] # Whether to use bootstrapped samples when building trees. 
                                           # Typical options: True or False. 
                                           # Setting it to True enables bagging, which is the standard approach.

      search_space["random_state"] = [ 0, 42, 99 ]
      search_space["split_shuffle_seed"] = [ 100 ]

      #-----------------------------------------------------------------------------------------------
      manual_space = []
      # Change order of these for-loops as the
      for n_estimators in search_space['n_estimators']:
         for criterion in search_space['criterion']:
            for max_depth in search_space['max_depth']:
               for min_samples_split in search_space['min_samples_split']:
                   for min_samples_leaf in search_space['min_samples_leaf']:
                        for max_features in search_space['max_features']:
                            for bootstrap in search_space['bootstrap']:
                                 for random_state in search_space["random_state"]:
                                     for split_shuffle_seed in search_space["split_shuffle_seed"]:                        
                                          print(f"appending {[n_estimators, criterion, max_depth, min_samples_split, min_samples_leaf, max_features, bootstrap, random_state, split_shuffle_seed ]}", flush = True )
                                          
                                          manual_space.append(
                                                   {
                                                   'n_estimators': n_estimators, 
                                                   'criterion': criterion, 
                                                   'max_depth': max_depth, 
                                                   'min_samples_split': min_samples_split, 
                                                   'min_samples_leaf': min_samples_leaf, 
                                                   'max_features': max_features,
                                                   'bootstrap': bootstrap,
                                                   'random_state': random_state,
                                                   'split_shuffle_seed': split_shuffle_seed
                                                   }
                                                   )
      return manual_space

    def XGBoost_default_hyperparam() -> dict :
      # https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html
         search_space = dict()

         search_space['n_estimators'] = [ 100 ] # The number of boosting stages (i.e., the number of weak learners or decision trees). Typical range: 50 to 2000 or more.
         search_space['learning_rate'] = [ 0.1 ] # Shrinks the contribution of each tree, helping to prevent overfitting. Typical range: 0.001 to 0.1.      
         search_space['max_depth'] = [ 3 ]  # The maximum depth of each decision tree. Typical range: 3 to 10. Can use None for no limit on the depth, but this can lead to overfitting
         search_space['min_samples_split'] = [ 2 ] # The minimum number of samples required to split an internal node. Typical range: 2 to 20.
         search_space['min_samples_leaf'] = [ 1 ]  # The minimum number of samples required to be at a leaf node. Typical range: 1 to 10.
         search_space['subsample'] = [ 1.0 ] # The fraction of samples used for fitting the individual trees. Typical range: 0.5 to 1.0.
         search_space['max_features'] = [ None ]  # The number of features to consider when looking for the best split. Typical values: 'sqrt' (square root of the total number of features) or 'log2' (log base 2 of the total number of features).
         search_space['loss'] = [ 'log_loss' ]  # The number of features to consider when looking for the best split. Typical values: 'sqrt' (square root of the total number of features) or 'log2' (log base 2 of the total number of features).

         search_space["random_state"] = [ 0, 42, 99 ]
         search_space["split_shuffle_seed"] = [ 100 ]

      #-----------------------------------------------------------------------------------------------
         manual_space = []
         # Change order of these for-loops as the
         for n_estimators in search_space['n_estimators']:
            for learning_rate in search_space['learning_rate']:
               for max_depth in search_space['max_depth']:
                  for min_samples_split in search_space['min_samples_split']:
                     for min_samples_leaf in search_space['min_samples_leaf']:
                        for subsample in search_space['subsample']:
                           for max_features in search_space['max_features']:
                              for loss in search_space['loss']:
                                    for random_state in search_space["random_state"]:
                                       for split_shuffle_seed in search_space["split_shuffle_seed"]:                        
                                             print(f"appending {[n_estimators, learning_rate, max_depth, min_samples_split, min_samples_leaf, subsample, max_features, loss, random_state, split_shuffle_seed ]}", flush = True )
                                             manual_space.append(
                                                      {
                                                      'n_estimators': n_estimators, 
                                                      'learning_rate': learning_rate, 
                                                      'max_depth': max_depth, 
                                                      'min_samples_split': min_samples_split, 
                                                      'min_samples_leaf': min_samples_leaf, 
                                                      'subsample': subsample, 
                                                      'max_features': max_features,
                                                      'loss': loss,
                                                      'random_state': random_state,
                                                      'split_shuffle_seed': split_shuffle_seed
                                                      } )
         return manual_space


    # PW : RF and XGboost best hyperparameter
    def RandomForest_best_hyperparameter_max_case1():
      manual_space = []
      manual_space.append(
               {
               'n_estimators': 100, 
               'criterion': 'gini', 
               'max_depth': 20, 
               'min_samples_split': 5, 
               'min_samples_leaf': 1, 
               'max_features': 'sqrt',
               'bootstrap': False,
               'random_state': 99,
               'split_shuffle_seed': 100
               }
               )
      return manual_space

    def RandomForest_best_hyperparameter_max_case2():
      manual_space = []
      manual_space.append(
               {
               'n_estimators': 100, 
               'criterion': 'gini', 
               'max_depth': 15, 
               'min_samples_split': 2, 
               'min_samples_leaf': 1, 
               'max_features': None,
               'bootstrap': True,
               'random_state': 42,
               'split_shuffle_seed': 100
               }
               )
      return manual_space

    def RandomForest_best_hyperparameter_max_case1_nograph():
      manual_space = []
      manual_space.append(
               {
               'n_estimators': 200, 
               'criterion': 'gini', 
               'max_depth': None, 
               'min_samples_split': 2, 
               'min_samples_leaf': 3, 
               'max_features': None,
               'bootstrap': True,
               'random_state': 99,
               'split_shuffle_seed': 100
               }
               )
      return manual_space

    def RandomForest_best_hyperparameter_max_case2_nograph():
      manual_space = []
      manual_space.append(
               {
               'n_estimators': 300, 
               'criterion': 'gini', 
               'max_depth': 20, 
               'min_samples_split': 2, 
               'min_samples_leaf': 1, 
               'max_features': 'sqrt',
               'bootstrap': False,
               'random_state': 0,
               'split_shuffle_seed': 100
               }
               )
      return manual_space

    def RandomForest_best_hyperparameter_max_case1_ahoc():
      manual_space = []
      manual_space.append(
               {
               'n_estimators': 100, 
               'criterion': 'gini', 
               'max_depth': 20, 
               'min_samples_split': 5, 
               'min_samples_leaf': 1, 
               'max_features': 'sqrt',
               'bootstrap': False,
               'random_state': 99,
               'split_shuffle_seed': 100
               }
               )
      return manual_space

    def RandomForest_best_hyperparameter_max_case2_ahoc():
      manual_space = []
      manual_space.append(
               {
               'n_estimators': 200, 
               'criterion': 'gini', 
               'max_depth': 15, 
               'min_samples_split': 2, 
               'min_samples_leaf': 1, 
               'max_features': None,
               'bootstrap': True,
               'random_state': 42,
               'split_shuffle_seed': 100
               }
               )
      return manual_space

    def RandomForest_best_hyperparameter_max_case1_ahoc_nograph(): #done
      manual_space = []
      manual_space.append(
               {
               'n_estimators': 200, 
               'criterion': 'gini', 
               'max_depth': None, 
               'min_samples_split': 2, 
               'min_samples_leaf': 3, 
               'max_features': None,
               'bootstrap': True,
               'random_state': 99,
               'split_shuffle_seed': 100
               }
               )
      return manual_space

    def RandomForest_best_hyperparameter_max_case2_ahoc_nograph(): #done
      manual_space = []
      manual_space.append(
               {
               'n_estimators': 300, 
               'criterion': 'gini', 
               'max_depth': 20, 
               'min_samples_split': 2, 
               'min_samples_leaf': 1, 
               'max_features': 'sqrt',
               'bootstrap': False,
               'random_state': 0,
               'split_shuffle_seed': 100
               }
               )
      return manual_space

    
 

    def XGBoost_best_hyperparameter_max_case1():   
         Top_1 =  {
                  'n_estimators': 500, 
                  'learning_rate': 0.01, 
                  'max_depth': 6, 
                  'min_samples_split': 2, 
                  'min_samples_leaf': 5, 
                  'subsample': 1.0, 
                  'max_features': 'sqrt',
                  'loss': 'log_loss',
                  'random_state': 99,
                  'split_shuffle_seed': 100 # not needed
                  }               
         manual_space = [ Top_1 ]
         return manual_space

    def XGBoost_best_hyperparameter_max_case2():   
         Top_1 =  {
         'n_estimators': 1000, 
         'learning_rate': 0.1, 
         'max_depth': 3, 
         'min_samples_split': 2, 
         'min_samples_leaf': 1, 
         'subsample': 1.0, 
         'max_features': None,
         'loss': 'log_loss',
         'random_state': 0,
         'split_shuffle_seed': 100 # not needed
         }
         manual_space = [ Top_1 ]
         return manual_space

    def XGBoost_best_hyperparameter_max_case1_nograph():   
         Top_1 =  {
         'n_estimators': 500, 
         'learning_rate': 0.1, 
         'max_depth': 6, 
         'min_samples_split': 2, 
         'min_samples_leaf': 5, 
         'subsample': 0.5, 
         'max_features': 'sqrt',
         'loss': 'log_loss',
         'random_state': 99,
         'split_shuffle_seed': 100 # not needed
         }
         manual_space = [ Top_1 ]
         return manual_space
    
    def XGBoost_best_hyperparameter_max_case2_nograph():   
         Top_1 =  {
         'n_estimators': 1000, 
         'learning_rate': 0.1, 
         'max_depth': 6, 
         'min_samples_split': 9, 
         'min_samples_leaf': 3, 
         'subsample': 0.75, 
         'max_features': None,
         'loss': 'log_loss',
         'random_state': 0,
         'split_shuffle_seed': 100 # not needed
         }
         manual_space = [ Top_1 ]
         return manual_space

    def XGBoost_best_hyperparameter_max_case1_ahoc():   
         Top_1 =  {
         'n_estimators': 500, 
         'learning_rate': 0.01, 
         'max_depth': 6, 
         'min_samples_split': 2, 
         'min_samples_leaf': 5, 
         'subsample': 1.0, 
         'max_features': 'sqrt',
         'loss': 'log_loss',
         'random_state': 99,
         'split_shuffle_seed': 100 # not needed
         }
         manual_space = [ Top_1 ]
         return manual_space

    def XGBoost_best_hyperparameter_max_case2_ahoc():   
         Top_1 =  {
         'n_estimators': 1000, 
         'learning_rate': 0.1, 
         'max_depth': 3, 
         'min_samples_split': 2, 
         'min_samples_leaf': 1, 
         'subsample': 1.0, 
         'max_features': None,
         'loss': 'log_loss',
         'random_state': 0,
         'split_shuffle_seed': 100 # not needed
         }
         manual_space = [ Top_1 ]
         return manual_space

    def XGBoost_best_hyperparameter_max_case1_ahoc_nograph():    #done
         Top_1 =  {
         'n_estimators': 500, 
         'learning_rate': 0.1, 
         'max_depth': 6, 
         'min_samples_split': 2, 
         'min_samples_leaf': 5, 
         'subsample': 0.5, 
         'max_features': 'sqrt',
         'loss': 'log_loss',
         'random_state': 99,
         'split_shuffle_seed': 100 # not needed
         }
         manual_space = [ Top_1]
         return manual_space

    def XGBoost_best_hyperparameter_max_case2_ahoc_nograph():   
         Top_1 =  {
         'n_estimators': 1000, 
         'learning_rate': 0.1, 
         'max_depth': 6, 
         'min_samples_split': 9, 
         'min_samples_leaf': 3, 
         'subsample': 0.75, 
         'max_features': None,
         'loss': 'log_loss',
         'random_state': 0,
         'split_shuffle_seed': 100 # not needed
         }
         manual_space = [ Top_1]
         return manual_space

    #For ----------------Mean------------------------
    def RandomForest_best_hyperparameter_mean_case1(): #done
      manual_space = []
      manual_space.append(
               {
               'n_estimators': 300, 
               'criterion': 'gini', 
               'max_depth': 9, 
               'min_samples_split': 2, 
               'min_samples_leaf': 1, 
               'max_features': None,
               'bootstrap': True,
               'random_state': 99,
               'split_shuffle_seed': 100
               }
               )
      return manual_space

    def RandomForest_best_hyperparameter_mean_case2():
      manual_space = []
      manual_space.append(
               {
               'n_estimators': 100, 
               'criterion': 'gini', 
               'max_depth': 20, 
               'min_samples_split': 5, 
               'min_samples_leaf': 1, 
               'max_features': 'sqrt',
               'bootstrap': False,
               'random_state': 99,
               'split_shuffle_seed': 100
               }
               )
      return manual_space
    
    def RandomForest_best_hyperparameter_mean_case1_nograph(): #done
      manual_space = []
      manual_space.append(
               {
               'n_estimators': 200, 
               'criterion': 'gini', 
               'max_depth': None, 
               'min_samples_split': 2, 
               'min_samples_leaf': 3, 
               'max_features': None,
               'bootstrap': True,
               'random_state': 99,
               'split_shuffle_seed': 100
               }
               )
      return manual_space
    
    def RandomForest_best_hyperparameter_mean_case2_nograph():
      manual_space = []
      manual_space.append(
               {
               'n_estimators': 100, 
               'criterion': 'gini', 
               'max_depth': 20, 
               'min_samples_split': 5, 
               'min_samples_leaf': 1, 
               'max_features': 'sqrt',
               'bootstrap': False,
               'random_state': 99,
               'split_shuffle_seed': 100
               }
               )
      return manual_space
    
    def RandomForest_best_hyperparameter_mean_case1_ahoc(): #done
      manual_space = []
      manual_space.append(
               {
               'n_estimators': 300, 
               'criterion': 'gini', 
               'max_depth': 9, 
               'min_samples_split': 2, 
               'min_samples_leaf': 1, 
               'max_features': None,
               'bootstrap': True,
               'random_state': 99,
               'split_shuffle_seed': 100
               }
               )
      return manual_space
    
    def RandomForest_best_hyperparameter_mean_case2_ahoc():
      manual_space = []
      manual_space.append(
               {
               'n_estimators': 100, 
               'criterion': 'gini', 
               'max_depth': 20, 
               'min_samples_split': 5, 
               'min_samples_leaf': 1, 
               'max_features': 'sqrt',
               'bootstrap': False,
               'random_state': 99,
               'split_shuffle_seed': 100
               }
               )
      return manual_space
    
    def RandomForest_best_hyperparameter_mean_case1_ahoc_nograph(): #done
      manual_space = []
      manual_space.append(
               {
               'n_estimators': 200, 
               'criterion': 'gini', 
               'max_depth': None, 
               'min_samples_split': 2, 
               'min_samples_leaf': 3, 
               'max_features': None,
               'bootstrap': True,
               'random_state': 99,
               'split_shuffle_seed': 100
               }
               )
      return manual_space
    
    def RandomForest_best_hyperparameter_mean_case2_ahoc_nograph():
      manual_space = []
      manual_space.append(
               {
               'n_estimators': 100, 
               'criterion': 'gini', 
               'max_depth': 20, 
               'min_samples_split': 5, 
               'min_samples_leaf': 1, 
               'max_features': 'sqrt',
               'bootstrap': False,
               'random_state': 99,
               'split_shuffle_seed': 100
               }
               )
      return manual_space



   #TODO PW add here when found best hyperparameter set

    ####################################################################################################################################################
    # defaults
    if search_space_option == "XGBoost_default_hyperparam": search_space = XGBoost_default_hyperparam()   
    elif search_space_option == "RandomForest_default_hyperparam": search_space = RandomForest_default_hyperparam()   

   #PW: RF and XGboost ----------------------------------------------------------
    # JY: non-ahoc seems to be '5bit eventdist'
    elif search_space_option == "RandomForest_best_hyperparameter_max_case1": search_space = RandomForest_best_hyperparameter_max_case1()   
    elif search_space_option == "RandomForest_best_hyperparameter_max_case2": search_space = RandomForest_best_hyperparameter_max_case2()   
    elif search_space_option == "RandomForest_best_hyperparameter_max_case1_nograph": search_space = RandomForest_best_hyperparameter_max_case1_nograph()   
    elif search_space_option == "RandomForest_best_hyperparameter_max_case2_nograph": search_space = RandomForest_best_hyperparameter_max_case2_nograph()   
    elif search_space_option == "RandomForest_best_hyperparameter_max_case1_ahoc": search_space = RandomForest_best_hyperparameter_max_case1_ahoc()   
    elif search_space_option == "RandomForest_best_hyperparameter_max_case2_ahoc": search_space = RandomForest_best_hyperparameter_max_case2_ahoc()   
    elif search_space_option == "RandomForest_best_hyperparameter_max_case1_ahoc_nograph": search_space = RandomForest_best_hyperparameter_max_case1_ahoc_nograph()   
    elif search_space_option == "RandomForest_best_hyperparameter_max_case2_ahoc_nograph": search_space = RandomForest_best_hyperparameter_max_case2_ahoc_nograph()   
    
    elif search_space_option == "XGBoost_best_hyperparameter_max_case1": search_space = XGBoost_best_hyperparameter_max_case1()   
    elif search_space_option == "XGBoost_best_hyperparameter_max_case2": search_space = XGBoost_best_hyperparameter_max_case2()   
    elif search_space_option == "XGBoost_best_hyperparameter_max_case1_nograph": search_space = XGBoost_best_hyperparameter_max_case1_nograph()   
    elif search_space_option == "XGBoost_best_hyperparameter_max_case2_nograph": search_space = XGBoost_best_hyperparameter_max_case2_nograph()   
    elif search_space_option == "XGBoost_best_hyperparameter_max_case1_ahoc": search_space = XGBoost_best_hyperparameter_max_case1_ahoc()   
    elif search_space_option == "XGBoost_best_hyperparameter_max_case2_ahoc": search_space = XGBoost_best_hyperparameter_max_case2_ahoc()   
    elif search_space_option == "XGBoost_best_hyperparameter_max_case1_ahoc_nograph": search_space = XGBoost_best_hyperparameter_max_case1_ahoc_nograph()   
    elif search_space_option == "XGBoost_best_hyperparameter_max_case2_ahoc_nograph": search_space = XGBoost_best_hyperparameter_max_case2_ahoc_nograph()   

   # -----------------Mean------------------------------------
    elif search_space_option == "RandomForest_best_hyperparameter_mean_case1": search_space = RandomForest_best_hyperparameter_mean_case1()   
    elif search_space_option == "RandomForest_best_hyperparameter_mean_case2": search_space = RandomForest_best_hyperparameter_mean_case2()   
    elif search_space_option == "RandomForest_best_hyperparameter_mean_case1_nograph": search_space = RandomForest_best_hyperparameter_mean_case1_nograph()   
    elif search_space_option == "RandomForest_best_hyperparameter_mean_case2_nograph": search_space = RandomForest_best_hyperparameter_mean_case2_nograph()   
    elif search_space_option == "RandomForest_best_hyperparameter_mean_case1_ahoc": search_space = RandomForest_best_hyperparameter_mean_case1_ahoc()   
    elif search_space_option == "RandomForest_best_hyperparameter_mean_case2_ahoc": search_space = RandomForest_best_hyperparameter_mean_case2_ahoc()   
    elif search_space_option == "RandomForest_best_hyperparameter_mean_case1_ahoc_nograph": search_space = RandomForest_best_hyperparameter_mean_case1_ahoc_nograph()   
    elif search_space_option == "RandomForest_best_hyperparameter_mean_case2_ahoc_nograph": search_space = RandomForest_best_hyperparameter_mean_case2_ahoc_nograph()   

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

    if search_on_train__or__final_test == "search_on_all":  # ***** #
         train_dataset = train_dataset + final_test_dataset

    # Now apply signal-amplification here (here least conflicts with existing code.)
    if signal_amplification_option == "signal_amplified__event_1gram_nodetype_5bit":
         train_dataset__signal_amplified_dict = get_signal_amplified_thread_level_eventdist_adjacent_5bit_dist_dict( dataset= train_dataset )

    elif signal_amplification_option == "signal_amplified__event_1gram":
         train_dataset__signal_amplified_dict = get_signal_amplified_thread_level_eventdist_dict( dataset= train_dataset )            

    elif signal_amplification_option == "signal_amplified__event_1gram_nodetype_5bit_and_Ahoc_Identifier":
         train_dataset__signal_amplified_dict = get_signal_amplified_thread_level_eventdist_adjacent_5bit_and_Adhoc_Identifier_dist_dict( dataset= train_dataset ) 

    elif signal_amplification_option == "signal_amplified__event_1gram_nodetype_5bit_INCOMING_OUTGOING_CONCATENATED_PROFGUANHUA_20230821":
         train_dataset__signal_amplified_dict = get_signal_amplified_thread_level_eventdist_adjacent_5bit_dist__INCOMING_AND_OUTGOING_CONCATENATED_20230821_PROF_GUANHUA__dict( dataset= train_dataset ) 
    # --------------------------------------------------------------------------------------
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
         
         if signal_amplification_option in {"signal_amplified__event_1gram_nodetype_5bit", "no_graph_structure__event_1gram_nodetype_5bit"}:
               nodetype_names = ["file", "registry", "network", "process", "thread"] # this is the correct-order
               feature_names = taskname_colnames + nodetype_names

         if signal_amplification_option in {"signal_amplified__event_1gram_nodetype_5bit_and_Ahoc_Identifier", 
                                            "no_graph_structure__event_1gram_nodetype_5bit_and_Ahoc_Identifier"}:
               nodetype_names = ["file", "registry", "network", "process", "thread"] # this is the correct-order
               feature_names = taskname_colnames + nodetype_names + [f"adhoc_pattern_{i}" for i in range(len(X.columns) - len(taskname_colnames) - len(nodetype_names))]
         
         if signal_amplification_option in {"signal_amplified__event_1gram", "no_graph_structure__event_1gram"} :
               feature_names = taskname_colnames
         
         X.columns = feature_names
         X.reset_index(inplace = True)
         X.rename(columns = {'index':'data_name'}, inplace = True)

         y = [data.y for data in train_dataset]
    # =================================================================================================================================================
    # =================================================================================================================================================

    if search_on_train__or__final_test == "final_test":
        # Also prepare for final-test dataset, to later test the best-fitted models on test-set
         # Now apply signal-amplification here (here least conflicts with existing code.)
         if signal_amplification_option == "signal_amplified__event_1gram_nodetype_5bit":
               final_test_dataset__signal_amplified_dict = get_signal_amplified_thread_level_eventdist_adjacent_5bit_dist_dict( dataset= final_test_dataset )

         elif signal_amplification_option == "signal_amplified__event_1gram":
               final_test_dataset__signal_amplified_dict = get_signal_amplified_thread_level_eventdist_dict( dataset= final_test_dataset )            

         elif signal_amplification_option == "signal_amplified__event_1gram_nodetype_5bit_and_Ahoc_Identifier":
               final_test_dataset__signal_amplified_dict = get_signal_amplified_thread_level_eventdist_adjacent_5bit_and_Adhoc_Identifier_dist_dict( dataset= final_test_dataset )            

         elif signal_amplification_option == "signal_amplified__event_1gram_nodetype_5bit_INCOMING_OUTGOING_CONCATENATED_PROFGUANHUA_20230821":
               final_test_dataset__signal_amplified_dict = get_signal_amplified_thread_level_eventdist_adjacent_5bit_dist__INCOMING_AND_OUTGOING_CONCATENATED_20230821_PROF_GUANHUA__dict( dataset= final_test_dataset ) 

         elif signal_amplification_option == "no_graph_structure__event_1gram_nodetype_5bit":
               final_test_dataset__no_graph_structure_dict = get_No_Graph_Structure_eventdist_nodetype5bit_dist_dict( dataset= final_test_dataset )  

         elif signal_amplification_option == "no_graph_structure__event_1gram":
               final_test_dataset__no_graph_structure_dict = get_No_Graph_Structure_eventdist_dict( dataset= final_test_dataset )  
         
         elif signal_amplification_option == "no_graph_structure__event_1gram_nodetype_5bit_and_Ahoc_Identifier":
               final_test_dataset__no_graph_structure_dict = get_No_Graph_Structure_eventdist_nodetype5bit_adhoc_identifier_dist_dict( dataset= final_test_dataset )  

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

    # *************************************************************************************************************************************************
    # *************************************************************************************************************************************************         
    # *************************************************************************************************************************************************
    # *************************************************************************************************************************************************         
    # *************************************************************************************************************************************************
    # *************************************************************************************************************************************************
    # *************************************************************************************************************************************************         
    # *************************************************************************************************************************************************
    # *************************************************************************************************************************************************         
    # *************************************************************************************************************************************************
    if search_on_train__or__final_test in {"search_on_train", "search_on_all"}:
               
            # Instantiate Experiments Results Dataframe
            colnames= list(search_space[0].keys()) + [t+"_"+m for t,m in list(product(["Avg_Val"],["Accuracy","F1","Precision","Recall"]))] +\
                                                            ["Std_Val_F1", "Std_Val_Accuracy"] + \
                                                            ["K_Val_"+m for m in list(["Accuracy","Precision","Recall","F1"])]
            experiment_results_df = pd.DataFrame(columns=colnames)

            ####################################################################################################################################################

            """ try out different tunable parameter combinations """
            
            # test_results = []
            for hyperparam_set in search_space:

               split_shuffle_seed = hyperparam_set['split_shuffle_seed']

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



               # ==================================================================================================================================================
               # [Double-Stratification] ===========================================================================================================================

               X_grouplist = []

               data_names = X['data_name']

               for data_name in data_names:

                     # malware ------------------------------------------
                     if "empire" in data_name: X_grouplist.append("malware_empire")
                     elif "invoke_obfuscation" in data_name: X_grouplist.append("malware_invoke_obfuscation")
                     elif "nishang" in data_name: X_grouplist.append("malware_nishang")
                     elif "poshc2" in data_name: X_grouplist.append("malware_poshc2")
                     elif "mafia" in data_name: X_grouplist.append("malware_mafia")
                     elif "offsec" in data_name: X_grouplist.append("malware_offsec")
                     elif "powershellery" in data_name: X_grouplist.append("malware_powershellery")
                     elif "psbits" in data_name: X_grouplist.append("malware_psbits")
                     elif "pt_toolkit" in data_name: X_grouplist.append("malware_pt_toolkit")
                     elif "randomps" in data_name: X_grouplist.append("malware_randomps")
                     elif "smallposh" in data_name: X_grouplist.append("malware_smallposh")
                     #PW: need to change depends on all types e.g., malware_rest
                     elif "asyncrat" in data_name: X_grouplist.append("malware_asyncrat")
                     elif "bumblebee" in data_name: X_grouplist.append("malware_bumblebee")
                     elif "cobalt_strike" in data_name: X_grouplist.append("malware_cobalt_strike")
                     elif "coinminer" in data_name: X_grouplist.append("malware_coinminer")
                     elif "gozi" in data_name: X_grouplist.append("malware_gozi")
                     elif "guloader" in data_name: X_grouplist.append("malware_guloader")
                     elif "netsupport" in data_name: X_grouplist.append("malware_netsupport")
                     elif "netwalker" in data_name: X_grouplist.append("malware_netwalker")
                     elif "nw0rm" in data_name: X_grouplist.append("malware_nw0rm")
                     elif "quakbot" in data_name: X_grouplist.append("malware_quakbot")
                     elif "quasarrat" in data_name: X_grouplist.append("malware_quasarrat")
                     elif "rest" in data_name: X_grouplist.append("malware_rest")
                     elif "metasploit" in data_name: X_grouplist.append("malware_metasploit")                       
                     
                     # benign -------------- # PW:for silketw we dont have benign source level identifier                  
                     else:
                        X_grouplist.append("benign")

               # correctness of X_grouplist can be checked by following
               # list(zip(X, [data_name for data_name in X.index], y, X_grouplist))

               dataset_cv = StratifiedKFold(n_splits=K, 
                                          shuffle = True, 
                                          random_state=np.random.RandomState(seed=split_shuffle_seed))

               #################################################################################################################
               k_validation_results = []
               k_results = defaultdict(list)
               avg_validation_results = {}

               for train_idx, validation_idx in dataset_cv.split(X, y = X_grouplist):  # JY @ 2023-06-27 : y = X_grouplist is crucial for double-stratification

                     def convert_to_label(value):
                        # if "benign" in value.lower(): return 0
                        if "malware" in value.lower(): return 1
                        else:
                           return 0
                        #else: ValueError("no benign or malware substr")

                     X_train = X.iloc[train_idx].drop("data_name", axis = 1)
                     y_train = X.iloc[train_idx]['data_name'].apply(convert_to_label)
                     X_validation = X.iloc[validation_idx].drop("data_name", axis = 1)
                     y_validation = X.iloc[validation_idx]['data_name'].apply(convert_to_label)
                     
                     #TODO: PW: look for the value while running
                     print(f"Benign: #train:{y_train.value_counts()[0]}, #validation:{y_validation.value_counts()[0]},\nMalware: #train:{y_train.value_counts()[1]}, #validation:{y_validation.value_counts()[1]}", flush=True)
                     
                     # train / record training time
                     train_start = datetime.now()
                     model.fit(X = X_train, y = y_train)
                     training_time =str( datetime.now() - train_start )
            
                     # test / record test-time
                     test_start = datetime.now()
                     preds = model.predict(X = X_validation) # modified return-statement for trainer.train() for this.
                     
                     val_accuracy = sklearn.metrics.accuracy_score(y_true = y_validation, y_pred = preds)
                     val_f1 = sklearn.metrics.f1_score(y_true = y_validation, y_pred = preds)
                     val_precision = sklearn.metrics.precision_score(y_true = y_validation, y_pred = preds)
                     val_recall = sklearn.metrics.recall_score(y_true = y_validation, y_pred = preds)
                     tn, fp, fn, tp = sklearn.metrics.confusion_matrix(y_validation, preds).ravel()
                     
                     ## JY @ 2023-07-26: Imp almost done : Start from here
            
                     validation_results = { 
                        "val_acc": val_accuracy, 
                        "val_F1": val_f1,
                        "val_precision": val_precision, 
                        "val_recall": val_recall, 
                        "preds":list(preds), 
                        "truths": list(y_validation),
                     }
                     k_validation_results.append(validation_results)
                     
                     for key in validation_results:
                        k_results[key].append( validation_results[key] )



               print("hyparameter-set: ", hyperparam_set, flush=True)
               # k_results=temp_results.copy()
               print("Avg Validation Result: ", flush=True)
               for key in k_results:
                     if key not in {'preds','truths'}:
                        avg_validation_results[key] = np.mean(k_results[key])
                        print(f"{key}: {avg_validation_results[key]}", flush=True)

               print("k-fold val results:", flush=True)
               for item in k_validation_results:
                     print(item, flush=True)

               # write out 
               with open(trace_filename, 'a') as f:
                     f.write("*"*120+"\n")
                     f.write("param_info:{}\n".format(hyperparam_set))
                     f.write("avg-validation_results: {}\n".format(avg_validation_results))
                     f.write("[validation preds]:\n{}\n".format(k_results['preds']))
                     f.write("[validation truths]:\n{}\n\n".format(k_results['truths']))
                     #f.write("\n[training epochs preds and truths]:\n{}\n\n".format("\n".join("\n<{}>\n\n{}".format(k, "\n".join("\n[{}]]\n\n{}".format(k2, "\n".join("*{}:\n{}".format(k3, v3) for k3, v3 in v2.items()) ) for k2, v2 in v.items())) for k, v in train_results['trainingepochs_preds_truth'].items())))
                     f.write("*"*120+"\n")

                     #f.write("\n[training epochs preds and truths]:\n{}\n\n".format(train_results['trainingepochs_preds_truth']))

                  # check how to make this simpler
               validation_info = {
                                    "Avg_Val_Accuracy": avg_validation_results["val_acc"], 
                                    "Std_Val_Accuracy": np.std(k_results["val_acc"]),
                                    "Avg_Val_F1": avg_validation_results["val_F1"], 
                                    "Std_Val_F1": np.std(k_results["val_F1"]),
                                    "Avg_Val_Precision": avg_validation_results["val_precision"],
                                    "Avg_Val_Recall": avg_validation_results["val_recall"] ,

                                    "K_Val_Accuracy": k_results["val_acc"],
                                    "K_Val_F1": k_results["val_F1"],
                                    "K_Val_Precision": k_results["val_precision"],
                                    "K_Val_Recall": k_results["val_recall"],
                                    # "Train_Time": training_time, "Test_Time": test_time
                                    }
               validation_info.update(hyperparam_set)


               #   tunable_param.update(train_test_info)

               # save collected info to experiments results dataframe
               # experiment_results_df = experiment_results_df.append( train_test_info , ignore_index= True )

               experiment_results_df = pd.concat([experiment_results_df, pd.DataFrame([validation_info])], axis=0)
               # write out csv file every iteration
               experiment_results_df.to_csv(path_or_buf=experiment_results_df_fpath)

    #--------------------------------------------------------------------------------------------------------------------------------------------------------
    #--------------------------------------------------------------------------------------------------------------------------------------------------------
    #--------------------------------------------------------------------------------------------------------------------------------------------------------
    #--------------------------------------------------------------------------------------------------------------------------------------------------------
    if search_on_train__or__final_test == "final_test":

            # Instantiate Experiments Results Dataframe
            colnames= list(search_space[0].keys()) + [t+"_"+m for t,m in list(product(["Final_Test"],["Accuracy","F1","Precision","Recall"]))] +\
                                                            ["FN", "FP", "TN", "TP"] 
            final_test_results_df = pd.DataFrame(columns=colnames)


            for hyperparam_set in search_space:

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

               def convert_to_label(value):
                  # if "benign" in value.lower(): return 0
                  if "malware" in value.lower(): return 1
                  else: return 0
                  # else: ValueError("no benign or malware substr")


               X_ = X.drop("data_name", axis = 1)
               y_ = X['data_name'].apply(convert_to_label)
               final_test_X_ = final_test_X.drop("data_name", axis = 1)
               final_test_y_ = final_test_X['data_name'].apply(convert_to_label)

                     #TODO: PW: look for the value while running

               print(f"Benign: #train:{y_.value_counts()[0]}, #final-test:{final_test_y_.value_counts()[0]},\nMalware: #train:{y_.value_counts()[1]}, #final-test:{final_test_y_.value_counts()[1]}",
                     flush=True)

               model.fit(X = X_, y = y_)
               preds = model.predict(X = final_test_X_) # modified return-statement for trainer.train() for this.
               

               final_test_accuracy = sklearn.metrics.accuracy_score(y_true = final_test_y_, y_pred = preds)
               final_test_f1 = sklearn.metrics.f1_score(y_true = final_test_y_, y_pred = preds)
               final_test_precision = sklearn.metrics.precision_score(y_true = final_test_y_, y_pred = preds)
               final_test_recall = sklearn.metrics.recall_score(y_true = final_test_y_, y_pred = preds)
               tn, fp, fn, tp = sklearn.metrics.confusion_matrix(final_test_y_, preds).ravel()
                        
               final_test_info = {
                                    "Final_Test_Accuracy": final_test_accuracy, 
                                    "Final_Test_F1": final_test_f1, 
                                    "Final_Test_Precision": final_test_precision,
                                    "Final_Test_Recall": final_test_recall ,

                                    "FN": fn,
                                    "FP": fp,
                                    "TN": tn,
                                    "TP": tp,
                                    # "Train_Time": training_time, "Test_Time": test_time
                                    }
               final_test_info.update(hyperparam_set)


               #   tunable_param.update(train_test_info)

               # save collected info to experiments results dataframe
               # experiment_results_df = experiment_results_df.append( train_test_info , ignore_index= True )

               final_test_results_df = pd.concat([final_test_results_df, pd.DataFrame([final_test_info])], axis=0)
               # write out csv file every iteration
               time.sleep(0.05)
               final_test_results_df.to_csv(path_or_buf=final_test_results_df_fpath)


               # Added by JY @ 2023-12-20
               Test_SG_names = final_test_X['data_name']
               triplets = list(zip(Test_SG_names, preds, final_test_y_))
               wrong_answer_incidients = [x for x in triplets if x[1] != x[2]] # if pred != true
               misprediction_subgraph_names = [wrong_answer_incident[0] for wrong_answer_incident in wrong_answer_incidients]


               # Added by JY @ 2023-12-20
               produce_SHAP_explanations(
                              N = 1,
                              classification_model = model,
                              Test_dataset = final_test_X,
                              Train_dataset = X,
                              Explanation_Results_save_dirpath = this_results_dirpath, 
                              model_cls_name = model_cls_name, 
                              Test_SG_names = Test_SG_names,
                              misprediction_subgraph_names = misprediction_subgraph_names
                              )