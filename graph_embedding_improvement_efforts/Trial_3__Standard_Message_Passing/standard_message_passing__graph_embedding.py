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

sys.path.append("/data/d1/jgwak1/tabby/graph_embedding_improvement_JY_git/graph_embedding_improvement_efforts/Trial_3__Standard_Message_Passing/source")

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


import copy # Added by JY @ 2023-12-27

##############################################################################################################################
# Explainer related (2023-12-20)

import shap
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


def produce_SHAP_explanations(classification_model, 
                              Test_dataset : pd.DataFrame,
                              Train_dataset : pd.DataFrame,
                              Explanation_Results_save_dirpath : str, 
                              model_cls_name : str, 
                              Test_SG_names : list,
                              misprediction_subgraph_names : list,
                              Predict_proba_dict : dict,
                              N : int = 1,
                              ):

      # JY @ 2023-12-20: Integrate SHAP into this file based on:
      #                  /data/d1/jgwak1/tabby/BASELINE_COMPARISONS/Sequential/RF+Ngrams/RFSVM_1gram_events_flattened_subgraph_only_psh.py
      #                  /data/d1/jgwak1/tabby/CXAI_2023_Experiments/Run_Explainers/SHAP_LIME__Ngram.py (*)   

      # =============================================================================================================================
      # First convert "EventID(<N>)" to its corresponding 
      Train_dataset.rename(columns = EventID_to_RegEventName_dict, inplace = True)
      Test_dataset.rename(columns = EventID_to_RegEventName_dict, inplace = True)

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

      # JY @ 2023-12-21: Need to get "dataname" here

      Global_Important_featureIndex_featureName = dict(zip(shap_importance.reset_index()['index'], shap_importance.reset_index()['col_name']))
      Global_Important_featureNames = [ v for k,v in Global_Important_featureIndex_featureName.items() ]
      # Save Global-First20Important Features "Train dataset"
      Global_Important_Features_Train_dataset = Train_dataset[ Global_Important_featureNames ]
      
      Train_dataset__data_name_column = Train_dataset['data_name']  # added by JY @ 2023-12-21
      
      Global_Important_Features_Train_dataset = Global_Important_Features_Train_dataset.assign(SUM = Global_Important_Features_Train_dataset.sum(axis=1)) 
      Global_Important_Features_Train_dataset = pd.concat([Train_dataset__data_name_column, Global_Important_Features_Train_dataset], axis = 1) # added by JY @ 2023-12-21

      # added by JY @ 2023-12-21
      def append_prefix(value, prefix): return prefix + value if not value.startswith('malware') else value      
      Global_Important_Features_Train_dataset['data_name'] = Global_Important_Features_Train_dataset['data_name'].apply(lambda x: append_prefix(x, "benign_"))
      Global_Important_Features_Train_dataset.sort_values(by = "data_name", inplace = True)
      Global_Important_Features_Train_dataset.set_index("data_name", inplace = True)

      Global_Important_Features_Train_dataset.to_csv(os.path.join(Explanation_Results_save_dirpath, f"{model_cls_name} {N}-gram Global-SHAP Important FeatureNames Train-Dataset.csv"))

      # Save Global-First20Important Features "Test dataset"
      Global_Important_Features_Test_dataset = Test_dataset[ Global_Important_featureNames ] 

      Test_dataset__data_name_column = Test_dataset['data_name']  # added by JY @ 2023-12-21

      Global_Important_Features_Test_dataset = Global_Important_Features_Test_dataset.assign(SUM = Global_Important_Features_Test_dataset.sum(axis=1)) 

      Global_Important_Features_Test_dataset = pd.concat([Test_dataset__data_name_column, Global_Important_Features_Test_dataset], axis = 1) # added by JY @ 2023-12-21
      Global_Important_Features_Test_dataset.set_index("data_name", inplace = True)
      Global_Important_Features_Test_dataset["predict_proba"] = pd.Series(Predict_proba_dict) # Added by JY @ 2023-12-21

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

      Global_Important_Features_Test_dataset['SHAP_sum_of_feature_shaps'] = None
      Global_Important_Features_Test_dataset['SHAP_base_value'] = None
      cnt = 0
      for Test_SG_name in Test_SG_names:
            cnt += 1
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
               shap_values = shap_values[1]                  # extent to which a sample resembles a malware sample

               base_value = shap_explainer.expected_value[1] #  base value represents the predicted probability of malware 
                                                             #  class if we did not have any information of the feature values 
                                                             #  of this sample
            
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


            # Added by JY @ 2023-12-21 : For local shap of sample (SUM of all feature's shap values for the sample)
            #                                                      ^-- corresponds to "f(x)" in Waterfall plot
            Test_SG_Local_Shap = sum(shap_values)
            Global_Important_Features_Test_dataset.loc[Test_SG_name,'SHAP_sum_of_feature_shaps'] = Test_SG_Local_Shap
            Global_Important_Features_Test_dataset.loc[Test_SG_name, 'SHAP_base_value'] = base_value

            print(f"{cnt} / {len(Test_SG_names)} : SHAP-local done for {Test_SG_name}", flush=True)

      # added by JY @ 2023-12-21
      def append_prefix(value, prefix): return prefix + value if not value.startswith('malware') else value      

      Global_Important_Features_Test_dataset.index = Global_Important_Features_Test_dataset.index.map(lambda x: append_prefix(x, "benign_"))
      Global_Important_Features_Test_dataset.sort_index(inplace=True)

      # Global_Important_Features_Test_dataset['data_name'] = Global_Important_Features_Test_dataset['data_name'].apply(lambda x: append_prefix(x, "benign_"))
      # Global_Important_Features_Test_dataset.sort_values(by = "data_name", inplace = True)
      # Global_Important_Features_Test_dataset.set_index("data_name", inplace = True)

      Global_Important_Features_Test_dataset.to_csv(os.path.join(Explanation_Results_save_dirpath, f"{model_cls_name} {N}-gram Global-SHAP Important FeatureNames Test-Dataset.csv"))

      print("done", flush=True)

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




# def get_readout_applied_df( data_dict : dict,
#                             readout_option : str, 
#                             signal_amplification_option: str):    
   
#    if signal_amplification_option == "signal_amplified__event_1gram_nodetype_5bit":
#       nodetype_names = ["file", "registry", "network", "process", "thread"] # this is the correct-order
#       feature_names = taskname_colnames + nodetype_names
   
#    elif signal_amplification_option == "signal_amplified__event_1gram":
#       feature_names = taskname_colnames

#    elif signal_amplification_option == "signal_amplified__event_1gram_nodetype_5bit_and_Ahoc_Identifier":
#       nodetype_names = ["file", "registry", "network", "process", "thread"] # this is the correct-order
#       feature_names = taskname_colnames + nodetype_names + \
#                       [f"adhoc_pattern_{i}" for i in range(len(data_dict[list(data_dict.keys())[0]][0]) -\
#                                                                len(taskname_colnames) - len(nodetype_names))]

#    elif signal_amplification_option == "signal_amplified__event_1gram_nodetype_5bit_INCOMING_OUTGOING_CONCATENATED_PROFGUANHUA_20230821":
#       nodetype_names = ["file", "registry", "network", "process", "thread"] # this is the correct-order
#       feature_names = [ f"{x}___incoming_edge" for x in taskname_colnames] +  [ f"{x}___incoming_edge" for x in nodetype_names] +\
#                       [ f"{x}___outgoing_edge" for x in taskname_colnames] +  [ f"{x}___outgoing_edge" for x in nodetype_names]
#    else:
#       ValueError(f"Invalid signal_amplification_option ({signal_amplification_option})")  

#    entire_data_df = pd.DataFrame( columns = ["thread_is_from"] + feature_names )
#    for data_name in data_dict:
#       data_name_df = pd.DataFrame( data_dict[data_name], columns = feature_names )
#       data_name_df['thread_is_from'] = data_name
#       entire_data_df = pd.concat([entire_data_df, data_name_df], axis=0, ignore_index=True)
   
#    if readout_option == "max":
#        sample_grouby_df = entire_data_df.groupby("thread_is_from").max()
#    elif readout_option == "mean":
#        sample_grouby_df = entire_data_df.groupby("thread_is_from").mean()


#    # entire_data_df.to_csv("concat_separate_incoming_outgoing_node_edge_featvectors__prof_guanhua.csv")
#    return sample_grouby_df

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

def get__standard_message_passing_graph_embedding__dict( dataset : list, 
                                                         n_hops : int = 1,
                                                         neighborhood_aggr : str = "sum", 
                                                         pool : str = "sum",
                                                         verbose : bool = True ):

   '''
    JY @ 2023-12-27
    
    Generate "graph embedding vectors" based on 
    Standard Message Passing, which is similar to "torch_geometric.nn.conv.SimpleConv" 
    which is "A simple message passing operator that performs (non-trainable) propagation."
    
    More specifically, 
    "n-hop neighborhood aggregation for all nodes within a directed graph; but without trainable parameters" (JY)

    Baseline against this approach is, strictly, "flattened graph 1 gram", and loosely, "flattened graph N gram (N>=1)".
    -----------------------------------------------------------------------------------------------------------------------
    [Pseudocode for standard message passing (with no learning; no trainable parameters)]
      
      1. Initialize each node's feature vector
         ^-- node-feature dimension can be "#original node-feat + #edge-feat" 
             (#edge-feat as zero-subvector in the beginning; think of placeholders) )
             since we don't involve any neural-networks here
      2. For "n_hops": 
      3.    For each node:
      4.        Get messages (concat of edge-feat and node-feat) from the node's neighbors (neighbor-nodes with 'incoming edges')
                ^-- Might have to handle the "duplicate neighboring nodes" problem due to multi-graph ; could utilize approach used in signal-amplification    
      5.        Aggregate messages from neighbors based on "neighborhood_aggr" 
      6.        Update the node's embedding (* Change made to other nodes during this hop must not be reflected; 
                                               So deep-copy a graph in advance that is not mutated but just used for collecting messages)
                                            (* Most basic way for the 'update' is to just 'add' the aggregated-message as node's current embedding )
      7. Since message-passing is done, "pool" all node's embedding, to generate the "graph embedding".
    -----------------------------------------------------------------------------------------------------------------------                                         
    '''

    # References:
    #      https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#convolutional-layers
    # 
    #      https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.SimpleConv.html#torch_geometric.nn.conv.SimpleConv 
    #      ^-- ( REFER TO ABOVE, WHICH IS:  A simple message passing operator that performs (non-trainable) propagation. )
    #          ( COULD LOOK INTO SOURCE CODE OF THIS "SimpleConv" AND MIMIC IT HERE : https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/conv/simple_conv.html#SimpleConv )
    #        
    #      ( Following are just for fun )
    #      https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.GINEConv.html#torch_geometric.nn.conv.GINEConv
    #      https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.SAGEConv.html#torch_geometric.nn.conv.SAGEConv
    # 
    #      ( References for Aggregators )
    #      https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#aggregation-operators
    #      https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.aggr.SumAggregation.html#torch_geometric.nn.aggr.SumAggregation
    #      https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.aggr.MeanAggregation.html#torch_geometric.nn.aggr.MeanAggregation
    #      https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.aggr.MaxAggregation.html#torch_geometric.nn.aggr.MaxAggregation  
    #      https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.aggr.MinAggregation.html#torch_geometric.nn.aggr.MinAggregation
    #      https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.aggr.MulAggregation.html#torch_geometric.nn.aggr.MulAggregation
    #      https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.aggr.VarAggregation.html#torch_geometric.nn.aggr.VarAggregation
    #      https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.aggr.LSTMAggregation.html#torch_geometric.nn.aggr.LSTMAggregation
    #
    #      ( References for Pooling )
    #      https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.pool.global_add_pool.html#torch_geometric.nn.pool.global_add_pool
    #      https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.pool.global_mean_pool.html#torch_geometric.nn.pool.global_mean_pool
    #      https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.pool.global_max_pool.html#torch_geometric.nn.pool.global_max_pool


    #**********************************************************************************************************************************************************************
    # Start implement here.


    # 1. Initialize each node's feature vector
    #    ^-- node-feature dimension can be "#original node-feat + #edge-feat" 
    #        (#edge-feat as zero-subvector in the beginning; think of placeholders) )
    #        since we don't involve any neural-networks here
    #
    # 2. For "n_hops": 
    # 3.    For each node:
    # 4.        Get messages (concat of edge-feat and node-feat) from the node's neighbors (neighbor-nodes with 'incoming edges')
    #           ^-- Might have to handle the "duplicate neighboring nodes" problem due to multi-graph ; could utilize approach used in signal-amplification    
    # 5.        Aggregate messages from neighbors based on "neighborhood_aggr" 
    # 6.        Update the node's embedding (* Change made to other nodes during this hop must not be reflected; 
    #                                          So deep-copy a graph in advance that is not mutated but just used for collecting messages)
    #                                       (* Most basic way for the 'update' is to just 'add' the aggregated-message as node's current embedding )
    #
    # 7. Since message-passing is done, "pool" all node's embedding, to generate the "graph embedding".


   data_dict = dict()
   cnt = 0
   for graph_data in dataset: 
         cnt += 1
         # graph_data corresponds to "torch_geometric.data.data.Data"
         print(f"{cnt} / {len(dataset)}: {graph_data.name}\n", flush = True)
         #original_graph_data = copy.deepcopy(graph_data) # just save it for just in case (debugging purpose)
         

         # ------------------------------------------------------------------------------------------------------------------------------
         # for just in case ; might not be needed 
         file_node_tensor = torch.tensor([1, 0, 0, 0, 0])
         reg_node_tensor = torch.tensor([0, 1, 0, 0, 0])
         net_node_tensor = torch.tensor([0, 0, 1, 0, 0])
         proc_node_tensor = torch.tensor([0, 0, 0, 1, 0])
         thread_node_tensor = torch.tensor([0, 0, 0, 0, 1])

         file_node_indices = torch.nonzero(torch.all(torch.eq( graph_data.x, file_node_tensor), dim=1), as_tuple=False).flatten().tolist()
         reg_node_indices = torch.nonzero(torch.all(torch.eq( graph_data.x, reg_node_tensor), dim=1), as_tuple=False).flatten().tolist()
         net_node_indices = torch.nonzero(torch.all(torch.eq( graph_data.x, net_node_tensor), dim=1), as_tuple=False).flatten().tolist()
         proc_node_indices = torch.nonzero(torch.all(torch.eq( graph_data.x, proc_node_tensor), dim=1), as_tuple=False).flatten().tolist()
         thread_node_indices = torch.nonzero(torch.all(torch.eq( graph_data.x, thread_node_tensor), dim=1), as_tuple=False).flatten().tolist()
         # ------------------------------------------------------------------------------------------------------------------------------

         ''' 1. Initialize each node's feature vector '''        
         edge_feat_len = len(graph_data.edge_attr[0][:-1]) # drop time-scalar for now
         edge_feat_placeholders = torch.zeros(graph_data.x.shape[0], edge_feat_len) # create a zero-tensors to concat
         graph_data.x = torch.cat((graph_data.x, edge_feat_placeholders), dim=1) # concatenate graph_data.x with zero tensors (~edge-feature placehodler) along the second dimension

         '''
            # 2. For "n_hops": 
            # 3.    For each node:
            # 4.        Get messages (concat of edge-feat and node-feat) from the node's neighbors (neighbor-nodes with 'incoming edges')
            #           ^-- Might have to handle the "duplicate neighboring nodes" problem due to multi-graph ; could utilize approach used in signal-amplification    
            # 5.        Aggregate messages from neighbors based on "neighborhood_aggr" 
            # 6.        Update the node's embedding (* Change made to other nodes during this hop must not be reflected; 
            #                                          So deep-copy a graph in advance that is not mutated but just used for collecting messages)
            #                                       (* Most basic way for the 'update' is to just 'add' the aggregated-message as node's current embedding )                   
         '''
         # JY @ 2023-12-28: Quite slow because doing 1 node by each, unlike in pytorch where parallel tensor operations are performed 
         for n in range(n_hops):
            
            print(f"{n+1} hop", flush = True)

            graph_data__from_previous_hop = copy.deepcopy(graph_data) # Need a graph_data that does not mutate,
                                                                      # (for 'retrieving messages' purposes)
                                                                      # as node's embedding should be updated by the
                                                                      # state of the graph of previous hop,
                                                                      # not by the graph that's being updated real-time in this hop.
            
            graph_data_x_first_5_bits = graph_data__from_previous_hop.x[:,:5] # corresponds to node-attributes 
            
            
            for node_idx in range( graph_data.x.shape[0] ):
               print(f"{cnt} / {len(dataset)}: {graph_data.name} | {n+1} hop | {node_idx+1} / {graph_data.x.shape[0]} : message passing for node", flush = True)

               ''' Get Messages '''
               # this graph's edge-target indices vector
               edge_tar_indices = graph_data.edge_index[1]
               
               # edge-indices of incoming-edges to this node   # "torch.nonzero" returns a 2-D tensor where each row is the index for a nonzero value.
               incoming_edges_to_this_node_idx = torch.nonzero( edge_tar_indices == node_idx ).flatten()  

               # ---------------------------------------------------------------------------------------------
               # edge-attributes of incoming edges to this node (i.e. edge-level messages)
               edge_level_messages = graph_data__from_previous_hop.edge_attr[incoming_edges_to_this_node_idx][:,:-1]  # drop the time-scalar            

               # ---------------------------------------------------------------------------------------------
               # Following is for handling "duplicate neighboring nodes" problem due to multi-graph
               # - Find unique column-wise pairs (dropping duplicates - src/dst pairs that come from multi-graph)
               unique_incoming_edges_to_this_node, _ = torch.unique( graph_data.edge_index[:, incoming_edges_to_this_node_idx ], 
                                                                      dim=1, return_inverse=True)

               source_nodes_of_incoming_edges_to_this_node = unique_incoming_edges_to_this_node[0] # edge-src is index 0

               unique__source_nodes_of_incoming_edges_to_this_node = torch.unique( source_nodes_of_incoming_edges_to_this_node )
               # node-attributes of 'unique' (for handling duplicate neighboring-nodes) incoming nodes
               # (i.e. node-level messages) # 
               node_level_messages = graph_data_x_first_5_bits[ unique__source_nodes_of_incoming_edges_to_this_node ]

               ''' First perform neighborhood aggregation for edge-feats and node-feats separately,
                   as the latter has considered unique incoming nodes to evade "duplicate neighboring nodes"
                   
                   Next, combine the separately aggregated node and edge level messages 
               '''

               if neighborhood_aggr == "sum": neighborhood_aggr__func = torch.sum
               elif neighborhood_aggr == "mean": neighborhood_aggr__func = torch.mean

               node_level_messages__aggregated = neighborhood_aggr__func(node_level_messages, dim = 0)
               edge_level_messages__aggregated = neighborhood_aggr__func(edge_level_messages, dim = 0)

               messages__aggregated = torch.cat([node_level_messages__aggregated, edge_level_messages__aggregated], dim = 0) # node-feat should come first

               ''' Update the node's embedding with the aggregated message
                   (* Most basic way for the 'update' is to just 'add' the aggregated-message as node's current embedding )
                   (  Perhaps could consider updating with some weights? ) 
               '''
               # Need to update with "graph_data" since this will be the next iteration's "graph_data__from_previous_hop"
               # (i.e. "graph_data" is the graph that is mutating and the medium for information-propagation )
               # For "SimpleConv", similar to edge-weight being considered as 1
               graph_data.x[node_idx] = graph_data.x[node_idx] + messages__aggregated


         ''' 7. Since message-passing is done, "pool" all node's embedding, to generate the "graph embedding". '''
         if pool == "sum": pool__func = torch.sum
         elif pool == "mean": pool__func = torch.mean

         graph_embedding = pool__func(graph_data.x, dim = 0)

         data_dict[ re.search(r'Processed_SUBGRAPH_P3_(.*)\.pickle', graph_data.name).group(1) ] = graph_embedding.tolist()

   return data_dict



#**********************************************************************************************************************************************************************
#**********************************************************************************************************************************************************************
#**********************************************************************************************************************************************************************

if __name__ == '__main__':


    parser = ArgumentParser()
   
    # ==================================================================================================================================
    parser.add_argument('-k', '--K', nargs = 1, type = int, default = [10])  

    model_cls_map = {"RandomForest": RandomForestClassifier, "XGBoost": GradientBoostingClassifier,
                     "LogisticRegression": LogisticRegression, "SVM": svm } 
    parser.add_argument('-mod_cls', '--trad_model_cls', nargs = 1, type = str, 
                        default = ["RandomForest"] )

    parser.add_argument('-data', '--dataset', 
                        choices= ['Dataset-Case-1', 'Dataset-Case-2'], 
                        default = ["Dataset-Case-2"])


    parser.add_argument('-graphemb_opt', '--graph_embedding_option', 
                        
                        choices= [
                                  'standard_message_passing_graph_embedding', # Added by JY @ 2023-12-27
                                  # vs. (for now)
                                  'no_graph_structure__event_1gram_nodetype_5bit', 
                                  'no_graph_structure__event_1gram',
                                  'no_graph_structure__event_1gram_nodetype_5bit_and_Ahoc_Identifier',
                                  ], 

                                  default = ["standard_message_passing_graph_embedding"])

    # ---------------------------------------------------------------------------------------------------

    parser.add_argument('-ss_opt', '--search_space_option', 
                        choices= [ 
                                 # defaults
                                 "XGBoost_default_hyperparam",
                                 "RandomForest_default_hyperparam",                                  

                                 "XGBoost_searchspace_1",
                                 "RandomForest_searchspace_1",

                                 # Best tuned of following -------------------------------------------
                                 "Best_RF__Dataset_Case_1__1hops__sum_aggr__sum_pool__2023_12_29_060125", # running
                                 "Best_RF__Dataset_Case_1__2hops__sum_aggr__sum_pool__2023_12_29_055515", # running
                                 "Best_RF__Dataset_Case_1__3hops__sum_aggr__sum_pool__2023_12_28_225029", # running

                                 "Best_RF__Dataset_Case_2__1hops__sum_aggr__sum_pool__2023_12_29_060149", # running
                                 "Best_RF__Dataset_Case_2__2hops__sum_aggr__sum_pool__2023_12_29_055539", # running
                                 "Best_RF__Dataset_Case_2__3hops__sum_aggr__sum_pool__2023_12_28_225047", # running
                                  ], 
                                  default = ["Best_RF__Dataset_Case_2__3hops__sum_aggr__sum_pool__2023_12_28_225047"])
   
    parser.add_argument("--search_on_train__or__final_test", 
                                 
                         choices= ["search_on_train", "final_test", "search_on_all"],  # TODO PW:use "final_test" on test dataset #PW: serach on all- more robust, --> next to run                                  
                         default = ["final_test"] )


    # --------- specific to standard-message-passing 
    parser.add_argument('-n', '--n_hops',  nargs = 1, type = int, 
                        default = [3])

    parser.add_argument('-aggr', '--neighborhood_aggregation', 
                        choices= ['sum', 'mean' ],  # mean 도 해봐라 
                        default = ["sum"])

    parser.add_argument('-pool_opt', '--pool_option', 
                        choices= ['sum', 'mean' ],  # mean 도 해봐라 
                        default = ["sum"])
    # --------------------------------------------------
   
   # ==================================================================================================================================

    # cmd args
    K = parser.parse_args().K[0]
    model_cls = model_cls_map[ parser.parse_args().trad_model_cls[0] ]
    dataset_choice = parser.parse_args().dataset[0]

    n_hops = parser.parse_args().n_hops[0]
    neighborhood_aggregation = parser.parse_args().neighborhood_aggregation[0]
    pool_option = parser.parse_args().pool_option[0]

    graph_embedding_option = parser.parse_args().graph_embedding_option[0]
    search_space_option = parser.parse_args().search_space_option[0]
    search_on_train__or__final_test = parser.parse_args().search_on_train__or__final_test[0] 

    # -----------------------------------------------------------------------------------------------------------------------------------
    model_cls_name = re.search(r"'(.*?)'", str(model_cls)).group(1)


    if search_on_train__or__final_test in {"search_on_train", "search_on_all"}:

       if graph_embedding_option == "standard_message_passing_graph_embedding":
         run_identifier = f"{model_cls_name}__{dataset_choice}__{search_space_option}__{K}_FoldCV__{search_on_train__or__final_test}__{graph_embedding_option}__{n_hops}hops__{neighborhood_aggregation}_aggr__{pool_option}_pool__{datetime.now().strftime('%Y-%m-%d_%H%M%S')}"
       else:
         run_identifier = f"{model_cls_name}__{dataset_choice}__{search_space_option}__{K}_FoldCV__{search_on_train__or__final_test}__{graph_embedding_option}__{datetime.now().strftime('%Y-%m-%d_%H%M%S')}"  
       this_results_dirpath = f"/data/d1/jgwak1/tabby/graph_embedding_improvement_JY_git/graph_embedding_improvement_efforts/Trial_3__Standard_Message_Passing/RESULTS/{run_identifier}"
       experiment_results_df_fpath = os.path.join(this_results_dirpath, f"{run_identifier}.csv")
       if not os.path.exists(this_results_dirpath):
           os.makedirs(this_results_dirpath)


    if search_on_train__or__final_test == "final_test":
       if graph_embedding_option == "standard_message_passing_graph_embedding":
         run_identifier = f"{model_cls_name}__{dataset_choice}__{search_space_option}__{search_on_train__or__final_test}__{graph_embedding_option}__{n_hops}hops__{neighborhood_aggregation}_aggr__{pool_option}_pool__{datetime.now().strftime('%Y-%m-%d_%H%M%S')}"
       else:
         run_identifier = f"{model_cls_name}__{dataset_choice}__{search_space_option}__{search_on_train__or__final_test}__{graph_embedding_option}__{datetime.now().strftime('%Y-%m-%d_%H%M%S')}"  


       this_results_dirpath = f"/data/d1/jgwak1/tabby/graph_embedding_improvement_JY_git/graph_embedding_improvement_efforts/Trial_3__Standard_Message_Passing/RESULTS/{run_identifier}"
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
         "/data/d1/jgwak1/tabby/SILKETW_DATASET_NEW/Silketw_benign_train_test_data_case1/offline_train/Processed_Benign_ONLY_TaskName_edgeattr",
      # ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
      # Dataset-2 (B#662, M#628)
      "Dataset-Case-2": \
         "/data/d1/jgwak1/tabby/SILKETW_DATASET_NEW/Silketw_benign_train_test_data_case1_case2/offline_train/Processed_Benign_ONLY_TaskName_edgeattr"
    }
    projection_datapath_Malware_Train_dict = {
      # Dataset-1 (B#288, M#248) ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
      "Dataset-Case-1": \
         "/data/d1/jgwak1/tabby/SILKETW_DATASET_NEW/Silketw_malware_train_test_data_case1/offline_train/Processed_Malware_ONLY_TaskName_edgeattr",
      # ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
      # Dataset-2 (B#662, M#628)
      "Dataset-Case-2": \
         "/data/d1/jgwak1/tabby/SILKETW_DATASET_NEW/Silketw_malware_train_test_data_case1_case2/offline_train/Processed_Malware_ONLY_TaskName_edgeattr"
    }
    projection_datapath_Benign_Test_dict = {
      # Dataset-1 (B#73, M#62) ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
      "Dataset-Case-1": \
         "/data/d1/jgwak1/tabby/SILKETW_DATASET_NEW/Silketw_benign_train_test_data_case1/offline_test/Processed_Benign_ONLY_TaskName_edgeattr",
      # ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
      # Dataset-2 (B#167, M#158)
      "Dataset-Case-2": \
         "/data/d1/jgwak1/tabby/SILKETW_DATASET_NEW/Silketw_benign_train_test_data_case1_case2/offline_test/Processed_Benign_ONLY_TaskName_edgeattr"
    }
    projection_datapath_Malware_Test_dict = {
      # Dataset-1 (B#73, M#62) ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
      "Dataset-Case-1": \
         "/data/d1/jgwak1/tabby/SILKETW_DATASET_NEW/Silketw_malware_train_test_data_case1/offline_test/Processed_Malware_ONLY_TaskName_edgeattr",
      # ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
      # Dataset-2 (B#167, M#158)
      "Dataset-Case-2": \
         "/data/d1/jgwak1/tabby/SILKETW_DATASET_NEW/Silketw_malware_train_test_data_case1_case2/offline_test/Processed_Malware_ONLY_TaskName_edgeattr"
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



    def RandomForest_searchspace_1() -> dict :

      # https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier
      # Following are typical ranges of RadnomForest hyperparameters

      search_space = dict()
      search_space['n_estimators'] = [ 100, 200, 300, 500 ] # The number of decision trees in the random forest. Typical range: 50 to 500 or more. More trees generally lead to better performance, but there are diminishing returns beyond a certain point.
      search_space['criterion'] = [ 'gini' ] # The function used to measure the quality of a split. Typical options: 'gini' for Gini impurity and 'entropy' for information gain. Gini is the default and works well in most cases.
      search_space['max_depth'] = [ None, 3,6,9,15,20 ]  # The maximum depth of each decision tree. Typical range: 3 to 20. Setting it to None allows nodes to expand until all leaves are pure or contain less than 
      search_space['min_samples_split'] = [ 2, 5, 10, 15 ] # The minimum number of samples required to split an internal node. Typical range: 2 to 20.
      search_space['min_samples_leaf'] = [ 1, 3, 5, 10 ]  # The minimum number of samples required to be at a leaf node. Typical range: 1 to 10.
      search_space['max_features'] = [ 'sqrt', 'log2', None ]  # The number of features to consider when looking for the best split. Typical options: 'sqrt' (square root of the total number of features) or 'log2' (log base 2 of the total number of features).
      search_space['bootstrap'] = [ True, False ] # Whether to use bootstrapped samples when building trees. 
                                           # Typical options: True or False. 
                                           # Setting it to True enables bagging, which is the "standard approach."

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

      # random.shuffle(manual_space) # For random-gridsearch
      return manual_space

    def XGBoost_searchspace_1() -> dict :

      # https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html

      # Following are typical ranges of XGBoost hyperparameters
      search_space = dict()
      search_space['n_estimators'] = [ 100, 500, 1000 ] # The number of boosting stages (i.e., the number of weak learners or decision trees). Typical range: 50 to 2000 or more.
      search_space['learning_rate'] = [ 0.1, 0.01, 0.001 ] # Shrinks the contribution of each tree, helping to prevent overfitting. Typical range: 0.001 to 0.1.      
      search_space['max_depth'] = [ 3, 6, 9 ]  # The maximum depth of each decision tree. Typical range: 3 to 10. Can use None for no limit on the depth, but this can lead to overfitting
      search_space['min_samples_split'] = [ 2, 9, 16 ] # The minimum number of samples required to split an internal node. Typical range: 2 to 20.
      search_space['min_samples_leaf'] = [ 1, 3, 5 ]  # The minimum number of samples required to be at a leaf node. Typical range: 1 to 10.

      # fully enjoy XG-Boost
      search_space['subsample'] = [ 1.0, 0.75, 0.5 ] # The fraction of samples used for fitting the individual trees. Typical range: 0.5 to 1.0.
      search_space['max_features'] = [ None, 'sqrt','log2' ]  # The number of features to consider when looking for the best split. Typical values: 'sqrt' (square root of the total number of features) or 'log2' (log base 2 of the total number of features).
      search_space['loss'] = [ 'log_loss', ]  # The number of features to consider when looking for the best split. Typical values: 'sqrt' (square root of the total number of features) or 'log2' (log base 2 of the total number of features).

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
                      for loss in search_space['loss']:
                        for subsample in search_space['subsample']:
                           for max_features in search_space['max_features']:
                                 
                                 for random_state in search_space["random_state"]:
                                     for split_shuffle_seed in search_space["split_shuffle_seed"]:                        
                                          print(f"appending {[n_estimators, learning_rate, max_depth, min_samples_split, min_samples_leaf, loss, subsample, max_features, random_state, split_shuffle_seed ]}", flush = True )
                                          
                                          manual_space.append(
                                                   {
                                                   'n_estimators': n_estimators, 
                                                   'learning_rate': learning_rate, 
                                                   'max_depth': max_depth, 
                                                   'min_samples_split': min_samples_split, 
                                                   'min_samples_leaf': min_samples_leaf, 
                                                   'loss': loss,
                                                   'subsample': subsample, 
                                                   'max_features': max_features,
                                                   'random_state': random_state,
                                                   'split_shuffle_seed': split_shuffle_seed
                                                   }
                                                   )

      # random.shuffle(manual_space) # For random-gridsearch
      return manual_space


    # Best found hyperparameter sets ===================================================================================================================================================

    # "sklearn.ensemble._forest.RandomForestClassifier__Dataset-Case-1__RandomForest_searchspace_1__10_FoldCV__search_on_train__standard_message_passing_graph_embedding__1hops__sum_aggr__sum_pool__2023-12-29_060125",
    def Best_RF__Dataset_Case_1__1hops__sum_aggr__sum_pool__2023_12_29_060125():
         manual_space = []
         manual_space.append(
               {'bootstrap': False,
               'criterion': 'gini',
               'max_depth': 15.0,
               'max_features': 'log2',
               'min_samples_leaf': 1,
               'min_samples_split': 2,
               'n_estimators': 300,
               'random_state': 42,
               'split_shuffle_seed': 100}
         )
         return manual_space      

    # "sklearn.ensemble._forest.RandomForestClassifier__Dataset-Case-1__RandomForest_searchspace_1__10_FoldCV__search_on_train__standard_message_passing_graph_embedding__2hops__sum_aggr__sum_pool__2023-12-29_055515",
    def Best_RF__Dataset_Case_1__2hops__sum_aggr__sum_pool__2023_12_29_055515():
         manual_space = []
         manual_space.append(
               {'bootstrap': False,
               'criterion': 'gini',
               'max_depth': 15.0,
               'max_features': 'sqrt',
               'min_samples_leaf': 1,
               'min_samples_split': 5,
               'n_estimators': 100,
               'random_state': 0,
               'split_shuffle_seed': 100}         
         )
         return manual_space      
                                 
    # "sklearn.ensemble._forest.RandomForestClassifier__Dataset-Case-1__RandomForest_searchspace_1__10_FoldCV__search_on_train__standard_message_passing_graph_embedding__3hops__sum_aggr__sum_pool__2023-12-28_225029",
    def Best_RF__Dataset_Case_1__3hops__sum_aggr__sum_pool__2023_12_28_225029():
         manual_space = []
         manual_space.append(
            {'bootstrap': False,
            'criterion': 'gini',
            'max_depth': 15.0,
            'max_features': 'sqrt',
            'min_samples_leaf': 1,
            'min_samples_split': 2,
            'n_estimators': 200,
            'random_state': 42,
            'split_shuffle_seed': 100}
         )
         return manual_space      

    # "sklearn.ensemble._forest.RandomForestClassifier__Dataset-Case-2__RandomForest_searchspace_1__10_FoldCV__search_on_train__standard_message_passing_graph_embedding__1hops__sum_aggr__sum_pool__2023-12-29_060149",
    def Best_RF__Dataset_Case_2__1hops__sum_aggr__sum_pool__2023_12_29_060149():
         manual_space = []
         manual_space.append(
            {'bootstrap': False,
            'criterion': 'gini',
            'max_depth': None,
            'max_features': 'sqrt',
            'min_samples_leaf': 1,
            'min_samples_split': 2,
            'n_estimators': 300,
            'random_state': 99,
            'split_shuffle_seed': 100}         
         )
         return manual_space      

    # "sklearn.ensemble._forest.RandomForestClassifier__Dataset-Case-2__RandomForest_searchspace_1__10_FoldCV__search_on_train__standard_message_passing_graph_embedding__2hops__sum_aggr__sum_pool__2023-12-29_055539",
    def Best_RF__Dataset_Case_2__2hops__sum_aggr__sum_pool__2023_12_29_055539():
         manual_space = []
         manual_space.append(
            {'bootstrap': True,
            'criterion': 'gini',
            'max_depth': None,
            'max_features': None,
            'min_samples_leaf': 1,
            'min_samples_split': 2,
            'n_estimators': 200,
            'random_state': 0,
            'split_shuffle_seed': 100}         
         )
         return manual_space      


    # "sklearn.ensemble._forest.RandomForestClassifier__Dataset-Case-2__RandomForest_searchspace_1__10_FoldCV__search_on_train__standard_message_passing_graph_embedding__3hops__sum_aggr__sum_pool__2023-12-28_225047",
    def Best_RF__Dataset_Case_2__3hops__sum_aggr__sum_pool__2023_12_28_225047():
         manual_space = []
         manual_space.append(
            {'bootstrap': True,
            'criterion': 'gini',
            'max_depth': 15.0,
            'max_features': None,
            'min_samples_leaf': 1,
            'min_samples_split': 2,
            'n_estimators': 100,
            'random_state': 0,
            'split_shuffle_seed': 100}
         )
         return manual_space      

    # ====================================================================================================================================================================================




    ####################################################################################################################################################
    # defaults
    if search_space_option == "XGBoost_default_hyperparam": search_space = XGBoost_default_hyperparam()   
    elif search_space_option == "RandomForest_default_hyperparam": search_space = RandomForest_default_hyperparam()   

    # extensive search spaces 
    elif search_space_option == "XGBoost_searchspace_1": search_space = XGBoost_searchspace_1()   
    elif search_space_option == "RandomForest_searchspace_1": search_space = RandomForest_searchspace_1()   

    # best found
    elif search_space_option == "Best_RF__Dataset_Case_1__1hops__sum_aggr__sum_pool__2023_12_29_060125":
         search_space = Best_RF__Dataset_Case_1__1hops__sum_aggr__sum_pool__2023_12_29_060125()

    elif search_space_option == "Best_RF__Dataset_Case_1__2hops__sum_aggr__sum_pool__2023_12_29_055515":
         search_space = Best_RF__Dataset_Case_1__2hops__sum_aggr__sum_pool__2023_12_29_055515()
    
    elif search_space_option == "Best_RF__Dataset_Case_1__3hops__sum_aggr__sum_pool__2023_12_28_225029":
         search_space = Best_RF__Dataset_Case_1__3hops__sum_aggr__sum_pool__2023_12_28_225029()

    elif search_space_option == "Best_RF__Dataset_Case_2__1hops__sum_aggr__sum_pool__2023_12_29_060149":
         search_space = Best_RF__Dataset_Case_2__1hops__sum_aggr__sum_pool__2023_12_29_060149()

    elif search_space_option == "Best_RF__Dataset_Case_2__2hops__sum_aggr__sum_pool__2023_12_29_055539":
         search_space = Best_RF__Dataset_Case_2__2hops__sum_aggr__sum_pool__2023_12_29_055539()

    elif search_space_option == "Best_RF__Dataset_Case_2__3hops__sum_aggr__sum_pool__2023_12_28_225047":
         search_space = Best_RF__Dataset_Case_2__3hops__sum_aggr__sum_pool__2023_12_28_225047()

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

   
    # Added by JY @ 2023-12-27
    if graph_embedding_option == "standard_message_passing_graph_embedding":
        train_dataset__standard_message_passing_dict = get__standard_message_passing_graph_embedding__dict( dataset= train_dataset,
                                                                                                            n_hops= n_hops,
                                                                                                            neighborhood_aggr= neighborhood_aggregation,
                                                                                                            pool= pool_option )
        nodetype_names = ["file", "registry", "network", "process", "thread"] 
        feature_names = nodetype_names + taskname_colnames # yes this order is correct
        X = pd.DataFrame(train_dataset__standard_message_passing_dict).T
    else:

         if graph_embedding_option == "no_graph_structure__event_1gram_nodetype_5bit":
               train_dataset__no_graph_structure_dict = get_No_Graph_Structure_eventdist_nodetype5bit_dist_dict( dataset= train_dataset )  
               nodetype_names = ["file", "registry", "network", "process", "thread"] # this is the correct-order
               feature_names = taskname_colnames + nodetype_names

         elif graph_embedding_option == "no_graph_structure__event_1gram":
               train_dataset__no_graph_structure_dict = get_No_Graph_Structure_eventdist_dict( dataset= train_dataset )  
               feature_names = taskname_colnames

         elif graph_embedding_option == "no_graph_structure__event_1gram_nodetype_5bit_and_Ahoc_Identifier":
               train_dataset__no_graph_structure_dict = get_No_Graph_Structure_eventdist_nodetype5bit_adhoc_identifier_dist_dict( dataset= train_dataset ) 
               nodetype_names = ["file", "registry", "network", "process", "thread"] # this is the correct-order
               feature_names = taskname_colnames + nodetype_names + [f"adhoc_pattern_{i}" for i in range(len(X.columns) - len(taskname_colnames) - len(nodetype_names))]

         else:
               ValueError(f"Invalid graph_embedding_option ({graph_embedding_option})")                  

         X = pd.DataFrame(train_dataset__no_graph_structure_dict).T


    X.columns = feature_names
    X.reset_index(inplace = True)
    X.rename(columns = {'index':'data_name'}, inplace = True)

    y = [data.y for data in train_dataset]

    #--------------------------------------------------------------------------
   #  # Now apply readout to the obtained thread-level vectors
   #  if "signal_amplified" in signal_amplification_option:
   #        train_dataset_signal_amplified_readout_df = get_readout_applied_df(data_dict = train_dataset__standard_message_passing_dict, 
   #                                                                           readout_option= readout_option,
   #                                                                           signal_amplification_option= signal_amplification_option)
   #        X = train_dataset_signal_amplified_readout_df
   #        X.reset_index(inplace = True)
   #        X.rename(columns = {'thread_is_from':'data_name'}, inplace = True)
   #        y = [data.y for data in train_dataset]
    
    # =================================================================================================================================================
    # =================================================================================================================================================

    if search_on_train__or__final_test == "final_test":
        # Also prepare for final-test dataset, to later test the best-fitted models on test-set
         # Now apply signal-amplification here (here least conflicts with existing code.)

         if graph_embedding_option == "standard_message_passing_graph_embedding":
            final_test_dataset__standard_message_passing_dict = get__standard_message_passing_graph_embedding__dict( dataset= final_test_dataset,
                                                                                                                     n_hops= n_hops,
                                                                                                                     neighborhood_aggr= neighborhood_aggregation,
                                                                                                                     pool= pool_option )
            nodetype_names = ["file", "registry", "network", "process", "thread"] 
            feature_names = nodetype_names + taskname_colnames # yes this order is correct
            final_test_X = pd.DataFrame(train_dataset__standard_message_passing_dict).T


         elif graph_embedding_option == "no_graph_structure__event_1gram_nodetype_5bit":
               final_test_dataset__no_graph_structure_dict = get_No_Graph_Structure_eventdist_nodetype5bit_dist_dict( dataset= final_test_dataset )  
               nodetype_names = ["file", "registry", "network", "process", "thread"] # this is the correct-order
               feature_names = taskname_colnames + nodetype_names

         elif graph_embedding_option == "no_graph_structure__event_1gram":
               final_test_dataset__no_graph_structure_dict = get_No_Graph_Structure_eventdist_dict( dataset= final_test_dataset )  
               feature_names = taskname_colnames

         elif graph_embedding_option == "no_graph_structure__event_1gram_nodetype_5bit_and_Ahoc_Identifier":
               final_test_dataset__no_graph_structure_dict = get_No_Graph_Structure_eventdist_nodetype5bit_adhoc_identifier_dist_dict( dataset= final_test_dataset ) 
               nodetype_names = ["file", "registry", "network", "process", "thread"] # this is the correct-order
               feature_names = taskname_colnames + nodetype_names + [f"adhoc_pattern_{i}" for i in range(len(X.columns) - len(taskname_colnames) - len(nodetype_names))]

         else:
               ValueError(f"Invalid graph_embedding_option ({graph_embedding_option})")                  

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
                  'xgboost' in model_cls_name.lower():

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
                  'randomforest' in model_cls_name.lower():
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
                  'logistic' in model_cls_name.lower():
                  model = model_cls()

               elif model_cls_name == 'sklearn.svm' and\
                  'svm' in model_cls_name:
                  model = model_cls()

               else:
                  print(model_cls_name, flush = True)
                  ValueError(f"{model_cls_name} is not supported")



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
               if model_cls_name == 'sklearn.ensemble._gb.GradientBoostingClassifier' or\
                  'xgboost' in model_cls_name.lower():

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


               elif model_cls_name == 'sklearn.ensemble._forest.RandomForestClassifier' or\
                    'randomforest' in model_cls_name.lower():
                    
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


               elif model_cls_name == 'sklearn.linear_model._logistic.LogisticRegression' or\
                  'logistic' in model_cls_name.lower():
                  model = model_cls()

               elif model_cls_name == 'sklearn.svm' or\
                  'svm' in model_cls_name.lower():
                  model = model_cls()

               else:
                  print(model_cls_name, flush = True)
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


               # Added by JY @ 2023-12-21
               Predict_proba_dict = dict()
               final_test_X___ = final_test_X.set_index("data_name")
               for data_name, data in list(zip(final_test_X___.index, final_test_X___.values)):
                  # prints out "X does not have valid feature names, but RandomForestClassifier was fitted with feature names" which is FINE
                  # correctness checked
                  Predict_proba_dict[data_name] = model.predict_proba(data.reshape(1,-1)).tolist()



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
                              misprediction_subgraph_names = misprediction_subgraph_names,
                              Predict_proba_dict = Predict_proba_dict
                              )