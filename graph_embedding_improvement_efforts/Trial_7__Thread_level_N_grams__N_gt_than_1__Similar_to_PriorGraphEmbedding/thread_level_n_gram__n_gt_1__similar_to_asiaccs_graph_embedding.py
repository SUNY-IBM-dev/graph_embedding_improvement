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

sys.path.append("/data/d1/jgwak1/tabby/graph_embedding_improvement_JY_git/graph_embedding_improvement_efforts/Trial_7__Thread_level_N_grams__N_gt_than_1__Similar_to_PriorGraphEmbedding/source")
sys.path.append("/home/jgwak1/tabby/graph_embedding_improvement_JY_git/graph_embedding_improvement_efforts/Trial_7__Thread_level_N_grams__N_gt_than_1__Similar_to_PriorGraphEmbedding/source")



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


from sklearn.feature_extraction.text import CountVectorizer


# TODO: Consider turning tensor operations to gpu-operations, for faster
# https://pytorch.org/docs/stable/tensors.html
# cuda0 = torch.device('cuda:0')
# >>> torch.ones([2, 4], dtype=torch.float64, device=cuda0)
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

      Global_Important_Features_Test_dataset = Global_Important_Features_Test_dataset.assign(SUM = Global_Important_Features_Test_dataset.sum(axis=1)) # SUM column

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

      ''' Added by JY @ 2024-1-10 for feature-value-level local explanation-comparison (for futher analysis of feature-value level patterns in malware vs. benign)'''
      Local_SHAP_values_Test_dataset = pd.DataFrame(columns = Test_dataset.columns)

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
               try:
                  waterfallplot_out.savefig( os.path.join(Mispredictions_WaterfallPlots_subdirpath, f"{N}-gram SHAP_local_interpretability_waterfall_plot_{Test_SG_name}.png") )
               except:
                  waterfallplot_out.figure.savefig( os.path.join(Mispredictions_WaterfallPlots_subdirpath, f"{N}-gram SHAP_local_interpretability_waterfall_plot_{Test_SG_name}.png") )
                  
            else:
               try:
                  waterfallplot_out.savefig( os.path.join(Correct_Predictions_WaterfallPlots_subdirpath, f"{N}-gram SHAP_local_interpretability_waterfall_plot_{Test_SG_name}.png") )
               except: # in felis
                   waterfallplot_out.figure.savefig( os.path.join(Correct_Predictions_WaterfallPlots_subdirpath, f"{N}-gram SHAP_local_interpretability_waterfall_plot_{Test_SG_name}.png") )

            # Added by JY @ 2023-12-21 : For local shap of sample (SUM of all feature's shap values for the sample)
            #                                                      ^-- corresponds to "f(x)" in Waterfall plot
            Test_SG_Local_Shap = sum(shap_values)
            Global_Important_Features_Test_dataset.loc[Test_SG_name,'SHAP_sum_of_feature_shaps'] = Test_SG_Local_Shap
            Global_Important_Features_Test_dataset.loc[Test_SG_name, 'SHAP_base_value'] = base_value


            ''' Added by JY @ 2024-1-10 for feature-value-level local explanation-comparison (for futher analysis of feature-value level patterns in malware vs. benign)'''
            # class-information? do it later in another file
            Local_SHAP_values_Test_dataset = pd.concat([ Local_SHAP_values_Test_dataset, pd.DataFrame([ dict(zip(Test_dataset.columns, shap_values)) | {'data_name': Test_SG_name} ]) ], 
                                                       axis = 0)



            print(f"{cnt} / {len(Test_SG_names)} : SHAP-local done for {Test_SG_name}", flush=True)

      # added by JY @ 2023-12-21
      def append_prefix(value, prefix): return prefix + value if not value.startswith('malware') else value      

      Global_Important_Features_Test_dataset.index = Global_Important_Features_Test_dataset.index.map(lambda x: append_prefix(x, "benign_"))
      Global_Important_Features_Test_dataset.sort_index(inplace=True)

      # Global_Important_Features_Test_dataset['data_name'] = Global_Important_Features_Test_dataset['data_name'].apply(lambda x: append_prefix(x, "benign_"))
      # Global_Important_Features_Test_dataset.sort_values(by = "data_name", inplace = True)
      # Global_Important_Features_Test_dataset.set_index("data_name", inplace = True)

      Global_Important_Features_Test_dataset.to_csv(os.path.join(Explanation_Results_save_dirpath, f"{model_cls_name} {N}-gram Global-SHAP Important FeatureNames Test-Dataset.csv"))

      ''' Added by JY @ 2024-1-10 for feature-value-level local explanation-comparison (for futher analysis of feature-value level patterns in malware vs. benign)
          Note that here, negative SHAP values are ones that push towards benign-prediction
      '''
      Local_SHAP_values_Test_dataset.set_index("data_name", inplace = True)
      Local_SHAP_values_Test_dataset.index = Local_SHAP_values_Test_dataset.index.map(lambda x: append_prefix(x, "benign_"))
      Local_SHAP_values_Test_dataset.sort_index(inplace=True)
      Local_SHAP_values_Test_dataset.to_csv(os.path.join(Explanation_Results_save_dirpath, f"{model_cls_name} {N}-gram Local-SHAP values Test-Dataset.csv"))


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
def pretrain__countvectorizer_on_training_set__before_graph_embedding_generation( 
                                                                                 dataset : list, 
                                                                                 Ngram : int = 4,
                                                                                 only_train_specified_Ngram = False,
                                                                                  ) -> list:
    
      ''' TODO
      start writing pretraining a n-gram countvectorizer on all thread-level event-sequences from every subgraph in entire training set, 
      PRIOR to peforming graph-embedding generation (signal amplication) -- 
      this is similar to what has been done in local ngram standard message passing -- 
      and a reasonable compromiziation althoguh in K-fold CV hypeparameter tuning context strictly speaking should be fitting the countvectorizer of K-1 training-folds
         -- but this will be too much

      # refer to : https://github.com/SUNY-IBM-dev/graph_embedding_improvement/blob/baee25391d90c9631e97f38fa84f1e13ba718cf5/graph_embedding_improvement_efforts/Trial_4__Local_N_gram__standard_message_passing/local_ngram__standard_message_passing__graph_embedding.py#L1277
         
      '''

      # JY @ 2024-2-3: Just enable 1-gram
      # if Ngram <= 1:
      #    ValueError("Ngram should be greater than 1 for this graph-embedding approach.\n-> AsiaCCS submission already handled thread-level 1gram event distirubtion")

      pretrained_countvectorizers_list =[]

      thread_node_tensor = torch.tensor([0, 0, 0, 0, 1])


      graph__to__thread_level_sorted_event_sequences_dict = dict() # Added by JY @ 2024-1-20


      cnt = 1
      for data in dataset:
            
            print(f"{cnt} / {len(dataset)}: {data.name}  |  in 'pretrain__countvectorizer_on_training_set__before_graph_embedding_generation()'", flush = True)

            # Added by JY @ 2023-07-18 to handle node-attr dim > 5  
            # if data.x.shape[1] != 5:
            data_x_first5 = data.x[:,:5]

            thread_node_indices = torch.nonzero(torch.all(torch.eq( data_x_first5, thread_node_tensor ), dim=1), as_tuple=False).flatten().tolist()


            graph__to__thread_level_sorted_event_sequences_dict[data.name] = []

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

               edge_attr_of_both_direction_edges_from_thread_node_idx = torch.cat([edge_attr_of_incoming_edges_from_thread_node_idx, 
                                                                                   edge_attr_of_outgoing_edges_from_thread_node_idx],
                                                                                   dim = 0)


               timestamp_sorted_indices = torch.argsort(edge_attr_of_both_direction_edges_from_thread_node_idx[:, -1], descending=False)

               edge_attr_of_both_direction_edges_from_thread_node_idx__sorted = edge_attr_of_both_direction_edges_from_thread_node_idx[ timestamp_sorted_indices ]

               taskname_indices = torch.nonzero(edge_attr_of_both_direction_edges_from_thread_node_idx__sorted[:,:-1], as_tuple=False)[:, -1]


               # Replace tensor elements with corresponding string values efficiently
               thread_sorted_event_sequence = [taskname_colnames[i] for i in taskname_indices]
               graph__to__thread_level_sorted_event_sequences_dict[data.name].append( thread_sorted_event_sequence )
   
               # --------------------------------------------------------------------------------------------------------------------------------------------------------------------------

            # data_dict[ data.name.lstrip("Processed_SUBGRAPH_P3_").rstrip(".pickle") ] = data_thread_node_both_direction_edges_edge_attrs.tolist()
            cnt+=1

      # Now apply countvectorizer
      # https://github.com/SUNY-IBM-dev/graph_embedding_improvement/blob/c14d7631f95a24d5e0c192d7075a184698af1e13/graph_embedding_improvement_efforts/Trial_4__Local_N_gram__standard_message_passing/local_ngram__standard_message_passing__graph_embedding.py#L1277

      thread_level_sorted_event_sequences__nested_list = [ list_of_thread_event_sequences for data_name, list_of_thread_event_sequences in graph__to__thread_level_sorted_event_sequences_dict.items() ]

      def flatten_list(nested_list):
         return [item for sublist in nested_list for item in sublist]

      all_thread_level_sorted_event_lists = flatten_list( thread_level_sorted_event_sequences__nested_list )



      # https://github.com/SUNY-IBM-dev/graph_embedding_improvement/blob/c14d7631f95a24d5e0c192d7075a184698af1e13/N_gram_hyperparameter_tuning/n_gram_hypeparam_tuning.py#L958
      # for compatiblity with countvectorizer
      # [ 'Threadstart Create Write Create Threadstop',
      #   'Threadstart OpenKey Create CloseKey Close Open Threadstop',
      #     ... ]
      if only_train_specified_Ngram:
         # Only fit for the specified Ngram countvectorizer
         #     e.g. If N == 4, then drop all event-sequences that have less than 4 events, 
         #          because since can't fit a 4gram countvectorizer out of such evnet-sequences.
         all_thread_level_sorted_event_str_sequences = \
            [ ' '.join(thread_sorted_event_list) for thread_sorted_event_list in all_thread_level_sorted_event_lists\
              if len(thread_sorted_event_list) >= Ngram ]

         print(f"Specified Ngram: {Ngram} | only_train_specified_Ngram: {only_train_specified_Ngram}", flush=True)
         countvectorizer = CountVectorizer(ngram_range=(Ngram, Ngram),
                                           max_df= 1.0,
                                           min_df= 1,
                                           max_features= None)   
         countvectorizer.fit( all_thread_level_sorted_event_str_sequences )

         print(f"Fitted {Ngram}-gram countvectorizer", flush=True)


         pretrained_countvectorizers_list.append( countvectorizer )


      else:
         # Fit multiple countvectorizers, to account for event-sequences that have less than than the number of specified Ngram
         #     e.g. Observed a good number of threads having small number of events (some have only 1; such as threadstop)
         #          To account for such cases, not only fit countvectorzier for specified Ngram, 
         #          but fit Ngram countvecotrizer, N-1 gram countvecotrizer, .. , 1 gram countvecotrizer.
         print(f"Specified Ngram: {Ngram} | only_train_specified_Ngram: {only_train_specified_Ngram}", flush=True)

         for N in list(range(1, Ngram+1)):

            # if N > 1:
            #    # max_df and min_df here are necessary and values are conventionally used ones
            #    # necessity is because , say, 61 C 4 will result in 61! / (61-4)! * 4! which is unreasonably large, 
            #    # so necesary to filter out features based on frequency 
            #    # intuition is meaningful ngram features should not appear too little or too frequent
            #    countvectorizer = CountVectorizer(ngram_range=(N, N), 
            #                                     max_df= 0.3, 
            #                                     min_df= 10, 
            #                                     max_features= None )  # ngram [use 4-gram or 8-gram] 
            # else:
            #    # N = 1 <-- we don't want to drop any 1-gram features since we don't have feature-space too big problem
            #    #           Following does not ignore any n-gram feautres
            #    countvectorizer = CountVectorizer(ngram_range=(N, N),
            #                                        max_df= 1.0,
            #                                        min_df= 1,
            #                                        max_features= None)


            all_thread_level_sorted_event_str_sequences = \
               [ ' '.join(thread_sorted_event_list) for thread_sorted_event_list in all_thread_level_sorted_event_lists\
               if len(thread_sorted_event_list) >= N ]

            if len(all_thread_level_sorted_event_str_sequences) == 0:
                print(f"None of the thread-level event-sequences have length greater or equal than {N} -- so continue")
                # this is very unlikely though
                continue

            countvectorizer = CountVectorizer(ngram_range=(N, N),
                                              max_df= 1.0,
                                              min_df= 1,
                                              max_features= None)   
            countvectorizer.fit( all_thread_level_sorted_event_str_sequences )
            print(f"Fitted {N}-gram countvectorizer", flush=True)

            pretrained_countvectorizers_list.append( countvectorizer )

      return pretrained_countvectorizers_list







##########################################################################################################################################################
# Signal-Amplification Function (Thread-level Event-Dist. 1gram + Adjacent Node's Node-Type 5Bit)
#PW: Thread node embedding by aggregating one hop neighbour nodes

# JY @ 2024-1-20: thread-level N>1 gram events + nodetype-5bits
def get_thread_level_N_gram_events__nodetype5bit__dict( 
                                                            pretrained_Ngram_countvectorizer_list : list, # TODO        
                                                            dataset : list, 
                                                            pool : str = "sum",

                                                              ):
      
      ''' JY @ 2024-1-20 : Implement this '''


      thread_node_tensor = torch.tensor([0, 0, 0, 0, 1])

      data_dict = dict()

      cnt = 1
      for data in dataset:
            
            print(f"{cnt} / {len(dataset)}: {data.name} -- generate graph-embedding", flush = True)

            # Added by JY @ 2023-07-18 to handle node-attr dim > 5  
            # if data.x.shape[1] != 5:
            data_x_first5 = data.x[:,:5]
            thread_node_indices = torch.nonzero(torch.all(torch.eq( data_x_first5, thread_node_tensor), dim=1), as_tuple=False).flatten().tolist()

            # which this node is a source-node (outgoing-edge w.r.t this node)
  


            data_thread_node_all_unique_adjacent_nodes_5bit_dists = torch.tensor([]) # Added by JY @ 2023-07-19

            data_thread_level_all_Ngram_features = torch.tensor([]) # Added by JY @ 2023-01-20

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

               thread_sorted_event_str_sequence = " ".join(thread_sorted_event_list) # for compabitiblity with countvecotrizer

               # transform and get count -- https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html#sklearn.feature_extraction.text.CountVectorizer.transform
               # Refer to: https://github.com/SUNY-IBM-dev/graph_embedding_improvement/blob/20627016d59466d3dad191ff208efce97b15d35e/graph_embedding_improvement_efforts/Trial_4__Local_N_gram__standard_message_passing/local_ngram__standard_message_passing__graph_embedding.py#L760C41-L760C72

               # Get all Ngrams (N could be multiple, depending on 'only_train_specified_Ngram' parameter value)
               thread_all_Ngram_counts__appended_nparray = np.array([]) # torch.Tensor()
               for pretrained_Ngram_countvectorizer in pretrained_Ngram_countvectorizer_list:
                  # If multiple countvectorizers, supposed to start from 1gram to Ngram
                  #print(f"pretrained_Ngram_countvectorizer.ngram_range: {pretrained_Ngram_countvectorizer.ngram_range}", flush= True)
                  thread_Ngram_counts__portion = pretrained_Ngram_countvectorizer.transform( [ thread_sorted_event_str_sequence ] ).toarray() # needs to be in a list
                  thread_all_Ngram_counts__appended_nparray = np.append(thread_all_Ngram_counts__appended_nparray, thread_Ngram_counts__portion )           
               #print("\n")

               # JY @ 2024-1-20: so that can stack on 'data_thread_level_all_Ngram_features' for all thread's Ngram features within this subgrpah(data)
               thread_all_Ngram_counts__appended_tensor = torch.Tensor(thread_all_Ngram_counts__appended_nparray).view(1,-1) # for Size([1,edge_feat_len])
               data_thread_level_all_Ngram_features = torch.cat((data_thread_level_all_Ngram_features, thread_all_Ngram_counts__appended_tensor), dim=0)                  



               # ==============================================================================================================================================
               # Node-level

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
               integrated_5bit_of_all_unique_adjacent_nodes_to_thread = torch.sum( data_x_first5[unique_adjacent_nodes_of_both_direction_edges_of_thread_node], dim = 0 )
               
               data_thread_node_all_unique_adjacent_nodes_5bit_dists = torch.cat(( data_thread_node_all_unique_adjacent_nodes_5bit_dists,
                                                                                   integrated_5bit_of_all_unique_adjacent_nodes_to_thread.unsqueeze(0) ), dim = 0)
               # --------------------------------------------------------------------------------------------------------------------------------------------------------------------------


            # [ node-type5bit + N>gram events ]
            thread_level_N_gram_events_adjacent_5bit_dist = torch.cat( [data_thread_node_all_unique_adjacent_nodes_5bit_dists, 
                                                                        data_thread_level_all_Ngram_features, 
                                                                        ], dim = 1)
            
            
            
            ''' JY @ 2024-1-20: Now "pool" all thead-node's embedding, to generate the "graph embedding". '''
            if pool == "sum": pool__func = torch.sum
            elif pool == "mean": pool__func = torch.mean

            graph_embedding = pool__func(thread_level_N_gram_events_adjacent_5bit_dist, dim = 0)            
            
            
            
            data_dict[ re.search(r'Processed_SUBGRAPH_P3_(.*?)\.pickle', data.name).group(1) ] = graph_embedding.tolist()

            # data_dict[ data.name.lstrip("Processed_SUBGRAPH_P3_").rstrip(".pickle") ] = data_thread_node_both_direction_edges_edge_attrs.tolist()
            cnt+=1
      return data_dict

   # -----------------------------------------------------------------------------------------------------------------------------------------------------

# JY @ 2024-1-27: thread-level N>1 gram events + nodetype-5bits + FRNPeventCount
def get_thread_level_N_gram_events__nodetype5bit__FRNPeventCount__dict( 
                                                                          pretrained_Ngram_countvectorizer_list : list, # TODO        
                                                                          dataset : list, 
                                                                          pool : str = "sum",
                                                                        ):
      ''' JY @ 2024-1-27 : Implement this '''


      thread_node_tensor = torch.tensor([0, 0, 0, 0, 1])

      data_dict = dict()

      cnt = 1
      for data in dataset:
            
            print(f"{cnt} / {len(dataset)}: {data.name} -- generate graph-embedding", flush = True)

            # Added by JY @ 2023-07-18 to handle node-attr dim > 5  
            # if data.x.shape[1] != 5:
            data_x_first5 = data.x[:,:5]
            thread_node_indices = torch.nonzero(torch.all(torch.eq( data_x_first5, thread_node_tensor), dim=1), as_tuple=False).flatten().tolist()

            # which this node is a source-node (outgoing-edge w.r.t this node)
  


            data_thread_node_all_unique_adjacent_nodes_5bit_dists = torch.tensor([]) # Added by JY @ 2023-07-19

            data_thread_level_all_Ngram_features = torch.tensor([]) # Added by JY @ 2023-01-20


            data_thread_node___dist_of__types_of_nodes__associated_with_ALL_edges__connected_to__thread_node = torch.tensor([]) # Added by JY @ 2024-01-27




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


               # ==============================================================================================================================================
               # Thread-level N-gram events features
               ''' JY @ 2024-1-20: Get thread-level event-sequence sorted by timestamps '''
               # Refered to : https://github.com/SUNY-IBM-dev/graph_embedding_improvement/blob/20627016d59466d3dad191ff208efce97b15d35e/graph_embedding_improvement_efforts/Trial_7__Thread_level_N_grams__N_gt_than_1__Similar_to_PriorGraphEmbedding/thread_level_n_gram__n_gt_1__similar_to_asiaccs_graph_embedding.py#L483C1-L491C95

               edge_attr_of_both_direction_edges_from_thread_node_idx = torch.cat([edge_attr_of_incoming_edges_from_thread_node_idx, 
                                                                                   edge_attr_of_outgoing_edges_from_thread_node_idx],
                                                                                   dim = 0)


               timestamp_sorted_indices = torch.argsort(edge_attr_of_both_direction_edges_from_thread_node_idx[:, -1], descending=False)

               edge_attr_of_both_direction_edges_from_thread_node_idx__sorted = edge_attr_of_both_direction_edges_from_thread_node_idx[ timestamp_sorted_indices ]

               taskname_indices = torch.nonzero(edge_attr_of_both_direction_edges_from_thread_node_idx__sorted[:,:-1], as_tuple=False)[:, -1]


               thread_sorted_event_list = [taskname_colnames[i] for i in taskname_indices]

               thread_sorted_event_str_sequence = " ".join(thread_sorted_event_list) # for compabitiblity with countvecotrizer

               # transform and get count -- https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html#sklearn.feature_extraction.text.CountVectorizer.transform
               # Refer to: https://github.com/SUNY-IBM-dev/graph_embedding_improvement/blob/20627016d59466d3dad191ff208efce97b15d35e/graph_embedding_improvement_efforts/Trial_4__Local_N_gram__standard_message_passing/local_ngram__standard_message_passing__graph_embedding.py#L760C41-L760C72

               # Get all Ngrams (N could be multiple, depending on 'only_train_specified_Ngram' parameter value)
               thread_all_Ngram_counts__appended_nparray = np.array([]) # torch.Tensor()
               for pretrained_Ngram_countvectorizer in pretrained_Ngram_countvectorizer_list:
                  # If multiple countvectorizers, supposed to start from 1gram to Ngram
                  #print(f"pretrained_Ngram_countvectorizer.ngram_range: {pretrained_Ngram_countvectorizer.ngram_range}", flush= True)
                  thread_Ngram_counts__portion = pretrained_Ngram_countvectorizer.transform( [ thread_sorted_event_str_sequence ] ).toarray() # needs to be in a list
                  thread_all_Ngram_counts__appended_nparray = np.append(thread_all_Ngram_counts__appended_nparray, thread_Ngram_counts__portion )           
               #print("\n")

               # JY @ 2024-1-20: so that can stack on 'data_thread_level_all_Ngram_features' for all thread's Ngram features within this subgrpah(data)
               thread_all_Ngram_counts__appended_tensor = torch.Tensor(thread_all_Ngram_counts__appended_nparray).view(1,-1) # for Size([1,edge_feat_len])
               data_thread_level_all_Ngram_features = torch.cat((data_thread_level_all_Ngram_features, thread_all_Ngram_counts__appended_tensor), dim=0)                  



               # ==============================================================================================================================================
               # Node-level

               # But also need to consider the multi-graph aspect here. 
               # So here, do not count twice for duplicate adjacent nodes due to multigraph.
               # Just need to get the distribution.
               # Find unique column-wise pairs (dropping duplicates - src/dst pairs that come from multi-graph)
               unique_outgoing_edges_from_thread_node, _ = torch.unique( data.edge_index[:, outgoing_edges_from_thread_node_idx ], dim=1, return_inverse=True)

               # Find unique column-wise pairs (dropping duplicates - src/dst pairs that come from multi-graph)
               unique_incoming_edges_to_thread_node, _ = torch.unique( data.edge_index[:, incoming_edges_to_thread_node_idx ], dim=1, return_inverse=True)

               target_nodes_of_outgoing_edges_from_thread_node___unique = unique_outgoing_edges_from_thread_node[1] # edge-target is index 1
               source_nodes_of_incoming_edges_to_thread_node___unique = unique_incoming_edges_to_thread_node[0] # edge-src is index 0


               #-- Option-1 --------------------------------------------------------------------------------------------------------------------------------------------------------------
               # # Already handled multi-graph case, but how about the bi-directional edge case?
               # # For, T-->F and T<--F, information of F will be recorded twice."Dont Let it happen"
               #
               # --- JY @ 2024-1-27: Need to to torch.unique once more, since target-nodes and source-nodes could be duplicate (if T has bi-directional relationship with F/R/N/P node)
               unique_adjacent_nodes_of_both_direction_edges_of_thread_node = torch.unique( torch.cat( [ target_nodes_of_outgoing_edges_from_thread_node___unique, 
                                                                                                         source_nodes_of_incoming_edges_to_thread_node___unique ] ) )
               integrated_5bit_of_all_unique_adjacent_nodes_to_thread = torch.sum( data_x_first5[unique_adjacent_nodes_of_both_direction_edges_of_thread_node], dim = 0 )
               
               data_thread_node_all_unique_adjacent_nodes_5bit_dists = torch.cat(( data_thread_node_all_unique_adjacent_nodes_5bit_dists,
                                                                                   integrated_5bit_of_all_unique_adjacent_nodes_to_thread.unsqueeze(0) ), dim = 0)
               # --------------------------------------------------------------------------------------------------------------------------------------------------------------------------
               # ==============================================================================================================================================
               # Thread's File/Registry/Network/Process Event-Counts

               ALL_outgoing_edges_from_thread_node = data.edge_index[:, outgoing_edges_from_thread_node_idx ]
               ALL_incoming_edges_to_thread_node = data.edge_index[:, incoming_edges_to_thread_node_idx ]


               target_nodes_of_ALL_outgoing_edges_from_thread_node___duplicate_possible = ALL_outgoing_edges_from_thread_node[1]
               source_nodes_of_ALL_incoming_edges_to_thread_node___duplicate_possible = ALL_incoming_edges_to_thread_node[0]

               types_of_target_nodes_of_ALL_outgoing_edges_from_thread_node = data_x_first5[ target_nodes_of_ALL_outgoing_edges_from_thread_node___duplicate_possible ]
               types_of_source_nodes_of_ALL_incoming_edges_to_thread_node = data_x_first5[ source_nodes_of_ALL_incoming_edges_to_thread_node___duplicate_possible ]


               # FRNP 'outgoing and incoming' event count
               # -- "dist_of__types_of_target_nodes_of_ALL_outgoing_edges_from_thread_node(duplicate possible)" is basically "F/R/N/P outgoing event counts from thread node" 
               # -- "dist_of__types_of_source_nodes_of_ALL_incoming_edges_to_thread_node(duplicate possible)" is basically "F/R/N/P incoming event counts to thread node"
               #
               #  * do need to drop the thread-node column though since we want F/R/N/P instead of F/R/N/P/T
               dist_of__types_of_target_nodes_of_ALL_outgoing_edges_from_thread_node = torch.sum( types_of_target_nodes_of_ALL_outgoing_edges_from_thread_node, dim = 0 )[:4]
               dist_of__types_of_source_nodes_of_ALL_incoming_edges_to_thread_node = torch.sum( types_of_source_nodes_of_ALL_incoming_edges_to_thread_node, dim = 0 )[:4]


               # 'FRNP event count' can be obtained by vector-summing 'FRNP outgoing event count' and 'FRNP incoming event count'
               # --- no need to involve any torch.unique operations
               #     because we are not trying to find unique nodes, 
               #     but just trying to get the total count of edges(==events) involved between thread-node and F/R/N/P nodes
               dist_of__types_of_nodes__associated_with_ALL_edges__connected_to__thread_node = \
                                                                  torch.sum( 
                                                                             torch.cat((dist_of__types_of_target_nodes_of_ALL_outgoing_edges_from_thread_node.unsqueeze(0),
                                                                                        dist_of__types_of_source_nodes_of_ALL_incoming_edges_to_thread_node.unsqueeze(0)), 
                                                                                        dim = 0 ) ,
                                                                           dim = 0 )

               data_thread_node___dist_of__types_of_nodes__associated_with_ALL_edges__connected_to__thread_node = \
                                                      torch.cat(( data_thread_node___dist_of__types_of_nodes__associated_with_ALL_edges__connected_to__thread_node,
                                                                  dist_of__types_of_nodes__associated_with_ALL_edges__connected_to__thread_node.unsqueeze(0) ), dim = 0)                                                                                  
                                                                  # unsqueeze is necessary -- Returns a new tensor with a dimension of size one inserted at the specified position.
                                                                  #                           https://pytorch.org/docs/stable/generated/torch.unsqueeze.html
               STOP_HERE = 1

            # **************************************************************************************************************************************************
            # [ node-type5bit + N>gram events ]
            thread_level_nodetype5bit__FRNPeventCount__N_gram_events__dist = torch.cat( [data_thread_node_all_unique_adjacent_nodes_5bit_dists,
                                                                                         data_thread_node___dist_of__types_of_nodes__associated_with_ALL_edges__connected_to__thread_node, 
                                                                                         data_thread_level_all_Ngram_features, 
                                                                                        ], dim = 1)
            
            
            
            ''' JY @ 2024-1-20: Now "pool" all thead-node's embedding, to generate the "graph embedding". '''
            if pool == "sum": pool__func = torch.sum
            elif pool == "mean": pool__func = torch.mean

            graph_embedding = pool__func(thread_level_nodetype5bit__FRNPeventCount__N_gram_events__dist, dim = 0)            
            
            
            
            data_dict[ re.search(r'Processed_SUBGRAPH_P3_(.*?)\.pickle', data.name).group(1) ] = graph_embedding.tolist()

            # data_dict[ data.name.lstrip("Processed_SUBGRAPH_P3_").rstrip(".pickle") ] = data_thread_node_both_direction_edges_edge_attrs.tolist()
            cnt+=1
      return data_dict

   # -----------------------------------------------------------------------------------------------------------------------------------------------------
   
# JY @ 2024-1-20: thread-level N>1 gram events + nodetype-5bits
def get_thread_level_N_gram_events__nodetype5bit__FRNPeventCount__FRNP_OutgoIncom_eventCount__dict( 
                                                            pretrained_Ngram_countvectorizer_list : list, # TODO        
                                                            dataset : list, 
                                                            pool : str = "sum",

                                                              ):
      
      ''' JY @ 2024-1-20 : Implement this '''

      thread_node_tensor = torch.tensor([0, 0, 0, 0, 1])

      data_dict = dict()

      cnt = 1
      for data in dataset:
            
            print(f"{cnt} / {len(dataset)}: {data.name} -- generate graph-embedding", flush = True)

            # Added by JY @ 2023-07-18 to handle node-attr dim > 5  
            # if data.x.shape[1] != 5:
            data_x_first5 = data.x[:,:5]
            thread_node_indices = torch.nonzero(torch.all(torch.eq( data_x_first5, thread_node_tensor), dim=1), as_tuple=False).flatten().tolist()

            # which this node is a source-node (outgoing-edge w.r.t this node)
  


            data_thread_node_all_unique_adjacent_nodes_5bit_dists = torch.tensor([]) # Added by JY @ 2023-07-19

            data_thread_level_all_Ngram_features = torch.tensor([]) # Added by JY @ 2023-01-20


            data_thread_node___dist_of__types_of_nodes__associated_with_ALL_edges__connected_to__thread_node = torch.tensor([]) # Added by JY @ 2024-01-27

            data_thread_node___dist_of__types_of_target_nodes_of_ALL_outgoing_edges_from_thread_node = torch.tensor([]) # Added by JY @ 2024-01-27
            data_thread_node___dist_of__types_of_source_nodes_of_ALL_incoming_edges_to_thread_node = torch.tensor([]) # Added by JY @ 2024-01-27




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


               # ==============================================================================================================================================
               # Thread-level N-gram events features
               ''' JY @ 2024-1-20: Get thread-level event-sequence sorted by timestamps '''
               # Refered to : https://github.com/SUNY-IBM-dev/graph_embedding_improvement/blob/20627016d59466d3dad191ff208efce97b15d35e/graph_embedding_improvement_efforts/Trial_7__Thread_level_N_grams__N_gt_than_1__Similar_to_PriorGraphEmbedding/thread_level_n_gram__n_gt_1__similar_to_asiaccs_graph_embedding.py#L483C1-L491C95

               edge_attr_of_both_direction_edges_from_thread_node_idx = torch.cat([edge_attr_of_incoming_edges_from_thread_node_idx, 
                                                                                   edge_attr_of_outgoing_edges_from_thread_node_idx],
                                                                                   dim = 0)


               timestamp_sorted_indices = torch.argsort(edge_attr_of_both_direction_edges_from_thread_node_idx[:, -1], descending=False)

               edge_attr_of_both_direction_edges_from_thread_node_idx__sorted = edge_attr_of_both_direction_edges_from_thread_node_idx[ timestamp_sorted_indices ]

               taskname_indices = torch.nonzero(edge_attr_of_both_direction_edges_from_thread_node_idx__sorted[:,:-1], as_tuple=False)[:, -1]


               thread_sorted_event_list = [taskname_colnames[i] for i in taskname_indices]

               thread_sorted_event_str_sequence = " ".join(thread_sorted_event_list) # for compabitiblity with countvecotrizer

               # transform and get count -- https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html#sklearn.feature_extraction.text.CountVectorizer.transform
               # Refer to: https://github.com/SUNY-IBM-dev/graph_embedding_improvement/blob/20627016d59466d3dad191ff208efce97b15d35e/graph_embedding_improvement_efforts/Trial_4__Local_N_gram__standard_message_passing/local_ngram__standard_message_passing__graph_embedding.py#L760C41-L760C72

               # Get all Ngrams (N could be multiple, depending on 'only_train_specified_Ngram' parameter value)
               thread_all_Ngram_counts__appended_nparray = np.array([]) # torch.Tensor()
               for pretrained_Ngram_countvectorizer in pretrained_Ngram_countvectorizer_list:
                  # If multiple countvectorizers, supposed to start from 1gram to Ngram
                  #print(f"pretrained_Ngram_countvectorizer.ngram_range: {pretrained_Ngram_countvectorizer.ngram_range}", flush= True)
                  thread_Ngram_counts__portion = pretrained_Ngram_countvectorizer.transform( [ thread_sorted_event_str_sequence ] ).toarray() # needs to be in a list
                  thread_all_Ngram_counts__appended_nparray = np.append(thread_all_Ngram_counts__appended_nparray, thread_Ngram_counts__portion )           
               #print("\n")

               # JY @ 2024-1-20: so that can stack on 'data_thread_level_all_Ngram_features' for all thread's Ngram features within this subgrpah(data)
               thread_all_Ngram_counts__appended_tensor = torch.Tensor(thread_all_Ngram_counts__appended_nparray).view(1,-1) # for Size([1,edge_feat_len])
               data_thread_level_all_Ngram_features = torch.cat((data_thread_level_all_Ngram_features, thread_all_Ngram_counts__appended_tensor), dim=0)                  



               # ==============================================================================================================================================
               # Node-level

               # But also need to consider the multi-graph aspect here. 
               # So here, do not count twice for duplicate adjacent nodes due to multigraph.
               # Just need to get the distribution.
               # Find unique column-wise pairs (dropping duplicates - src/dst pairs that come from multi-graph)
               unique_outgoing_edges_from_thread_node, _ = torch.unique( data.edge_index[:, outgoing_edges_from_thread_node_idx ], dim=1, return_inverse=True)

               # Find unique column-wise pairs (dropping duplicates - src/dst pairs that come from multi-graph)
               unique_incoming_edges_to_thread_node, _ = torch.unique( data.edge_index[:, incoming_edges_to_thread_node_idx ], dim=1, return_inverse=True)

               target_nodes_of_outgoing_edges_from_thread_node___unique = unique_outgoing_edges_from_thread_node[1] # edge-target is index 1
               source_nodes_of_incoming_edges_to_thread_node___unique = unique_incoming_edges_to_thread_node[0] # edge-src is index 0


               #-- Option-1 --------------------------------------------------------------------------------------------------------------------------------------------------------------
               # # Already handled multi-graph case, but how about the bi-directional edge case?
               # # For, T-->F and T<--F, information of F will be recorded twice."Dont Let it happen"
               #
               # --- JY @ 2024-1-27: Need to to torch.unique once more, since target-nodes and source-nodes could be duplicate (if T has bi-directional relationship with F/R/N/P node)
               unique_adjacent_nodes_of_both_direction_edges_of_thread_node = torch.unique( torch.cat( [ target_nodes_of_outgoing_edges_from_thread_node___unique, 
                                                                                                         source_nodes_of_incoming_edges_to_thread_node___unique ] ) )
               integrated_5bit_of_all_unique_adjacent_nodes_to_thread = torch.sum( data_x_first5[unique_adjacent_nodes_of_both_direction_edges_of_thread_node], dim = 0 )
               
               data_thread_node_all_unique_adjacent_nodes_5bit_dists = torch.cat(( data_thread_node_all_unique_adjacent_nodes_5bit_dists,
                                                                                   integrated_5bit_of_all_unique_adjacent_nodes_to_thread.unsqueeze(0) ), dim = 0)
               # --------------------------------------------------------------------------------------------------------------------------------------------------------------------------
               # ==============================================================================================================================================
               # Thread's File/Registry/Network/Process Event-Counts

               ALL_outgoing_edges_from_thread_node = data.edge_index[:, outgoing_edges_from_thread_node_idx ]
               ALL_incoming_edges_to_thread_node = data.edge_index[:, incoming_edges_to_thread_node_idx ]


               target_nodes_of_ALL_outgoing_edges_from_thread_node___duplicate_possible = ALL_outgoing_edges_from_thread_node[1]
               source_nodes_of_ALL_incoming_edges_to_thread_node___duplicate_possible = ALL_incoming_edges_to_thread_node[0]

               types_of_target_nodes_of_ALL_outgoing_edges_from_thread_node = data_x_first5[ target_nodes_of_ALL_outgoing_edges_from_thread_node___duplicate_possible ]
               types_of_source_nodes_of_ALL_incoming_edges_to_thread_node = data_x_first5[ source_nodes_of_ALL_incoming_edges_to_thread_node___duplicate_possible ]


               # FRNP 'outgoing and incoming' event count
               # -- "dist_of__types_of_target_nodes_of_ALL_outgoing_edges_from_thread_node(duplicate possible)" is basically "F/R/N/P outgoing event counts from thread node" 
               # -- "dist_of__types_of_source_nodes_of_ALL_incoming_edges_to_thread_node(duplicate possible)" is basically "F/R/N/P incoming event counts to thread node"
               #
               #  * do need to drop the thread-node column though since we want F/R/N/P instead of F/R/N/P/T
               dist_of__types_of_target_nodes_of_ALL_outgoing_edges_from_thread_node = torch.sum( types_of_target_nodes_of_ALL_outgoing_edges_from_thread_node, dim = 0 )[:4]
               dist_of__types_of_source_nodes_of_ALL_incoming_edges_to_thread_node = torch.sum( types_of_source_nodes_of_ALL_incoming_edges_to_thread_node, dim = 0 )[:4]


               # 'FRNP event count' can be obtained by vector-summing 'FRNP outgoing event count' and 'FRNP incoming event count'
               # --- no need to involve any torch.unique operations
               #     because we are not trying to find unique nodes, 
               #     but just trying to get the total count of edges(==events) involved between thread-node and F/R/N/P nodes
               dist_of__types_of_nodes__associated_with_ALL_edges__connected_to__thread_node = \
                                                                  torch.sum( 
                                                                             torch.cat((dist_of__types_of_target_nodes_of_ALL_outgoing_edges_from_thread_node.unsqueeze(0),
                                                                                        dist_of__types_of_source_nodes_of_ALL_incoming_edges_to_thread_node.unsqueeze(0)), 
                                                                                        dim = 0 ) ,
                                                                           dim = 0 )

               data_thread_node___dist_of__types_of_nodes__associated_with_ALL_edges__connected_to__thread_node = \
                                                      torch.cat(( data_thread_node___dist_of__types_of_nodes__associated_with_ALL_edges__connected_to__thread_node,
                                                                  dist_of__types_of_nodes__associated_with_ALL_edges__connected_to__thread_node.unsqueeze(0) ), dim = 0)                                                                                  
                                                                  # unsqueeze is necessary -- Returns a new tensor with a dimension of size one inserted at the specified position.
                                                                  #                           https://pytorch.org/docs/stable/generated/torch.unsqueeze.html


               data_thread_node___dist_of__types_of_target_nodes_of_ALL_outgoing_edges_from_thread_node = \
                                                      torch.cat(( data_thread_node___dist_of__types_of_target_nodes_of_ALL_outgoing_edges_from_thread_node,
                                                                  dist_of__types_of_target_nodes_of_ALL_outgoing_edges_from_thread_node.unsqueeze(0) ), dim = 0)

               data_thread_node___dist_of__types_of_source_nodes_of_ALL_incoming_edges_to_thread_node = \
                                                      torch.cat(( data_thread_node___dist_of__types_of_source_nodes_of_ALL_incoming_edges_to_thread_node,
                                                                  dist_of__types_of_source_nodes_of_ALL_incoming_edges_to_thread_node.unsqueeze(0)), dim = 0)


               STOP_HERE = 1



            # **************************************************************************************************************************************************
            # [ node-type5bit + N>gram events ]
            thread_level_nodetype5bit__FRNPeventCount__FRNP_OutgoIncom_eventCount__N_gram_events__dist = torch.cat( 
                                                                                         [data_thread_node_all_unique_adjacent_nodes_5bit_dists,

                                                                                          data_thread_node___dist_of__types_of_nodes__associated_with_ALL_edges__connected_to__thread_node,

                                                                                          data_thread_node___dist_of__types_of_target_nodes_of_ALL_outgoing_edges_from_thread_node,
                                                                                          data_thread_node___dist_of__types_of_source_nodes_of_ALL_incoming_edges_to_thread_node,

                                                                                          data_thread_level_all_Ngram_features 
                                                                                         ], dim = 1)
            
            
            
            ''' JY @ 2024-1-20: Now "pool" all thead-node's embedding, to generate the "graph embedding". '''
            if pool == "sum": pool__func = torch.sum
            elif pool == "mean": pool__func = torch.mean

            graph_embedding = pool__func(thread_level_nodetype5bit__FRNPeventCount__FRNP_OutgoIncom_eventCount__N_gram_events__dist, dim = 0)            
            
            
            
            data_dict[ re.search(r'Processed_SUBGRAPH_P3_(.*?)\.pickle', data.name).group(1) ] = graph_embedding.tolist()

            # data_dict[ data.name.lstrip("Processed_SUBGRAPH_P3_").rstrip(".pickle") ] = data_thread_node_both_direction_edges_edge_attrs.tolist()
            cnt+=1
      return data_dict


   # -----------------------------------------------------------------------------------------------------------------------------------------------------

# JY @ 2024-1-29: thread-level N>1 gram events + nodetype-5bits + Average Number of Different Threads per F/R/N/P node 
def get_thread_level_N_gram_events__nodetype5bit__AvgNum_DiffThreads_perFRNP__dict( 
                                                            pretrained_Ngram_countvectorizer_list : list, # TODO        
                                                            dataset : list, 
                                                            pool : str = "sum",

                                                              ):
      

      thread_node_tensor = torch.tensor([0, 0, 0, 0, 1])

      file_node_tensor = torch.tensor([1, 0, 0, 0, 0])
      reg_node_tensor = torch.tensor([0, 1, 0, 0, 0])
      net_node_tensor = torch.tensor([0, 0, 1, 0, 0])
      proc_node_tensor = torch.tensor([0, 0, 0, 1, 0])


      data_dict = dict()

      cnt = 1
      for data in dataset:
            
            print(f"{cnt} / {len(dataset)}: {data.name} -- generate graph-embedding", flush = True)

            # Added by JY @ 2023-07-18 to handle node-attr dim > 5  
            # if data.x.shape[1] != 5:
  

            # ------------------------------------------------------------------------------------------------------------------------------------------------
            data_x_first5 = data.x[:,:5]
            thread_node_indices = torch.nonzero(torch.all(torch.eq( data_x_first5, thread_node_tensor), dim=1), as_tuple=False).flatten().tolist()

            # which this node is a source-node (outgoing-edge w.r.t this node)
  
            data_thread_node_all_unique_adjacent_nodes_5bit_dists = torch.tensor([]) # Added by JY @ 2023-07-19
            data_thread_level_all_Ngram_features = torch.tensor([]) # Added by JY @ 2023-01-20
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

               thread_sorted_event_str_sequence = " ".join(thread_sorted_event_list) # for compabitiblity with countvecotrizer

               # transform and get count -- https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html#sklearn.feature_extraction.text.CountVectorizer.transform
               # Refer to: https://github.com/SUNY-IBM-dev/graph_embedding_improvement/blob/20627016d59466d3dad191ff208efce97b15d35e/graph_embedding_improvement_efforts/Trial_4__Local_N_gram__standard_message_passing/local_ngram__standard_message_passing__graph_embedding.py#L760C41-L760C72

               # Get all Ngrams (N could be multiple, depending on 'only_train_specified_Ngram' parameter value)
               thread_all_Ngram_counts__appended_nparray = np.array([]) # torch.Tensor()
               for pretrained_Ngram_countvectorizer in pretrained_Ngram_countvectorizer_list:
                  # If multiple countvectorizers, supposed to start from 1gram to Ngram
                  #print(f"pretrained_Ngram_countvectorizer.ngram_range: {pretrained_Ngram_countvectorizer.ngram_range}", flush= True)
                  thread_Ngram_counts__portion = pretrained_Ngram_countvectorizer.transform( [ thread_sorted_event_str_sequence ] ).toarray() # needs to be in a list
                  thread_all_Ngram_counts__appended_nparray = np.append(thread_all_Ngram_counts__appended_nparray, thread_Ngram_counts__portion )           
               #print("\n")

               # JY @ 2024-1-20: so that can stack on 'data_thread_level_all_Ngram_features' for all thread's Ngram features within this subgrpah(data)
               thread_all_Ngram_counts__appended_tensor = torch.Tensor(thread_all_Ngram_counts__appended_nparray).view(1,-1) # for Size([1,edge_feat_len])
               data_thread_level_all_Ngram_features = torch.cat((data_thread_level_all_Ngram_features, thread_all_Ngram_counts__appended_tensor), dim=0)                  



               # ==============================================================================================================================================
               # Node-level

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
               integrated_5bit_of_all_unique_adjacent_nodes_to_thread = torch.sum( data_x_first5[unique_adjacent_nodes_of_both_direction_edges_of_thread_node], dim = 0 )
               
               data_thread_node_all_unique_adjacent_nodes_5bit_dists = torch.cat(( data_thread_node_all_unique_adjacent_nodes_5bit_dists,
                                                                                   integrated_5bit_of_all_unique_adjacent_nodes_to_thread.unsqueeze(0) ), dim = 0)
               # --------------------------------------------------------------------------------------------------------------------------------------------------------------------------
            # ======================================================================================================================================================

            # JY @ 2024-1-29 : Now deal with F/R/N/P

            file_node_indices = torch.nonzero(torch.all(torch.eq( data_x_first5, file_node_tensor), dim=1), as_tuple=False).flatten().tolist()
            reg_node_indices = torch.nonzero(torch.all(torch.eq( data_x_first5, reg_node_tensor), dim=1), as_tuple=False).flatten().tolist()
            net_node_indices = torch.nonzero(torch.all(torch.eq( data_x_first5, net_node_tensor), dim=1), as_tuple=False).flatten().tolist()
            proc_node_indices = torch.nonzero(torch.all(torch.eq( data_x_first5, proc_node_tensor), dim=1), as_tuple=False).flatten().tolist()


            edge_src_indices = data.edge_index[0]
            edge_tar_indices = data.edge_index[1]
            data__num_diff_thread_nodes__file_node_interacted_with = []
            for file_node_idx in file_node_indices:
               outgoing_edges_from_file_node_idx = torch.nonzero( edge_src_indices == file_node_idx ).flatten()
               incoming_edges_to_file_node_idx = torch.nonzero( edge_tar_indices == file_node_idx ).flatten()

               unique_outgoing_edges_from_file_node, _ = torch.unique( data.edge_index[:, outgoing_edges_from_file_node_idx ], dim=1, return_inverse=True)
               unique_incoming_edges_to_file_node, _ = torch.unique( data.edge_index[:, incoming_edges_to_file_node_idx ], dim=1, return_inverse=True)

               target_nodes_of_outgoing_edges_from_file_node = unique_outgoing_edges_from_file_node[1] # edge-target is index 1
               source_nodes_of_incoming_edges_to_file_node = unique_incoming_edges_to_file_node[0] # edge-src is index 0

               unique_adjacent_nodes_of_both_direction_edges_of_file_node = torch.unique( torch.cat( [ target_nodes_of_outgoing_edges_from_file_node, 
                                                                                                       source_nodes_of_incoming_edges_to_file_node ] ) )
               integrated_5bit_of_all_unique_adjacent_nodes_to_file = torch.sum( data_x_first5[unique_adjacent_nodes_of_both_direction_edges_of_file_node], dim = 0 )
               num_diff_thread_nodes__file_node_interacted_with = int(integrated_5bit_of_all_unique_adjacent_nodes_to_file[-1])
               data__num_diff_thread_nodes__file_node_interacted_with.append(num_diff_thread_nodes__file_node_interacted_with)

            data__num_diff_thread_nodes__reg_node_interacted_with = []
            for reg_node_idx in reg_node_indices:
               outgoing_edges_from_reg_node_idx = torch.nonzero( edge_src_indices == reg_node_idx ).flatten()
               incoming_edges_to_reg_node_idx = torch.nonzero( edge_tar_indices == reg_node_idx ).flatten()

               unique_outgoing_edges_from_reg_node, _ = torch.unique( data.edge_index[:, outgoing_edges_from_reg_node_idx ], dim=1, return_inverse=True)
               unique_incoming_edges_to_reg_node, _ = torch.unique( data.edge_index[:, incoming_edges_to_reg_node_idx ], dim=1, return_inverse=True)

               target_nodes_of_outgoing_edges_from_reg_node = unique_outgoing_edges_from_reg_node[1] # edge-target is index 1
               source_nodes_of_incoming_edges_to_reg_node = unique_incoming_edges_to_reg_node[0] # edge-src is index 0

               unique_adjacent_nodes_of_both_direction_edges_of_reg_node = torch.unique( torch.cat( [ target_nodes_of_outgoing_edges_from_reg_node, 
                                                                                                       source_nodes_of_incoming_edges_to_reg_node ] ) )
               integrated_5bit_of_all_unique_adjacent_nodes_to_reg = torch.sum( data_x_first5[unique_adjacent_nodes_of_both_direction_edges_of_reg_node], dim = 0 )
               num_diff_thread_nodes__reg_node_interacted_with = int(integrated_5bit_of_all_unique_adjacent_nodes_to_reg[-1])
               data__num_diff_thread_nodes__reg_node_interacted_with.append(num_diff_thread_nodes__reg_node_interacted_with)


            data__num_diff_thread_nodes__net_node_interacted_with = []
            for net_node_idx in net_node_indices:
               outgoing_edges_from_net_node_idx = torch.nonzero( edge_src_indices == net_node_idx ).flatten()
               incoming_edges_to_net_node_idx = torch.nonzero( edge_tar_indices == net_node_idx ).flatten()

               unique_outgoing_edges_from_net_node, _ = torch.unique( data.edge_index[:, outgoing_edges_from_net_node_idx ], dim=1, return_inverse=True)
               unique_incoming_edges_to_net_node, _ = torch.unique( data.edge_index[:, incoming_edges_to_net_node_idx ], dim=1, return_inverse=True)

               target_nodes_of_outgoing_edges_from_net_node = unique_outgoing_edges_from_net_node[1] # edge-target is index 1
               source_nodes_of_incoming_edges_to_net_node = unique_incoming_edges_to_net_node[0] # edge-src is index 0

               unique_adjacent_nodes_of_both_direction_edges_of_net_node = torch.unique( torch.cat( [ target_nodes_of_outgoing_edges_from_net_node, 
                                                                                                       source_nodes_of_incoming_edges_to_net_node ] ) )
               integrated_5bit_of_all_unique_adjacent_nodes_to_net = torch.sum( data_x_first5[unique_adjacent_nodes_of_both_direction_edges_of_net_node], dim = 0 )
               num_diff_thread_nodes__net_node_interacted_with = int(integrated_5bit_of_all_unique_adjacent_nodes_to_net[-1])
               data__num_diff_thread_nodes__net_node_interacted_with.append(num_diff_thread_nodes__net_node_interacted_with)

            data__num_diff_thread_nodes__proc_node_interacted_with = []
            for proc_node_idx in proc_node_indices:
               outgoing_edges_from_proc_node_idx = torch.nonzero( edge_src_indices == proc_node_idx ).flatten()
               incoming_edges_to_proc_node_idx = torch.nonzero( edge_tar_indices == proc_node_idx ).flatten()

               unique_outgoing_edges_from_proc_node, _ = torch.unique( data.edge_index[:, outgoing_edges_from_proc_node_idx ], dim=1, return_inverse=True)
               unique_incoming_edges_to_proc_node, _ = torch.unique( data.edge_index[:, incoming_edges_to_proc_node_idx ], dim=1, return_inverse=True)

               target_nodes_of_outgoing_edges_from_proc_node = unique_outgoing_edges_from_proc_node[1] # edge-target is index 1
               source_nodes_of_incoming_edges_to_proc_node = unique_incoming_edges_to_proc_node[0] # edge-src is index 0

               unique_adjacent_nodes_of_both_direction_edges_of_proc_node = torch.unique( torch.cat( [ target_nodes_of_outgoing_edges_from_proc_node, 
                                                                                                       source_nodes_of_incoming_edges_to_proc_node ] ) )
               integrated_5bit_of_all_unique_adjacent_nodes_to_proc = torch.sum( data_x_first5[unique_adjacent_nodes_of_both_direction_edges_of_proc_node], dim = 0 )
               num_diff_thread_nodes__proc_node_interacted_with = int(integrated_5bit_of_all_unique_adjacent_nodes_to_proc[-1])
               data__num_diff_thread_nodes__proc_node_interacted_with.append(num_diff_thread_nodes__proc_node_interacted_with)



            if data__num_diff_thread_nodes__file_node_interacted_with == []:
               AvgNum_DiffThreads_for_File_Nodes = 0
            else:
               AvgNum_DiffThreads_for_File_Nodes = float(np.mean(data__num_diff_thread_nodes__file_node_interacted_with))
            
            if data__num_diff_thread_nodes__reg_node_interacted_with == []:
               AvgNum_DiffThreads_for_Registry_Nodes = 0
            else:
               AvgNum_DiffThreads_for_Registry_Nodes = float(np.mean(data__num_diff_thread_nodes__reg_node_interacted_with))
            
            if data__num_diff_thread_nodes__net_node_interacted_with == []:
               AvgNum_DiffThreads_for_Network_Nodes = 0
            else:
               AvgNum_DiffThreads_for_Network_Nodes = float(np.mean(data__num_diff_thread_nodes__net_node_interacted_with))
            
            if data__num_diff_thread_nodes__proc_node_interacted_with == []:
               AvgNum_DiffThreads_for_Process_Nodes = 0
            else:
               AvgNum_DiffThreads_for_Process_Nodes = float(np.mean(data__num_diff_thread_nodes__proc_node_interacted_with))


            data_AvgNum_DiffThreads_perFRNP = [AvgNum_DiffThreads_for_File_Nodes, AvgNum_DiffThreads_for_Registry_Nodes, 
                                               AvgNum_DiffThreads_for_Network_Nodes, AvgNum_DiffThreads_for_Process_Nodes]
            # 다 sumpool하고 
            # feature value 똑갘나 비교해보는방법도있음 
            # ---> torch.sum( data_thread_node_all_unique_adjacent_nodes_5bit_dists, dim = 0)[0] 와 sum(data__num_diff_thread_nodes__file_node_interacted_with) 비교햇을때 
            #                  같은경우있는거 보니 맞는듯.


            # ======================================================================================================================================================
            # [ node-type5bit + AvgNum_DiffThreads_perFRNP + N>1gram events ]
            thread_level_N_gram_events_adjacent_5bit_dist = torch.cat( [data_thread_node_all_unique_adjacent_nodes_5bit_dists, 
                                                                        data_thread_level_all_Ngram_features, 
                                                                         ], dim = 1)
            
            
            
            ''' JY @ 2024-1-29: Following pooling applys to 'node-type5bit' and 'N>1gram events' '''
            if pool == "sum": pool__func = torch.sum
            elif pool == "mean": pool__func = torch.mean

            data__pooled__nodetype5bit__NgramEvents = pool__func(thread_level_N_gram_events_adjacent_5bit_dist, dim = 0)            
            
            graph_embedding = data__pooled__nodetype5bit__NgramEvents[:5].tolist() + data_AvgNum_DiffThreads_perFRNP + data__pooled__nodetype5bit__NgramEvents[5:].tolist()
            
            data_dict[ re.search(r'Processed_SUBGRAPH_P3_(.*?)\.pickle', data.name).group(1) ] = graph_embedding

            # data_dict[ data.name.lstrip("Processed_SUBGRAPH_P3_").rstrip(".pickle") ] = data_thread_node_both_direction_edges_edge_attrs.tolist()
            cnt+=1
      return data_dict



get_1_grams_events__nodetype5bit__FRNPeventCount__FRNP_OutgoIncom_eventCount__AvgNum_DiffThreads_perFRNP__dict

# JY @ 2024-2-10: thread-level N>1 gram events + nodetype-5bits + FRNP event-counts (+incoming & outgoing) + Average Number of Different Threads per F/R/N/P node 
def get_1_grams_events__nodetype5bit__FRNPeventCount__FRNP_OutgoIncom_eventCount__AvgNum_DiffThreads_perFRNP__dict( 
                                                            pretrained_Ngram_countvectorizer_list : list, # TODO        
                                                            dataset : list, 
                                                            pool : str = "sum",
                                                      ):

##########################################################################################################################################################
##########################################################################################################################################################


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

# https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMClassifier.html#lightgbm.LGBMClassifier
#from lightgbm import LGBMClassifier # pip install lightgbm -- added by JY @ 2023-08-10

# for baseline
from sklearn.feature_extraction.text import CountVectorizer


##############################################################################################################################


# JY @ 2023-07-25: Here, not only tune using K-fold-CV, but also fit the best hyperparam on entire-train and test on final-test

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('-k', '--K', nargs = 1, type = int, default = [10])  


    parser.add_argument('-data', '--dataset', 
                        choices= ['Dataset-Case-1',
                                  'Dataset-Case-2',
                                  'Dataset-Case-3',
                                  'Dataset-Case-3__FR_UID_rule_updated',
                                 
                                   # 
                                  'Dataset_1__NoTrace_UIDruleUpdated', # Partial Dataset-1
                                  'Dataset_2__NoTrace_UIDruleUpdated', # Partial Dataset-2

                                  'Full_Dataset_1_NoTraceUIDupdated',
                                  'Full_Dataset_2_NoTraceUIDupdated',


                                  'Full_Dataset_1_Double_Stratified',
                                  'Full_Dataset_2_Double_Stratified'
                                  ], 
                        default = ['Full_Dataset_2_Double_Stratified'])


    model_cls_map = {"RandomForest": RandomForestClassifier, "XGBoost": GradientBoostingClassifier,
                     "LogisticRegression": LogisticRegression, "SVM": svm } 
    
    parser.add_argument('-mod_cls', '--model_choice', nargs = 1, type = str, 
                        default = ["RandomForest"] )

    parser.add_argument('-ss_opt', '--search_space_option', 
                        choices= [
                                  'XGBoost_searchspace_1',
                                  'RandomForest_searchspace_1',

                                 #  # ------------------------------------------------------------------------------------
                                 #  # thread_level__N>1_grams_events__nodetype5bit
                                 #  'Best_RF__Dataset_1__2gram__sum_pool__only_train_specified_Ngram_True', # tuning-complete
                                 #  'Best_RF__Dataset_1__4gram__sum_pool__only_train_specified_Ngram_True', # tuning-complete
                                 #  'Best_RF__Dataset_1__6gram__sum_pool__only_train_specified_Ngram_True',

                                 #  'Best_RF__Dataset_2__2gram__sum_pool__only_train_specified_Ngram_True', 
                                 #  'Best_RF__Dataset_2__4gram__sum_pool__only_train_specified_Ngram_True', 
                                 #  'Best_RF__Dataset_2__6gram__sum_pool__only_train_specified_Ngram_True',
                                 #  # ------------------------------------------------------------------------------------
                                 #  # thread_level__N>1_grams_events__nodetype5bit__FRNPeventCount__FRNP_OutgoIncom_eventCount
                                 #  'Best_RF__Dataset_1__2gram__FRNPeventCount__FRNP_OutgoIncom_eventCount',
                                 #  'Best_RF__Dataset_1__4gram__FRNPeventCount__FRNP_OutgoIncom_eventCount',

                                 #  # -------------------------------------------------------------------------------------
                                 #  # thread_level__N>1_grams_events__nodetype5bit__AvgNum_DiffThreads_perFRNP
                                 #  'Best_RF__Dataset_3_FR_UID_rule_updated__2gram_with__AvgNum_DiffThreads_perFRNP',

                                  # -------------------------------------------------------------------------------------
                                  # thread_level__N>1_grams_events__nodetype5bit
                                  'Best_RF__Partial_Dataset_1_NoTraceUIDUpdated__1gram__sum_pool', # tuning done # final-tested
                                  'Best_RF__Partial_Dataset_1_NoTraceUIDUpdated__2gram__sum_pool', # tuning done # final-tested
                                  'Best_RF__Partial_Dataset_1_NoTraceUIDUpdated__4gram__sum_pool', # tuning done     

                                  'Best_RF__Partial_Dataset_2_NoTraceUIDUpdated__1gram__sum_pool', # tuning done # final-tested
                                  'Best_RF__Partial_Dataset_2_NoTraceUIDUpdated__2gram__sum_pool', # tuning done # final-tested
                                  'Best_RF__Partial_Dataset_2_NoTraceUIDUpdated__4gram__sum_pool',                                  

                                  # -------------------------------------------------------------------------------------
                                  # thread_level__N>1_grams_events__nodetype5bit
                                  'Best_RF__Full_Dataset_1_NoTraceUIDupdated__1gram__sum_pool', 
                                  'Best_RF__Full_Dataset_1_NoTraceUIDupdated__2gram__sum_pool', 
                                  'Best_RF__Full_Dataset_1_NoTraceUIDupdated__4gram__sum_pool',      

                                  'Best_RF__Full_Dataset_2_NoTraceUIDupdated__1gram__sum_pool', 
                                  'Best_RF__Full_Dataset_2_NoTraceUIDupdated__2gram__sum_pool', 
                                  'Best_RF__Full_Dataset_2_NoTraceUIDupdated__4gram__sum_pool',    
                                  # -------------------------------------------------------------------------------------

                                  "Best_RF__Full_Dataset_1_Double_Stratified__1gram__sum_pool", # tuning-complete # final-tested
                                  "Best_RF__Full_Dataset_1_Double_Stratified__2gram__sum_pool", # tuning-complete # final-tested


                                  "Best_RF__Full_Dataset_2_Double_Stratified__1gram__sum_pool", # tuning-complete 
                                  "Best_RF__Full_Dataset_2_Double_Stratified__2gram__sum_pool", # tuning-complete # final-tested

                                  ], 
                                  default = ["RandomForest_searchspace_1"])

#PW: Why 10 Kfold? just common values
 # flatten vs no graph ?? is that only ML tuning differece??
   
    parser.add_argument('-graphemb_opt', '--graph_embedding_option', 
                        choices= [
                                  'thread_level__N>1_grams_events__nodetype5bit', 
                                 
                                  'thread_level__N>1_grams_events__nodetype5bit__FRNPeventCount', # implemented at 2024-1-27
                                  'thread_level__N>1_grams_events__nodetype5bit__FRNPeventCount__FRNP_OutgoIncom_eventCount',  # implemented at 2024-1-27

                                  'thread_level__N>1_grams_events__nodetype5bit__AvgNum_DiffThreads_perFRNP', # implemented at 2024-1-29

                                  'thread_level__N>1_grams_events__nodetype5bit__FRNPeventCount__FRNP_OutgoIncom_eventCount__AvgNum_DiffThreads_perFRNP', # TODO: implement

                                  ], 
                                  default = ['thread_level__N>1_grams_events__nodetype5bit'])
    
    parser.add_argument('-pool_opt', '--pool_option', 
                        choices= ['sum',
                                  'mean' # PW : also try
                                  ], 
                                  default = ["sum"])



    parser.add_argument("--search_on_train__or__final_test", 
                                 
                         choices= ["search_on_train", "final_test", "search_on_all"],  # TODO PW:use "final_test" on test dataset
                         #PW: serach on all- more robust, --> next to run
                                  
                         #default = ["search_on_train"] )
                         default = ["search_on_train"] )


    # --------- For Thread-level N-gram
    parser.add_argument('--N', nargs = 1, type = int, 
                        default = [4])  # Added by JY @ 2024-1-20


    parser.add_argument('--only_train_specified_Ngram', nargs = 1, type = bool, 
                        default = [True])  # Added by JY @ 2024-1-20



    # --------- JY @ 2024-1-23: For path resolve -- os.expanduser() also dependent on curr-dir, so better to do this way for now.
    parser.add_argument("--running_from_machine", 
                                 
                         choices= ["panther", "ocelot", "felis"], 
                         default = ["panther"] )
    
    parser.add_argument('--RF__n_jobs', nargs = 1, type = int, 
                        default = [30])  # Added by JY @ 2024-1-20

   # ==================================================================================================================================

    # cmd args
    K = parser.parse_args().K[0]
    model_choice = parser.parse_args().model_choice[0]
    model_cls = model_cls_map[ model_choice ]
    dataset_choice = parser.parse_args().dataset[0]

    graph_embedding_option = parser.parse_args().graph_embedding_option[0]
    pool_option = parser.parse_args().pool_option[0]
    Ngram = parser.parse_args().N[0] # for n-gram
    only_train_specified_Ngram = parser.parse_args().only_train_specified_Ngram[0]

    search_space_option = parser.parse_args().search_space_option[0]
    search_on_train__or__final_test = parser.parse_args().search_on_train__or__final_test[0] 

    running_from_machine = parser.parse_args().running_from_machine[0] 
    RF__n_jobs = parser.parse_args().RF__n_jobs[0] 
    # -----------------------------------------------------------------------------------------------------------------------------------
 
    if running_from_machine == "ocelot":
      abs_path_to_tabby = "/data/d1/jgwak1/tabby"
    else: # panther or felis
      abs_path_to_tabby = "/home/jgwak1/tabby" 
 
    model_cls_name = re.search(r"'(.*?)'", str(model_cls)).group(1)


    if search_on_train__or__final_test in {"search_on_train", "search_on_all"}:

      #  if "thread_level__N>1_grams_events__nodetype5bit" in graph_embedding_option:
      #    run_identifier = f"{model_choice}__{dataset_choice}__{search_space_option}__{K}_FoldCV__{search_on_train__or__final_test}__{graph_embedding_option}__{pool_option}_pool__{datetime.now().strftime('%Y-%m-%d_%H%M%S')}"
      #  else:
      #    run_identifier = f"{model_choice}__{dataset_choice}__{search_space_option}__{K}_FoldCV__{search_on_train__or__final_test}__{graph_embedding_option}__{datetime.now().strftime('%Y-%m-%d_%H%M%S')}"  

       run_identifier = f"{model_choice}__{dataset_choice}__{search_space_option}__{K}_FoldCV__{search_on_train__or__final_test}__{graph_embedding_option}__{Ngram}gram__{pool_option}_pool__only_train_specified_Ngram_{only_train_specified_Ngram}__{datetime.now().strftime('%Y-%m-%d_%H%M%S')}"
       this_results_dirpath = f"{abs_path_to_tabby}/graph_embedding_improvement_JY_git/graph_embedding_improvement_efforts/Trial_7__Thread_level_N_grams__N_gt_than_1__Similar_to_PriorGraphEmbedding/RESULTS/{run_identifier}"
       experiment_results_df_fpath = os.path.join(this_results_dirpath, f"{run_identifier}.csv")
       if not os.path.exists(this_results_dirpath):
           os.makedirs(this_results_dirpath)


    if search_on_train__or__final_test == "final_test":
      #  if "thread_level__N>1_grams_events__nodetype5bit" in graph_embedding_option:
      #  run_identifier = f"{model_choice}__{dataset_choice}__{search_space_option}__{search_on_train__or__final_test}__{graph_embedding_option}__{pool_option}_pool__{datetime.now().strftime('%Y-%m-%d_%H%M%S')}"
      #  else:
      #    run_identifier = f"{model_choice}__{dataset_choice}__{search_space_option}__{search_on_train__or__final_test}__{graph_embedding_option}__{datetime.now().strftime('%Y-%m-%d_%H%M%S')}"  

      #  run_identifier = f"{model_choice}__{dataset_choice}__{search_space_option}__{search_on_train__or__final_test}__{graph_embedding_option}__{Ngram}gram__{pool_option}_pool__only_train_specified_Ngram_{only_train_specified_Ngram}__{datetime.now().strftime('%Y-%m-%d_%H%M%S')}"
       run_identifier = f"{model_choice}__{dataset_choice}__{search_space_option}__{search_on_train__or__final_test}__{graph_embedding_option}__{Ngram}gram__{pool_option}_pool__{datetime.now().strftime('%Y-%m-%d_%H%M%S')}"

       this_results_dirpath = f"{abs_path_to_tabby}/graph_embedding_improvement_JY_git/graph_embedding_improvement_efforts/Trial_7__Thread_level_N_grams__N_gt_than_1__Similar_to_PriorGraphEmbedding/RESULTS/{run_identifier}"
       final_test_results_df_fpath = os.path.join(this_results_dirpath, f"{run_identifier}.csv")
       if not os.path.exists(this_results_dirpath):
           os.makedirs(this_results_dirpath)

    trace_filename = f'traces_stratkfold_double_strat_{model_cls_name}__generated@'+str(datetime.now())+".txt" 


    ###############################################################################################################################################

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
      # ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
      "Dataset-Case-3": \
        {"5": f"{abs_path_to_tabby}/Graph_embedding_aka_signal_amplification_files/Non_trace_commad_benign_dataset/train/Processed_Benign_ONLY_TaskName_edgeattr"},

      "Dataset-Case-3__FR_UID_rule_updated": \
        {"5": f"{abs_path_to_tabby}/graph_embedding_improvement_JY_git/making_CG_more_accurate/Subgraphs/Dataset_3_Benign/train"},

      # JY @ 2024-2-3
      'Dataset_1__NoTrace_UIDruleUpdated':\
        {"5": f"{abs_path_to_tabby}/PW_NON_TRACE_COMMAND_DATASET/Benign_Case1/train/Processed_Benign_ONLY_TaskName_edgeattr"}, # dim-node == 5
      'Dataset_2__NoTrace_UIDruleUpdated':\
        {"5": f"{abs_path_to_tabby}/PW_NON_TRACE_COMMAND_DATASET/Benign_Case2/train/Processed_Benign_ONLY_TaskName_edgeattr"},

      # JY @ 2024-2-4
      'Full_Dataset_1_NoTraceUIDupdated':\
        {"5": f"{abs_path_to_tabby}/PW_NON_TRACE_COMMAND_DATASET/Benign_Case1/Full_train_set/Processed_Benign_ONLY_TaskName_edgeattr", # dim-node == 5,
         "35":f"{abs_path_to_tabby}/PW_NON_TRACE_COMMAND_DATASET/Benign_Case1/Full_train_adhoc"}, # dim-node == 35 (adhoc)
      'Full_Dataset_2_NoTraceUIDupdated':\
        {"5": f"{abs_path_to_tabby}/PW_NON_TRACE_COMMAND_DATASET/Benign_Case2/Full_train_set/Processed_Benign_ONLY_TaskName_edgeattr",
         "35": f"{abs_path_to_tabby}/PW_NON_TRACE_COMMAND_DATASET/Benign_case2/Full_train_adhoc"},


      # JY @ 2024-2-5
      'Full_Dataset_1_Double_Stratified':\
        {"5": f"{abs_path_to_tabby}/graph_embedding_improvement_JY_git/making_CG_more_accurate/PW_NON_TRACE_COMMAND_DATASET/Benign_Case1/Full_train_set__double_strat"},
      'Full_Dataset_2_Double_Stratified':\
        {"5": f"{abs_path_to_tabby}/graph_embedding_improvement_JY_git/making_CG_more_accurate/PW_NON_TRACE_COMMAND_DATASET/Benign_Case2/Full_train_set__double_strat"},


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
      # ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
      "Dataset-Case-3": \
        {"5": f"{abs_path_to_tabby}/Graph_embedding_aka_signal_amplification_files/Non_trace_command_malware_dataset/train/Processed_Malware_ONLY_TaskName_edgeattr"},


      "Dataset-Case-3__FR_UID_rule_updated": \
        {"5": f"{abs_path_to_tabby}/graph_embedding_improvement_JY_git/making_CG_more_accurate/Subgraphs/Dataset_3_Malware/train"},


      # JY @ 2024-2-3
      'Dataset_1__NoTrace_UIDruleUpdated':\
        {"5": f"{abs_path_to_tabby}/PW_NON_TRACE_COMMAND_DATASET/Malware_Case1/train/Processed_Malware_ONLY_TaskName_edgeattr"}, # dim-node == 5
      'Dataset_2__NoTrace_UIDruleUpdated':\
        {"5": f"{abs_path_to_tabby}/PW_NON_TRACE_COMMAND_DATASET/Malware_Case2/train/Processed_Malware_ONLY_TaskName_edgeattr"},


      # JY @ 2024-2-4
      'Full_Dataset_1_NoTraceUIDupdated':\
        {"5":f"{abs_path_to_tabby}/PW_NON_TRACE_COMMAND_DATASET/Malware_Case1/Full_train_set/Processed_Malware_ONLY_TaskName_edgeattr",
         "35":f"{abs_path_to_tabby}/PW_NON_TRACE_COMMAND_DATASET/Malware_case1/Full_train_adhoc"},
      'Full_Dataset_2_NoTraceUIDupdated':\
        {"5": f"{abs_path_to_tabby}/PW_NON_TRACE_COMMAND_DATASET/Malware_Case2/Full_train_set/Processed_Malware_ONLY_TaskName_edgeattr",
         "35": f"{abs_path_to_tabby}/PW_NON_TRACE_COMMAND_DATASET/Malware_case2/Full_train_adhoc"},
      

      # JY @ 2024-2-5
      'Full_Dataset_1_Double_Stratified':\
        {"5": f"{abs_path_to_tabby}/graph_embedding_improvement_JY_git/making_CG_more_accurate/PW_NON_TRACE_COMMAND_DATASET/Malware_Case1/Full_train_set__double_strat"},
      'Full_Dataset_2_Double_Stratified':\
        {"5": f"{abs_path_to_tabby}/graph_embedding_improvement_JY_git/making_CG_more_accurate/PW_NON_TRACE_COMMAND_DATASET/Malware_Case2/Full_train_set__double_strat"},


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
      # ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
      "Dataset-Case-3": \
        {"5": f"{abs_path_to_tabby}/Graph_embedding_aka_signal_amplification_files/Non_trace_commad_benign_dataset/test/Processed_Benign_ONLY_TaskName_edgeattr"},

      "Dataset-Case-3__FR_UID_rule_updated": \
        {"5": f"{abs_path_to_tabby}/graph_embedding_improvement_JY_git/making_CG_more_accurate/Subgraphs/Dataset_3_Benign/test"},



      # JY @ 2024-2-3
      'Dataset_1__NoTrace_UIDruleUpdated':\
        {"5": f"{abs_path_to_tabby}/PW_NON_TRACE_COMMAND_DATASET/Benign_Case1/test/Processed_Benign_ONLY_TaskName_edgeattr"}, # dim-node == 5
      'Dataset_2__NoTrace_UIDruleUpdated':\
        {"5": f"{abs_path_to_tabby}/PW_NON_TRACE_COMMAND_DATASET/Benign_Case2/test/Processed_Benign_ONLY_TaskName_edgeattr"},

      # JY @ 2024-2-4
      'Full_Dataset_1_NoTraceUIDupdated':\
        {"5": f"{abs_path_to_tabby}/PW_NON_TRACE_COMMAND_DATASET/Benign_Case1/Full_test_set/Processed_Benign_ONLY_TaskName_edgeattr",
         "35": f"{abs_path_to_tabby}/PW_NON_TRACE_COMMAND_DATASET/Benign_Case1/Full_test_adhoc"},
      'Full_Dataset_2_NoTraceUIDupdated':\
        {"5": f"{abs_path_to_tabby}/PW_NON_TRACE_COMMAND_DATASET/Benign_Case2/Full_test_set/Processed_Benign_ONLY_TaskName_edgeattr",
         "35": f"{abs_path_to_tabby}/PW_NON_TRACE_COMMAND_DATASET/Benign_Case2/Full_test_adhoc"},


      # JY @ 2024-2-5
      'Full_Dataset_1_Double_Stratified':\
        {"5": f"{abs_path_to_tabby}/graph_embedding_improvement_JY_git/making_CG_more_accurate/PW_NON_TRACE_COMMAND_DATASET/Benign_Case1/Full_test_set__double_strat"},
      'Full_Dataset_2_Double_Stratified':\
        {"5": f"{abs_path_to_tabby}/graph_embedding_improvement_JY_git/making_CG_more_accurate/PW_NON_TRACE_COMMAND_DATASET/Benign_Case2/Full_test_set__double_strat"},

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
      # ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
      "Dataset-Case-3": \
        {"5": f"{abs_path_to_tabby}/Graph_embedding_aka_signal_amplification_files/Non_trace_command_malware_dataset/test/Processed_Malware_ONLY_TaskName_edgeattr"},

      "Dataset-Case-3__FR_UID_rule_updated": \
        {"5": f"{abs_path_to_tabby}/graph_embedding_improvement_JY_git/making_CG_more_accurate/Subgraphs/Dataset_3_Malware/test"},

      # JY @ 2024-2-3
      'Dataset_1__NoTrace_UIDruleUpdated':\
        {"5": f"{abs_path_to_tabby}/PW_NON_TRACE_COMMAND_DATASET/Malware_Case1/test/Processed_Malware_ONLY_TaskName_edgeattr"}, # dim-node == 5
      'Dataset_2__NoTrace_UIDruleUpdated':\
        {"5": f"{abs_path_to_tabby}/PW_NON_TRACE_COMMAND_DATASET/Malware_Case2/test/Processed_Malware_ONLY_TaskName_edgeattr"},

      # JY @ 2024-2-4
      'Full_Dataset_1_NoTraceUIDupdated':\
        {"5": f"{abs_path_to_tabby}/PW_NON_TRACE_COMMAND_DATASET/Malware_Case1/Full_test_set/Processed_Malware_ONLY_TaskName_edgeattr",
         "35": f"{abs_path_to_tabby}/PW_NON_TRACE_COMMAND_DATASET/Malware_Case1/Full_test_adhoc"},
      'Full_Dataset_2_NoTraceUIDupdated':\
        {"5": f"{abs_path_to_tabby}/PW_NON_TRACE_COMMAND_DATASET/Malware_Case2/Full_test_set/Processed_Malware_ONLY_TaskName_edgeattr",
         "35": f"{abs_path_to_tabby}/PW_NON_TRACE_COMMAND_DATASET/Malware_Case2/Full_test_adhoc"},

      # JY @ 2024-2-5
      'Full_Dataset_1_Double_Stratified':\
        {"5": f"{abs_path_to_tabby}/graph_embedding_improvement_JY_git/making_CG_more_accurate/PW_NON_TRACE_COMMAND_DATASET/Malware_Case1/Full_test_set__double_strat"},
      'Full_Dataset_2_Double_Stratified':\
        {"5": f"{abs_path_to_tabby}/graph_embedding_improvement_JY_git/making_CG_more_accurate/PW_NON_TRACE_COMMAND_DATASET/Malware_Case2/Full_test_set__double_strat"},

    }

    _num_classes = 2  # number of class labels and always binary classification.

    if "Adhoc" in graph_embedding_option: # JY @ 2024-1-20: "Adhoc" grpah-embeddings are not implemented yet.
        _dim_node = 35 #46   # num node features ; the #feats    
    else:
        _dim_node = 5 #46   # num node features ; the #feats


    #PW: 5 and 62 (61 taskname + 1 timestamp) based on new silketw
    _dim_edge = 62 #72    # (or edge_dim) ; num edge features

    _benign_train_data_path = projection_datapath_Benign_Train_dict[dataset_choice][str(_dim_node)]
    _malware_train_data_path = projection_datapath_Malware_Train_dict[dataset_choice][str(_dim_node)]
    _benign_final_test_data_path = projection_datapath_Benign_Test_dict[dataset_choice][str(_dim_node)]
    _malware_final_test_data_path = projection_datapath_Malware_Test_dict[dataset_choice][str(_dim_node)]


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
                                                      }
                                                      )
         return manual_space


   #  def Best_RF__Dataset_1__2gram__sum_pool__only_train_specified_Ngram_True() -> dict :
   #    # RandomForest__Dataset-Case-1__RandomForest_searchspace_1__10_FoldCV__search_on_train__thread_level__N>1_grams_events__nodetype_5bit__2gram__sum_pool__only_train_specified_Ngram_True__2024-01-20_202742
   #    # -- tuning done
   #       manual_space = []
   #       manual_space.append(
   #          {'bootstrap': False,
   #          'criterion': 'gini',
   #          'max_depth': 20,
   #          'max_features': 'log2',
   #          'min_samples_leaf': 1,
   #          'min_samples_split': 2,
   #          'n_estimators': 100,
   #          'random_state': 42,
   #          'split_shuffle_seed': 100}
   #       )

   #       return manual_space


   #  def Best_RF__Dataset_1__4gram__sum_pool__only_train_specified_Ngram_True() -> dict :
   #    # RandomForest__Dataset-Case-1__RandomForest_searchspace_1__10_FoldCV__search_on_train__thread_level__N>1_grams_events__nodetype_5bit__4gram__sum_pool__only_train_specified_Ngram_True__2024-01-20_202813
   #    # -- tuning done
   #       manual_space = []
   #       manual_space.append(
   #          {'bootstrap': True,
   #          'criterion': 'gini',
   #          'max_depth': None,
   #          'max_features': None,
   #          'min_samples_leaf': 1,
   #          'min_samples_split': 5,
   #          'n_estimators': 200,
   #          'random_state': 42,
   #          'split_shuffle_seed': 100}
   #       )
   #       return manual_space
      
   #  def Best_RF__Dataset_1__6gram__sum_pool__only_train_specified_Ngram_True() -> dict :
   #    # -- tuning NOT done
   #       manual_space = []
   #       manual_space.append(
   #          {'bootstrap': True,
   #          'criterion': 'gini',
   #          'max_depth': 6,
   #          'max_features': None,
   #          'min_samples_leaf': 1,
   #          'min_samples_split': 15,
   #          'n_estimators': 100,
   #          'random_state': 99,
   #          'split_shuffle_seed': 100}
   #       )
   #       return manual_space



   #  def Best_RF__Dataset_2__2gram__sum_pool__only_train_specified_Ngram_True() -> dict :
   #    # RandomForest__Dataset-Case-2__RandomForest_searchspace_1__10_FoldCV__search_on_train__thread_level__N>1_grams_events__nodetype_5bit__2gram__sum_pool__only_train_specified_Ngram_True__2024-01-20_202900
   #    # -- tuning done
   #       manual_space = []
   #       manual_space.append(
   #          {'bootstrap': False,
   #          'criterion': 'gini',
   #          'max_depth': 20,
   #          'max_features': 'sqrt',
   #          'min_samples_leaf': 1,
   #          'min_samples_split': 2,
   #          'n_estimators': 300,
   #          'random_state': 99,
   #          'split_shuffle_seed': 100}
   #       )

   #       return manual_space
      
   #  def Best_RF__Dataset_2__4gram__sum_pool__only_train_specified_Ngram_True() -> dict :
   #    # RandomForest__Dataset-Case-2__RandomForest_searchspace_1__10_FoldCV__search_on_train__thread_level__N>1_grams_events__nodetype_5bit__4gram__sum_pool__only_train_specified_Ngram_True__2024-01-20_202918
   #    # -- tuning NOT done
   #       manual_space = []
   #       manual_space.append(
   #          {'bootstrap': True,
   #          'criterion': 'gini',
   #          'max_depth': None,
   #          'max_features': None,
   #          'min_samples_leaf': 5,
   #          'min_samples_split': 2,
   #          'n_estimators': 100,
   #          'random_state': 42,
   #          'split_shuffle_seed': 100}

   #       )

   #       return manual_space
            
   #  def Best_RF__Dataset_2__6gram__sum_pool__only_train_specified_Ngram_True() -> dict :
   #    # -- tuning NOT done
   #       manual_space = []
   #       manual_space.append(
   #          {'bootstrap': False,
   #          'criterion': 'gini',
   #          'max_depth': None,
   #          'max_features': 'sqrt',
   #          'min_samples_leaf': 1,
   #          'min_samples_split': 15,
   #          'n_estimators': 100,
   #          'random_state': 0,
   #          'split_shuffle_seed': 100}
   #       )
   #       return manual_space

   #  def Best_RF__Dataset_1__2gram__also_with__FRNPeventCount__FRNP_OutgoIncom_eventCount() -> dict :
   #    # /data/d1/jgwak1/tabby/graph_embedding_improvement_JY_git/graph_embedding_improvement_efforts/Trial_7__Thread_level_N_grams__N_gt_than_1__Similar_to_PriorGraphEmbedding/RESULTS/RandomForest__Dataset-Case-1__RandomForest_searchspace_1__10_FoldCV__search_on_train__thread_level__N>1_grams_events__nodetype5bit__FRNPeventCount__FRNP_OutgoIncom_eventCount__2gram__sum_pool__only_train_specified_Ngram_True__2024-01-27_141047/RandomForest__Dataset-Case-1__RandomForest_searchspace_1__10_FoldCV__search_on_train__thread_level__N>1_grams_events__nodetype5bit__FRNPeventCount__FRNP_OutgoIncom_eventCount__2gram__sum_pool__only_train_specified_Ngram_True__2024-01-27_141047.csv
   #    # -- tuning done
   #       manual_space = []
   #       manual_space.append(
   #             {'bootstrap': True,
   #             'criterion': 'gini',
   #             'max_depth': None,
   #             'max_features': None,
   #             'min_samples_leaf': 1,
   #             'min_samples_split': 2,
   #             'n_estimators': 200,
   #             'random_state': 42,
   #             'split_shuffle_seed': 100}
   #       )
   #       return manual_space

   #  def Best_RF__Dataset_1__4gram__also_with__FRNPeventCount__FRNP_OutgoIncom_eventCount() -> dict :
   #    # /data/d1/jgwak1/tabby/graph_embedding_improvement_JY_git/graph_embedding_improvement_efforts/Trial_7__Thread_level_N_grams__N_gt_than_1__Similar_to_PriorGraphEmbedding/RESULTS/RandomForest__Dataset-Case-1__RandomForest_searchspace_1__10_FoldCV__search_on_train__thread_level__N>1_grams_events__nodetype5bit__FRNPeventCount__FRNP_OutgoIncom_eventCount__4gram__sum_pool__only_train_specified_Ngram_True__2024-01-27_140946/RandomForest__Dataset-Case-1__RandomForest_searchspace_1__10_FoldCV__search_on_train__thread_level__N>1_grams_events__nodetype5bit__FRNPeventCount__FRNP_OutgoIncom_eventCount__4gram__sum_pool__only_train_specified_Ngram_True__2024-01-27_140946.csv
   #       manual_space = []
   #       manual_space.append(
   #          {'bootstrap': True,
   #          'criterion': 'gini',
   #          'max_depth': None,
   #          'max_features': None,
   #          'min_samples_leaf': 1,
   #          'min_samples_split': 5,
   #          'n_estimators': 200,
   #          'random_state': 0,
   #          'split_shuffle_seed': 100}
   #       )
   #       return manual_space

   #  def Best_RF__Dataset_3_FR_UID_rule_updated__2gram_with__AvgNum_DiffThreads_perFRNP() -> dict :
   #       # /data/d1/jgwak1/tabby/graph_embedding_improvement_JY_git/graph_embedding_improvement_efforts/Trial_7__Thread_level_N_grams__N_gt_than_1__Similar_to_PriorGraphEmbedding/RESULTS/RandomForest__Dataset-Case-3__FR_UID_rule_updated__RandomForest_searchspace_1__10_FoldCV__search_on_train__thread_level__N>1_grams_events__nodetype5bit__AvgNum_DiffThreads_perFRNP__2gram__sum_pool__only_train_specified_Ngram_True__2024-01-31_232601/RandomForest__Dataset-Case-3__FR_UID_rule_updated__RandomForest_searchspace_1__10_FoldCV__search_on_train__thread_level__N>1_grams_events__nodetype5bit__AvgNum_DiffThreads_perFRNP__2gram__sum_pool__only_train_specified_Ngram_True__2024-01-31_232601.csv
   #       manual_space = []
   #       manual_space.append(
   #          {'bootstrap': False,
   #          'criterion': 'gini',
   #          'max_depth': None,
   #          'max_features': 'log2',
   #          'min_samples_leaf': 1,
   #          'min_samples_split': 2,
   #          'n_estimators': 500,
   #          'random_state': 42,
   #          'split_shuffle_seed': 100}
   #       )
   #       return manual_space




    def Best_RF__Partial_Dataset_1_NoTraceUIDUpdated__1gram__sum_pool() -> dict :
         # /home/jgwak1/tabby/graph_embedding_improvement_JY_git/graph_embedding_improvement_efforts/Trial_7__Thread_level_N_grams__N_gt_than_1__Similar_to_PriorGraphEmbedding/RESULTS/RandomForest__Dataset_1__NoTrace_UIDruleUpdated__RandomForest_searchspace_1__10_FoldCV__search_on_train__thread_level__N>1_grams_events__nodetype5bit__1gram__sum_pool__only_train_specified_Ngram_True__2024-02-03_102515/RandomForest__Dataset_1__NoTrace_UIDruleUpdated__RandomForest_searchspace_1__10_FoldCV__search_on_train__thread_level__N>1_grams_events__nodetype5bit__1gram__sum_pool__only_train_specified_Ngram_True__2024-02-03_102515.csv
         manual_space = []
         manual_space.append(
               {'bootstrap': False,
               'criterion': 'gini',
               'max_depth': 20,
               'max_features': 'sqrt',
               'min_samples_leaf': 1,
               'min_samples_split': 2,
               'n_estimators': 500,
               'random_state': 42,
               'split_shuffle_seed': 100}             
         )
         return manual_space


    def Best_RF__Partial_Dataset_1_NoTraceUIDUpdated__2gram__sum_pool() -> dict :
         # /home/jgwak1/tabby/graph_embedding_improvement_JY_git/graph_embedding_improvement_efforts/Trial_7__Thread_level_N_grams__N_gt_than_1__Similar_to_PriorGraphEmbedding/RESULTS/RandomForest__Dataset_1__NoTrace_UIDruleUpdated__RandomForest_searchspace_1__10_FoldCV__search_on_train__thread_level__N>1_grams_events__nodetype5bit__2gram__sum_pool__only_train_specified_Ngram_True__2024-02-03_102737/RandomForest__Dataset_1__NoTrace_UIDruleUpdated__RandomForest_searchspace_1__10_FoldCV__search_on_train__thread_level__N>1_grams_events__nodetype5bit__2gram__sum_pool__only_train_specified_Ngram_True__2024-02-03_102737.csv
         manual_space = []
         manual_space.append(
            {'bootstrap': False,
            'criterion': 'gini',
            'max_depth': 15,
            'max_features': 'sqrt',
            'min_samples_leaf': 1,
            'min_samples_split': 5,
            'n_estimators': 100,
            'random_state': 99,
            'split_shuffle_seed': 100}
         )
         return manual_space

    def Best_RF__Partial_Dataset_1_NoTraceUIDUpdated__4gram__sum_pool() -> dict :
         # /home/jgwak1/tabby/graph_embedding_improvement_JY_git/graph_embedding_improvement_efforts/Trial_7__Thread_level_N_grams__N_gt_than_1__Similar_to_PriorGraphEmbedding/RESULTS/RandomForest__Dataset_1__NoTrace_UIDruleUpdated__RandomForest_searchspace_1__10_FoldCV__search_on_train__thread_level__N>1_grams_events__nodetype5bit__4gram__sum_pool__only_train_specified_Ngram_True__2024-02-03_102937/RandomForest__Dataset_1__NoTrace_UIDruleUpdated__RandomForest_searchspace_1__10_FoldCV__search_on_train__thread_level__N>1_grams_events__nodetype5bit__4gram__sum_pool__only_train_specified_Ngram_True__2024-02-03_102937.csv
         manual_space = []
         manual_space.append(
            {'bootstrap': False,
            'criterion': 'gini',
            'max_depth': None,
            'max_features': 'sqrt',
            'min_samples_leaf': 1,
            'min_samples_split': 2,
            'n_estimators': 500,
            'random_state': 42,
            'split_shuffle_seed': 100}
         )
         return manual_space        



    def Best_RF__Partial_Dataset_2_NoTraceUIDUpdated__1gram__sum_pool() -> dict :
         # /home/jgwak1/tabby/graph_embedding_improvement_JY_git/graph_embedding_improvement_efforts/Trial_7__Thread_level_N_grams__N_gt_than_1__Similar_to_PriorGraphEmbedding/RESULTS/RandomForest__Dataset_2__NoTrace_UIDruleUpdated__RandomForest_searchspace_1__10_FoldCV__search_on_train__thread_level__N>1_grams_events__nodetype5bit__1gram__sum_pool__only_train_specified_Ngram_True__2024-02-03_102605/RandomForest__Dataset_2__NoTrace_UIDruleUpdated__RandomForest_searchspace_1__10_FoldCV__search_on_train__thread_level__N>1_grams_events__nodetype5bit__1gram__sum_pool__only_train_specified_Ngram_True__2024-02-03_102605.csv
         manual_space = []
         manual_space.append(
            {'bootstrap': False,
            'criterion': 'gini',
            'max_depth': 20,
            'max_features': 'log2',
            'min_samples_leaf': 1,
            'min_samples_split': 15,
            'n_estimators': 100,
            'random_state': 0,
            'split_shuffle_seed': 100}             
         )
         return manual_space
    

    def Best_RF__Partial_Dataset_2_NoTraceUIDUpdated__2gram__sum_pool() -> dict :
         # /home/jgwak1/tabby/graph_embedding_improvement_JY_git/graph_embedding_improvement_efforts/Trial_7__Thread_level_N_grams__N_gt_than_1__Similar_to_PriorGraphEmbedding/RESULTS/RandomForest__Dataset_2__NoTrace_UIDruleUpdated__RandomForest_searchspace_1__10_FoldCV__search_on_train__thread_level__N>1_grams_events__nodetype5bit__2gram__sum_pool__only_train_specified_Ngram_True__2024-02-03_102809/RandomForest__Dataset_2__NoTrace_UIDruleUpdated__RandomForest_searchspace_1__10_FoldCV__search_on_train__thread_level__N>1_grams_events__nodetype5bit__2gram__sum_pool__only_train_specified_Ngram_True__2024-02-03_102809.csv
         manual_space = []
         manual_space.append(
            {'bootstrap': False,
            'criterion': 'gini',
            'max_depth': 20,
            'max_features': 'sqrt',
            'min_samples_leaf': 1,
            'min_samples_split': 10,
            'n_estimators': 500,
            'random_state': 0,
            'split_shuffle_seed': 100}             
         )
         return manual_space


    def Best_RF__Partial_Dataset_2_NoTraceUIDUpdated__4gram__sum_pool() -> dict :
         # /home/jgwak1/tabby/graph_embedding_improvement_JY_git/graph_embedding_improvement_efforts/Trial_7__Thread_level_N_grams__N_gt_than_1__Similar_to_PriorGraphEmbedding/RESULTS/RandomForest__Dataset_2__NoTrace_UIDruleUpdated__RandomForest_searchspace_1__10_FoldCV__search_on_train__thread_level__N>1_grams_events__nodetype5bit__4gram__sum_pool__only_train_specified_Ngram_True__2024-02-03_103053/RandomForest__Dataset_2__NoTrace_UIDruleUpdated__RandomForest_searchspace_1__10_FoldCV__search_on_train__thread_level__N>1_grams_events__nodetype5bit__4gram__sum_pool__only_train_specified_Ngram_True__2024-02-03_103053.csv
         manual_space = []
         manual_space.append(
            {'bootstrap': False,
            'criterion': 'gini',
            'max_depth': 20,
            'max_features': 'sqrt',
            'min_samples_leaf': 1,
            'min_samples_split': 2,
            'n_estimators': 200,
            'random_state': 0,
            'split_shuffle_seed': 100}

         )
         return manual_space
    


    def Best_RF__Full_Dataset_1_Double_Stratified__1gram__sum_pool() -> dict:
         # /home/jgwak1/tabby/graph_embedding_improvement_JY_git/graph_embedding_improvement_efforts/Trial_7__Thread_level_N_grams__N_gt_than_1__Similar_to_PriorGraphEmbedding/RESULTS/RandomForest__Full_Dataset_1_Double_Stratified__RandomForest_searchspace_1__10_FoldCV__search_on_train__thread_level__N>1_grams_events__nodetype5bit__1gram__sum_pool__only_train_specified_Ngram_True__2024-02-06_134503/RandomForest__Full_Dataset_1_Double_Stratified__RandomForest_searchspace_1__10_FoldCV__search_on_train__thread_level__N>1_grams_events__nodetype5bit__1gram__sum_pool__only_train_specified_Ngram_True__2024-02-06_134503.csv
         manual_space = []
         manual_space.append(
               {'bootstrap': False,
               'criterion': 'gini',
               'max_depth': 20,
               'max_features': 'sqrt',
               'min_samples_leaf': 1,
               'min_samples_split': 5,
               'n_estimators': 100,
               'random_state': 0,
               'split_shuffle_seed': 100}      
         )
         return manual_space
    


    def Best_RF__Full_Dataset_1_Double_Stratified__2gram__sum_pool() -> dict:        
         # /home/jgwak1/tabby/graph_embedding_improvement_JY_git/graph_embedding_improvement_efforts/Trial_7__Thread_level_N_grams__N_gt_than_1__Similar_to_PriorGraphEmbedding/RESULTS/RandomForest__Full_Dataset_1_Double_Stratified__RandomForest_searchspace_1__10_FoldCV__search_on_train__thread_level__N>1_grams_events__nodetype5bit__2gram__sum_pool__only_train_specified_Ngram_True__2024-02-05_230350/RandomForest__Full_Dataset_1_Double_Stratified__RandomForest_searchspace_1__10_FoldCV__search_on_train__thread_level__N>1_grams_events__nodetype5bit__2gram__sum_pool__only_train_specified_Ngram_True__2024-02-05_230350.csv
         manual_space = []
         manual_space.append(
               {'bootstrap': False,
               'criterion': 'gini',
               'max_depth': 15,
               'max_features': 'sqrt',
               'min_samples_leaf': 1,
               'min_samples_split': 2,
               'n_estimators': 200,
               'random_state': 42,
               'split_shuffle_seed': 100}             
         )
         return manual_space
    


    def Best_RF__Full_Dataset_2_Double_Stratified__1gram__sum_pool() -> dict:
         # /home/jgwak1/tabby/graph_embedding_improvement_JY_git/graph_embedding_improvement_efforts/Trial_7__Thread_level_N_grams__N_gt_than_1__Similar_to_PriorGraphEmbedding/RESULTS/RandomForest__Full_Dataset_2_Double_Stratified__RandomForest_searchspace_1__10_FoldCV__search_on_train__thread_level__N>1_grams_events__nodetype5bit__1gram__sum_pool__only_train_specified_Ngram_True__2024-02-07_145320/RandomForest__Full_Dataset_2_Double_Stratified__RandomForest_searchspace_1__10_FoldCV__search_on_train__thread_level__N>1_grams_events__nodetype5bit__1gram__sum_pool__only_train_specified_Ngram_True__2024-02-07_145320.csv
         manual_space = []
         manual_space.append(
            {'bootstrap': True,
            'criterion': 'gini',
            'max_depth': None,
            'max_features': None,
            'min_samples_leaf': 1,
            'min_samples_split': 10,
            'n_estimators': 500,
            'random_state': 0,
            'split_shuffle_seed': 100}            
         )
         return manual_space
          



    def Best_RF__Full_Dataset_2_Double_Stratified__2gram__sum_pool() -> dict:
         # /home/jgwak1/tabby/graph_embedding_improvement_JY_git/graph_embedding_improvement_efforts/Trial_7__Thread_level_N_grams__N_gt_than_1__Similar_to_PriorGraphEmbedding/RESULTS/RandomForest__Full_Dataset_2_Double_Stratified__RandomForest_searchspace_1__10_FoldCV__search_on_train__thread_level__N>1_grams_events__nodetype5bit__2gram__sum_pool__only_train_specified_Ngram_True__2024-02-05_230519/RandomForest__Full_Dataset_2_Double_Stratified__RandomForest_searchspace_1__10_FoldCV__search_on_train__thread_level__N>1_grams_events__nodetype5bit__2gram__sum_pool__only_train_specified_Ngram_True__2024-02-05_230519.csv
         manual_space = []
         manual_space.append(
            {'bootstrap': True,
            'criterion': 'gini',
            'max_depth': 15,
            'max_features': None,
            'min_samples_leaf': 3,
            'min_samples_split': 2,
            'n_estimators': 500,
            'random_state': 0,
            'split_shuffle_seed': 100}           
         )
         return manual_space
    

    

    ####################################################################################################################################################


    if search_space_option == "XGBoost_searchspace_1":
       search_space = XGBoost_searchspace_1()   

    elif search_space_option == "RandomForest_searchspace_1":
       search_space = RandomForest_searchspace_1()   


    # defaults
    elif search_space_option == "XGBoost_default_hyperparam":
       search_space = XGBoost_default_hyperparam()   

    elif search_space_option == "RandomForest_default_hyperparam":
       search_space = RandomForest_default_hyperparam()   


      

    # -----------------------------------------------------------

   #  elif search_space_option == 'Best_RF__Dataset_1__2gram__sum_pool__only_train_specified_Ngram_True':
   #     search_space = Best_RF__Dataset_1__2gram__sum_pool__only_train_specified_Ngram_True()   

   #  elif search_space_option == 'Best_RF__Dataset_1__4gram__sum_pool__only_train_specified_Ngram_True':
   #     search_space = Best_RF__Dataset_1__4gram__sum_pool__only_train_specified_Ngram_True()   

   #  elif search_space_option == 'Best_RF__Dataset_1__6gram__sum_pool__only_train_specified_Ngram_True':
   #     search_space = Best_RF__Dataset_1__6gram__sum_pool__only_train_specified_Ngram_True()   


   #  elif search_space_option == 'Best_RF__Dataset_2__2gram__sum_pool__only_train_specified_Ngram_True':
   #     search_space = Best_RF__Dataset_2__2gram__sum_pool__only_train_specified_Ngram_True()   

   #  elif search_space_option == 'Best_RF__Dataset_2__4gram__sum_pool__only_train_specified_Ngram_True':
   #     search_space = Best_RF__Dataset_2__4gram__sum_pool__only_train_specified_Ngram_True()   

   #  elif search_space_option == 'Best_RF__Dataset_2__6gram__sum_pool__only_train_specified_Ngram_True':
   #     search_space = Best_RF__Dataset_2__6gram__sum_pool__only_train_specified_Ngram_True()   

   #  # -----------------------------------------------------------

   #  elif search_space_option == 'Best_RF__Dataset_1__2gram__FRNPeventCount__FRNP_OutgoIncom_eventCount':
   #     search_space = Best_RF__Dataset_1__2gram__also_with__FRNPeventCount__FRNP_OutgoIncom_eventCount()   
       
   #  elif search_space_option == 'Best_RF__Dataset_1__4gram__FRNPeventCount__FRNP_OutgoIncom_eventCount':
   #     search_space = Best_RF__Dataset_1__4gram__also_with__FRNPeventCount__FRNP_OutgoIncom_eventCount()   

   #  elif search_space_option == 'Best_RF__Dataset_3_FR_UID_rule_updated__2gram_with__AvgNum_DiffThreads_perFRNP':
   #     search_space = Best_RF__Dataset_3_FR_UID_rule_updated__2gram_with__AvgNum_DiffThreads_perFRNP()   
   #  # -----------------------------------------------------------
    # No-Trace & FID-rule updated 


    elif search_space_option == 'Best_RF__Partial_Dataset_1_NoTraceUIDUpdated__1gram__sum_pool': # tuning done
            search_space = Best_RF__Partial_Dataset_1_NoTraceUIDUpdated__1gram__sum_pool()   
           
    elif search_space_option == 'Best_RF__Partial_Dataset_1_NoTraceUIDUpdated__2gram__sum_pool': # tuning done
            search_space = Best_RF__Partial_Dataset_1_NoTraceUIDUpdated__2gram__sum_pool()

    elif search_space_option == "Best_RF__Partial_Dataset_1_NoTraceUIDUpdated__4gram__sum_pool": # tuning done
            search_space = Best_RF__Partial_Dataset_1_NoTraceUIDUpdated__4gram__sum_pool()


    elif search_space_option == 'Best_RF__Partial_Dataset_2_NoTraceUIDUpdated__1gram__sum_pool': # tuning done
            search_space = Best_RF__Partial_Dataset_2_NoTraceUIDUpdated__1gram__sum_pool()

    elif search_space_option == 'Best_RF__Partial_Dataset_2_NoTraceUIDUpdated__2gram__sum_pool': # tuning done
            search_space = Best_RF__Partial_Dataset_2_NoTraceUIDUpdated__2gram__sum_pool()

    elif search_space_option == 'Best_RF__Partial_Dataset_2_NoTraceUIDUpdated__4gram__sum_pool': # tuning done
            search_space = Best_RF__Partial_Dataset_2_NoTraceUIDUpdated__4gram__sum_pool()

    # -----------------------------------------------------------

    elif search_space_option == "Best_RF__Full_Dataset_1_Double_Stratified__1gram__sum_pool": # tuning done
            search_space = Best_RF__Full_Dataset_1_Double_Stratified__1gram__sum_pool()

    elif search_space_option == "Best_RF__Full_Dataset_1_Double_Stratified__2gram__sum_pool": # tuning done
            search_space = Best_RF__Full_Dataset_1_Double_Stratified__2gram__sum_pool()

    elif search_space_option == "Best_RF__Full_Dataset_2_Double_Stratified__1gram__sum_pool": # tuning done
            search_space = Best_RF__Full_Dataset_2_Double_Stratified__1gram__sum_pool()



    elif search_space_option == "Best_RF__Full_Dataset_2_Double_Stratified__2gram__sum_pool": # tuning done
            search_space = Best_RF__Full_Dataset_2_Double_Stratified__2gram__sum_pool()


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

    if search_on_train__or__final_test == "search_on_all":
      train_dataset = train_dataset + final_test_dataset



   # Now apply signal-amplification here (here least conflicts with existing code.)
    if graph_embedding_option == "thread_level__N>1_grams_events__nodetype5bit":
         
         pretrained_Ngram_countvectorizer_list = pretrain__countvectorizer_on_training_set__before_graph_embedding_generation( Ngram = Ngram,
                                                                                                                               dataset= train_dataset,
                                                                                                                               only_train_specified_Ngram = only_train_specified_Ngram 
                                                                                                                              )


         train_dataset__signal_amplified_dict = get_thread_level_N_gram_events__nodetype5bit__dict( pretrained_Ngram_countvectorizer_list = pretrained_Ngram_countvectorizer_list,
                                                                                                    dataset= train_dataset,
                                                                                                    pool = pool_option
                                                                                                   )


         nodetype_names = ["file", "registry", "network", "process", "thread"] 
         Ngram_edge_feature_names = []
         for pretrained_Ngram_countvectorizer in pretrained_Ngram_countvectorizer_list:
             Ngram_edge_feature_names += pretrained_Ngram_countvectorizer.get_feature_names_out().tolist()
         feature_names = nodetype_names + Ngram_edge_feature_names # yes this order is correct


    elif graph_embedding_option == "thread_level__N>1_grams_events__nodetype5bit__FRNPeventCount":
        
         pretrained_Ngram_countvectorizer_list = pretrain__countvectorizer_on_training_set__before_graph_embedding_generation( Ngram = Ngram,
                                                                                                                               dataset= train_dataset,
                                                                                                                               only_train_specified_Ngram = only_train_specified_Ngram 
                                                                                                                              )


         train_dataset__signal_amplified_dict = get_thread_level_N_gram_events__nodetype5bit__FRNPeventCount__dict( pretrained_Ngram_countvectorizer_list = pretrained_Ngram_countvectorizer_list,
                                                                                                                    dataset= train_dataset,
                                                                                                                    pool = pool_option
                                                                                                                   )

         nodetype_names = ["file", "registry", "network", "process", "thread"] 
         FRNP_event_count_features = ["file_event_cnt", "registry_event_cnt", "network_event_cnt", "process_event_cnt" ]

         # file_node_tensor = torch.tensor([1, 0, 0, 0, 0])
         # reg_node_tensor = torch.tensor([0, 1, 0, 0, 0])
         # net_node_tensor = torch.tensor([0, 0, 1, 0, 0])
         # proc_node_tensor = torch.tensor([0, 0, 0, 1, 0])
         # thread_node_tensor = torch.tensor([0, 0, 0, 0, 1])

         Ngram_edge_feature_names = []
         for pretrained_Ngram_countvectorizer in pretrained_Ngram_countvectorizer_list:
             Ngram_edge_feature_names += pretrained_Ngram_countvectorizer.get_feature_names_out().tolist()
         # JY @ 2024-1-27
         feature_names = nodetype_names + FRNP_event_count_features + \
                         Ngram_edge_feature_names # yes this order is correct


    elif graph_embedding_option == "thread_level__N>1_grams_events__nodetype5bit__FRNPeventCount__FRNP_OutgoIncom_eventCount":

         pretrained_Ngram_countvectorizer_list = pretrain__countvectorizer_on_training_set__before_graph_embedding_generation( Ngram = Ngram,
                                                                                                                               dataset= train_dataset,
                                                                                                                               only_train_specified_Ngram = only_train_specified_Ngram 
                                                                                                                              )


         train_dataset__signal_amplified_dict = get_thread_level_N_gram_events__nodetype5bit__FRNPeventCount__FRNP_OutgoIncom_eventCount__dict( 
                                                                                                                    pretrained_Ngram_countvectorizer_list = pretrained_Ngram_countvectorizer_list,
                                                                                                                    dataset= train_dataset,
                                                                                                                    pool = pool_option
                                                                                                                   )

         nodetype_names = ["file", "registry", "network", "process", "thread"] 
         FRNP_event_count_features = ["file_event_cnt", "registry_event_cnt", "network_event_cnt", "process_event_cnt" ]
         FRNP_outgoing_and_incoming_events_count_features = \
                                    ["file_outgoing_event_cnt", "registry_outgoing_event_cnt", "network_outgoing_event_cnt", "process_outgoing_event_cnt",
                                     "file_incoming_event_cnt", "registry_incoming_event_cnt", "network_incoming_event_cnt", "process_incoming_event_cnt"]   

         Ngram_edge_feature_names = []
         for pretrained_Ngram_countvectorizer in pretrained_Ngram_countvectorizer_list:
             Ngram_edge_feature_names += pretrained_Ngram_countvectorizer.get_feature_names_out().tolist()
         # JY @ 2024-1-27
         feature_names = nodetype_names + FRNP_event_count_features + FRNP_outgoing_and_incoming_events_count_features +\
                         Ngram_edge_feature_names # yes this order is correct


    elif graph_embedding_option == "thread_level__N>1_grams_events__nodetype5bit__AvgNum_DiffThreads_perFRNP":

         pretrained_Ngram_countvectorizer_list = pretrain__countvectorizer_on_training_set__before_graph_embedding_generation( Ngram = Ngram,
                                                                                                                               dataset= train_dataset,
                                                                                                                               only_train_specified_Ngram = only_train_specified_Ngram 
                                                                                                                              )
         train_dataset__signal_amplified_dict = get_thread_level_N_gram_events__nodetype5bit__AvgNum_DiffThreads_perFRNP__dict( 
                                                                                                                    pretrained_Ngram_countvectorizer_list = pretrained_Ngram_countvectorizer_list,
                                                                                                                    dataset= train_dataset,
                                                                                                                    pool = pool_option
                                                                                                                   )
         nodetype_names = ["file", "registry", "network", "process", "thread"] 
         FRNP_AvgNum_DiffThreads = ["file_avg_diff_threads", "registry_avg_diff_threads", "network_avg_diff_threads", "process_avg_diff_threads" ]
 
         Ngram_edge_feature_names = []
         for pretrained_Ngram_countvectorizer in pretrained_Ngram_countvectorizer_list:
             Ngram_edge_feature_names += pretrained_Ngram_countvectorizer.get_feature_names_out().tolist()
         feature_names = nodetype_names + FRNP_AvgNum_DiffThreads +\
                         Ngram_edge_feature_names # yes this order is correct
   


    # JY @ 2024-2-10: Integrate all thread-level N-gram variants (i.e., thread-level N-1gram with all additional features)
    elif graph_embedding_option == "thread_level__N>1_grams_events__nodetype5bit__FRNPeventCount__FRNP_OutgoIncom_eventCount__AvgNum_DiffThreads_perFRNP":


         pretrained_Ngram_countvectorizer_list = pretrain__countvectorizer_on_training_set__before_graph_embedding_generation( Ngram = Ngram,
                                                                                                                               dataset= train_dataset,
                                                                                                                               only_train_specified_Ngram = only_train_specified_Ngram 
                                                                                                                              )


         train_dataset__signal_amplified_dict = get_1_grams_events__nodetype5bit__FRNPeventCount__FRNP_OutgoIncom_eventCount__AvgNum_DiffThreads_perFRNP__dict( 
                                                                                       pretrained_Ngram_countvectorizer_list = pretrained_Ngram_countvectorizer_list,
                                                                                       dataset= train_dataset,
                                                                                       pool = pool_option
                                                                                    )

         nodetype_names = ["file", "registry", "network", "process", "thread"] 
         FRNP_event_count_features = ["file_event_cnt", "registry_event_cnt", "network_event_cnt", "process_event_cnt" ]
         FRNP_outgoing_and_incoming_events_count_features = \
                                    ["file_outgoing_event_cnt", "registry_outgoing_event_cnt", "network_outgoing_event_cnt", "process_outgoing_event_cnt",
                                     "file_incoming_event_cnt", "registry_incoming_event_cnt", "network_incoming_event_cnt", "process_incoming_event_cnt"]   
         FRNP_AvgNum_DiffThreads = ["file_avg_diff_threads", "registry_avg_diff_threads", "network_avg_diff_threads", "process_avg_diff_threads" ]


         Ngram_edge_feature_names = []
         for pretrained_Ngram_countvectorizer in pretrained_Ngram_countvectorizer_list:
             Ngram_edge_feature_names += pretrained_Ngram_countvectorizer.get_feature_names_out().tolist()
         # JY @ 2024-2-10
         feature_names = nodetype_names + FRNP_event_count_features + FRNP_outgoing_and_incoming_events_count_features + FRNP_AvgNum_DiffThreads +\
                         Ngram_edge_feature_names # yes this order is correct



    else:
         ValueError(f"Invalid graph_embedding_option ({graph_embedding_option})")                  


    X = pd.DataFrame(train_dataset__signal_amplified_dict).T
    X.columns = feature_names
    X.reset_index(inplace = True)
    X.rename(columns = {'index':'data_name'}, inplace = True)
    X.to_csv(os.path.join(this_results_dirpath,"X.csv"))



    # =================================================================================================================================================

    if search_on_train__or__final_test == "final_test":
        # Also prepare for final-test dataset, to later test the best-fitted models on test-set
         # Now apply signal-amplification here (here least conflicts with existing code.)
         
         if graph_embedding_option == "thread_level__N>1_grams_events__nodetype5bit":
               final_test_dataset__signal_amplified_dict = get_thread_level_N_gram_events__nodetype5bit__dict( pretrained_Ngram_countvectorizer_list = pretrained_Ngram_countvectorizer_list,
                                                                                                               dataset= final_test_dataset,
                                                                                                               pool = pool_option
                                                                                                             )

               nodetype_names = ["file", "registry", "network", "process", "thread"] 
               Ngram_edge_feature_names = []
               for pretrained_Ngram_countvectorizer in pretrained_Ngram_countvectorizer_list:
                  Ngram_edge_feature_names += pretrained_Ngram_countvectorizer.get_feature_names_out().tolist()
               feature_names = nodetype_names + Ngram_edge_feature_names # yes this order is correct               
         

         elif graph_embedding_option == "thread_level__N>1_grams_events__nodetype5bit__FRNPeventCount":
         
               final_test_dataset__signal_amplified_dict = get_thread_level_N_gram_events__nodetype5bit__FRNPeventCount__dict( 
                                                                                                               pretrained_Ngram_countvectorizer_list = pretrained_Ngram_countvectorizer_list,
                                                                                                               dataset= final_test_dataset,
                                                                                                               pool = pool_option
                                                                                                             )

               nodetype_names = ["file", "registry", "network", "process", "thread"]
               FRNP_event_count_features = ["file_event_cnt", "registry_event_cnt", "network_event_cnt", "process_event_cnt" ] 
               Ngram_edge_feature_names = []
               for pretrained_Ngram_countvectorizer in pretrained_Ngram_countvectorizer_list:
                   Ngram_edge_feature_names += pretrained_Ngram_countvectorizer.get_feature_names_out().tolist()
               feature_names = nodetype_names + FRNP_event_count_features + \
                               Ngram_edge_feature_names # yes this order is correct       


         elif graph_embedding_option == "thread_level__N>1_grams_events__nodetype5bit__FRNPeventCount__FRNP_OutgoIncom_eventCount":

               final_test_dataset__signal_amplified_dict = get_thread_level_N_gram_events__nodetype5bit__FRNPeventCount__FRNP_OutgoIncom_eventCount__dict( 
                                                                                                               pretrained_Ngram_countvectorizer_list = pretrained_Ngram_countvectorizer_list,
                                                                                                               dataset= final_test_dataset,
                                                                                                               pool = pool_option
                                                                                                             )

               nodetype_names = ["file", "registry", "network", "process", "thread"]
               FRNP_event_count_features = ["file_event_cnt", "registry_event_cnt", "network_event_cnt", "process_event_cnt" ]
               FRNP_outgoing_and_incoming_events_count_features = \
                                          ["file_outgoing_event_cnt", "registry_outgoing_event_cnt", "network_outgoing_event_cnt", "process_outgoing_event_cnt",
                                           "file_incoming_event_cnt", "registry_incoming_event_cnt", "network_incoming_event_cnt", "process_incoming_event_cnt"]   

               Ngram_edge_feature_names = []
               for pretrained_Ngram_countvectorizer in pretrained_Ngram_countvectorizer_list:
                  Ngram_edge_feature_names += pretrained_Ngram_countvectorizer.get_feature_names_out().tolist()
               # JY @ 2024-1-27
               feature_names = nodetype_names + FRNP_event_count_features + FRNP_outgoing_and_incoming_events_count_features +\
                               Ngram_edge_feature_names # yes this order is correct


         elif graph_embedding_option == "thread_level__N>1_grams_events__nodetype5bit__AvgNum_DiffThreads_perFRNP":
               final_test_dataset__signal_amplified_dict = get_thread_level_N_gram_events__nodetype5bit__AvgNum_DiffThreads_perFRNP__dict( 
                                                                                                               pretrained_Ngram_countvectorizer_list = pretrained_Ngram_countvectorizer_list,
                                                                                                               dataset= final_test_dataset,
                                                                                                               pool = pool_option
                                                                                                             )
               nodetype_names = ["file", "registry", "network", "process", "thread"] 
               FRNP_AvgNum_DiffThreads = ["file_avg_diff_threads", "registry_avg_diff_threads", "network_avg_diff_threads", "process_avg_diff_threads" ]
      
               Ngram_edge_feature_names = []
               for pretrained_Ngram_countvectorizer in pretrained_Ngram_countvectorizer_list:
                  Ngram_edge_feature_names += pretrained_Ngram_countvectorizer.get_feature_names_out().tolist()
               feature_names = nodetype_names + FRNP_AvgNum_DiffThreads +\
                              Ngram_edge_feature_names # yes this order is correct


         # JY @ 2024-2-10: Integrate all thread-level N-gram variants (i.e., thread-level N-1gram with all additional features)
         elif graph_embedding_option == "thread_level__N>1_grams_events__nodetype5bit__FRNPeventCount__FRNP_OutgoIncom_eventCount__AvgNum_DiffThreads_perFRNP":


               final_test_dataset__signal_amplified_dict = get_1_grams_events__nodetype5bit__FRNPeventCount__FRNP_OutgoIncom_eventCount__AvgNum_DiffThreads_perFRNP__dict( 
                                                                                             pretrained_Ngram_countvectorizer_list = pretrained_Ngram_countvectorizer_list,
                                                                                             dataset= final_test_dataset,
                                                                                             pool = pool_option
                                                                                          )

               nodetype_names = ["file", "registry", "network", "process", "thread"] 
               FRNP_event_count_features = ["file_event_cnt", "registry_event_cnt", "network_event_cnt", "process_event_cnt" ]
               FRNP_outgoing_and_incoming_events_count_features = \
                                          ["file_outgoing_event_cnt", "registry_outgoing_event_cnt", "network_outgoing_event_cnt", "process_outgoing_event_cnt",
                                          "file_incoming_event_cnt", "registry_incoming_event_cnt", "network_incoming_event_cnt", "process_incoming_event_cnt"]   
               FRNP_AvgNum_DiffThreads = ["file_avg_diff_threads", "registry_avg_diff_threads", "network_avg_diff_threads", "process_avg_diff_threads" ]


               Ngram_edge_feature_names = []
               for pretrained_Ngram_countvectorizer in pretrained_Ngram_countvectorizer_list:
                  Ngram_edge_feature_names += pretrained_Ngram_countvectorizer.get_feature_names_out().tolist()
               # JY @ 2024-2-10
               feature_names = nodetype_names + FRNP_event_count_features + FRNP_outgoing_and_incoming_events_count_features + FRNP_AvgNum_DiffThreads +\
                               Ngram_edge_feature_names # yes this order is correct



         else:
               ValueError(f"Invalid graph_embedding_option ({graph_embedding_option})")



         final_test_X = pd.DataFrame(final_test_dataset__signal_amplified_dict).T               
         final_test_X.columns = feature_names
         final_test_X.reset_index(inplace = True)
         final_test_X.rename(columns = {'index':'data_name'}, inplace = True)

         final_test_X.to_csv(os.path.join(this_results_dirpath,"final_test_X.csv"))


         #--------------------------------------------------------------------------
     

         
    #--------------------------------------------------------------------------------------------------------------------------------------------------------
    #--------------------------------------------------------------------------------------------------------------------------------------------------------
    #--------------------------------------------------------------------------------------------------------------------------------------------------------
    #--------------------------------------------------------------------------------------------------------------------------------------------------------
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


            elif model_cls_name == 'sklearn.ensemble._forest.RandomForestClassifier' or \
               'randomforest' in search_space_option.lower() or 'rf' in search_space_option.lower():
               model = model_cls(
                                 n_estimators= hyperparam_set['n_estimators'],
                                 criterion= hyperparam_set['criterion'], 
                                 max_depth= hyperparam_set['max_depth'],
                                 min_samples_split= hyperparam_set['min_samples_split'], 
                                 min_samples_leaf= hyperparam_set['min_samples_leaf'], 
                                 max_features= hyperparam_set['max_features'],
                                 bootstrap= hyperparam_set['bootstrap'],
                                 random_state= hyperparam_set['random_state'],

                                 n_jobs = RF__n_jobs
                                 )
                                 # Added by JY @ 2024-1-23:
                                 #     "n_jobs" == This parameter is used to specify how many concurrent processes or threads should be used for routines that are parallelized with joblib.
                                 #     The number of jobs to run in parallel. fit, predict, decision_path and apply are all parallelized over the trees.
                                 #     -1 means using all processors
                                 #     https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier.fit
                                 #     https://scikit-learn.org/stable/glossary.html#term-n_jobs


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

            data_names = X['data_name']

            for data_name in data_names:
                  # benign --------------
                  if "powershell-master" in data_name:
                     X_grouplist.append("benign_fleschutz")
                  elif "jhochwald" in data_name:
                     X_grouplist.append("benign_jhochwald")
                  elif "devblackops" in data_name:
                     X_grouplist.append("benign_devblackops")
                  elif "farag2" in data_name:
                     X_grouplist.append("benign_farag2")
                  elif "jimbrig" in data_name:
                     X_grouplist.append("benign_jimbrig")
                  elif "jrussellfreelance" in data_name:
                     X_grouplist.append("benign_jrussellfreelance")
                  elif "nickrod518" in data_name:
                     X_grouplist.append("benign_nickrod518")
                  elif "redttr" in data_name:
                     X_grouplist.append("benign_redttr")
                  elif "sysadmin-survival-kit" in data_name:
                     X_grouplist.append("benign_sysadmin-survival-kit")
                  elif "stevencohn" in data_name:
                     X_grouplist.append("benign_stevencohn")
                  elif "ledragox" in data_name:
                     X_grouplist.append("benign_ledrago")
                  elif "floriantim" in data_name: # Added by JY @ 2024-2-3 : pattern in dataset-2
                     X_grouplist.append("benign_floriantim")
                  elif "nickbeau" in data_name: # Added by JY @ 2024-2-3 : pattern in dataset-2
                     X_grouplist.append("benign_nickbeau")
                  # malware ------------------------------------------
                  elif "empire" in data_name:
                     X_grouplist.append("malware_empire")
                  elif "invoke_obfuscation" in data_name:
                     X_grouplist.append("malware_invoke_obfuscation")
                  elif "nishang" in data_name:
                     X_grouplist.append("malware_nishang")
                  elif "poshc2" in data_name:
                     X_grouplist.append("malware_poshc2")
                  elif "mafia" in data_name:
                     X_grouplist.append("malware_mafia")
                  elif "offsec" in data_name:
                     X_grouplist.append("malware_offsec")
                  elif "powershellery" in data_name:
                     X_grouplist.append("malware_powershellery")
                  elif "psbits" in data_name:
                     X_grouplist.append("malware_psbits")
                  elif "pt_toolkit" in data_name:
                     X_grouplist.append("malware_pt_toolkit")
                  elif "randomps" in data_name:
                     X_grouplist.append("malware_randomps")
                  elif "smallposh" in data_name:
                     X_grouplist.append("malware_smallposh")
                  elif "asyncrat" in data_name: #PW: need to change depends on all types e.g., malware_rest
                     X_grouplist.append("malware_asyncrat")
                  elif "bumblebee" in data_name:
                     X_grouplist.append("malware_bumblebee")
                  elif "cobalt_strike" in data_name:
                     X_grouplist.append("malware_cobalt_strike")
                  elif "coinminer" in data_name:
                     X_grouplist.append("malware_coinminer")
                  elif "gozi" in data_name:
                     X_grouplist.append("malware_gozi")
                  elif "guloader" in data_name:
                     X_grouplist.append("malware_guloader")
                  elif "netsupport" in data_name:
                     X_grouplist.append("malware_netsupport")
                  elif "netwalker" in data_name:
                     X_grouplist.append("malware_netwalker")
                  elif "nw0rm" in data_name:
                     X_grouplist.append("malware_nw0rm")
                  elif "quakbot" in data_name:
                     X_grouplist.append("malware_quakbot")
                  elif "quasarrat" in data_name:
                     X_grouplist.append("malware_quasarrat")
                  elif "rest" in data_name:
                     X_grouplist.append("malware_rest")
                  elif "metasploit" in data_name:
                     X_grouplist.append("malware_metasploit")                       
                  # if "recollected" in data_name:
                     # X_grouplist.append("malware_recollected")
                  else:
                     raise ValueError(f"unidentifeid pattern in {data_name}")


            # correctness of X_grouplist can be checked by following
            # list(zip(X, [data_name for data_name in X.index], y, X_grouplist))

            dataset_cv = StratifiedKFold(n_splits=K, 
                                          shuffle = True, 
                                          random_state=np.random.RandomState(seed=split_shuffle_seed))

            # Model-id Prefix

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
                  print(f"Benign: #train:{y_train.value_counts()[0]}, #validation:{y_validation.value_counts()[0]},\nMalware: #train:{y_train.value_counts()[1]}, #validation:{y_validation.value_counts()[1]}",
                        flush=True)
                  

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
               'xgboost' in search_space_option.lower() or 'xgb' in search_space_option.lower() :

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
               'randomforest' in search_space_option.lower() or 'rf' in search_space_option.lower() :
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


            else:
               ValueError(f"{model_cls_name} is not supported", flush = True)

            def convert_to_label(value):
               # if "benign" in value.lower(): return 0
               if "malware" in value.lower(): return 1
               else:
                   return 0
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