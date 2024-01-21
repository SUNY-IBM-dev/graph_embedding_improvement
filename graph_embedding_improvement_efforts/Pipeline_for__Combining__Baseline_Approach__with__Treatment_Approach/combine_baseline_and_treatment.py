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

sys.path.append("/data/d1/jgwak1/tabby/graph_embedding_improvement_JY_git/graph_embedding_improvement_efforts/Pipeline_for__Combining__Baseline_Approach__with__Treatment_Approach/source")

from source.dataprocessor_graphs import LoadGraphs
from source.model import GIN
from source.trainer_meng_ver import TrainModel

from itertools import product
import pandas as pd
from datetime import datetime

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold



sys.path.append("/data/d1/jgwak1/tabby/graph_embedding_improvement_JY_git/graph_embedding_improvement_efforts/Pipeline_for__Combining__Baseline_Approach__with__Treatment_Approach")

from _baseline_graph_embedding_functions import *
from _baseline_graph_embedding_functions import *
from _explanation_functions import *



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






##############################################################################################################################

#**********************************************************************************************************************************************************************






##########################################################################################################################################################
# Signal-Amplification Function (Thread-level Event-Dist. 1gram + Adjacent Node's Node-Type 5Bit)
#PW: Thread node embedding by aggregating one hop neighbour nodes

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
                                  'Dataset-Case-2' # try
                                  ], 
                        default = ['Dataset-Case-2'])


    model_cls_map = {"RandomForest": RandomForestClassifier, "XGBoost": GradientBoostingClassifier,
                     "LogisticRegression": LogisticRegression, "SVM": svm } 
    
    parser.add_argument('-mod_cls', '--model_choice', nargs = 1, type = str, 
                        default = ["RandomForest"] )

    parser.add_argument('-ss_opt', '--search_space_option', 
                        choices= [
                                  'XGBoost_searchspace_1',
                                  'RandomForest_searchspace_1',
                                  ], 
                                  default = ["RandomForest_searchspace_1"])

#PW: Why 10 Kfold? just common values
 # flatten vs no graph ?? is that only ML tuning differece??
   
    parser.add_argument('-graphemb_opt', '--graph_embedding_option', 
                        choices= [
                                 #  'thread_level__N_grams_events',

                                  'thread_level__N>1_grams_events__nodetype_5bit', 


                                 #  'thread_level__N>1_grams_events__nodetype_5bit_and_Adhoc_Identifier',

                                  ], 
                                  default = ["thread_level__N>1_grams_events__nodetype_5bit"])
    
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


    # -----------------------------------------------------------------------------------------------------------------------------------
    model_cls_name = re.search(r"'(.*?)'", str(model_cls)).group(1)


    if search_on_train__or__final_test in {"search_on_train", "search_on_all"}:

      #  if "thread_level__N>1_grams_events__nodetype_5bit" in graph_embedding_option:
      #    run_identifier = f"{model_choice}__{dataset_choice}__{search_space_option}__{K}_FoldCV__{search_on_train__or__final_test}__{graph_embedding_option}__{pool_option}_pool__{datetime.now().strftime('%Y-%m-%d_%H%M%S')}"
      #  else:
      #    run_identifier = f"{model_choice}__{dataset_choice}__{search_space_option}__{K}_FoldCV__{search_on_train__or__final_test}__{graph_embedding_option}__{datetime.now().strftime('%Y-%m-%d_%H%M%S')}"  

       run_identifier = f"{model_choice}__{dataset_choice}__{search_space_option}__{K}_FoldCV__{search_on_train__or__final_test}__{graph_embedding_option}__{Ngram}gram__{pool_option}_pool__only_train_specified_Ngram_{only_train_specified_Ngram}__{datetime.now().strftime('%Y-%m-%d_%H%M%S')}"
       this_results_dirpath = f"/data/d1/jgwak1/tabby/graph_embedding_improvement_JY_git/graph_embedding_improvement_efforts/Trial_7__Thread_level_N_grams__N_gt_than_1__Similar_to_PriorGraphEmbedding/RESULTS/{run_identifier}"
       experiment_results_df_fpath = os.path.join(this_results_dirpath, f"{run_identifier}.csv")
       if not os.path.exists(this_results_dirpath):
           os.makedirs(this_results_dirpath)


    if search_on_train__or__final_test == "final_test":
      #  if "thread_level__N>1_grams_events__nodetype_5bit" in graph_embedding_option:
      #  run_identifier = f"{model_choice}__{dataset_choice}__{search_space_option}__{search_on_train__or__final_test}__{graph_embedding_option}__{pool_option}_pool__{datetime.now().strftime('%Y-%m-%d_%H%M%S')}"
      #  else:
      #    run_identifier = f"{model_choice}__{dataset_choice}__{search_space_option}__{search_on_train__or__final_test}__{graph_embedding_option}__{datetime.now().strftime('%Y-%m-%d_%H%M%S')}"  

       run_identifier = f"{model_choice}__{dataset_choice}__{search_space_option}__{search_on_train__or__final_test}__{graph_embedding_option}__{Ngram}gram__{pool_option}_pool__only_train_specified_Ngram_{only_train_specified_Ngram}__{datetime.now().strftime('%Y-%m-%d_%H%M%S')}"
       this_results_dirpath = f"/data/d1/jgwak1/tabby/graph_embedding_improvement_JY_git/graph_embedding_improvement_efforts/Trial_7__Thread_level_N_grams__N_gt_than_1__Similar_to_PriorGraphEmbedding/RESULTS/{run_identifier}"
       final_test_results_df_fpath = os.path.join(this_results_dirpath, f"{run_identifier}.csv")
       if not os.path.exists(this_results_dirpath):
           os.makedirs(this_results_dirpath)

    trace_filename = f'traces_stratkfold_double_strat_{model_cls_name}__generated@'+str(datetime.now())+".txt" 


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
    if graph_embedding_option == "thread_level__N>1_grams_events__nodetype_5bit":
         
         pretrained_Ngram_countvectorizer_list = pretrain__countvectorizer_on_training_set__before_graph_embedding_generation( Ngram = Ngram,
                                                                                                                               dataset= train_dataset,
                                                                                                                               only_train_specified_Ngram = only_train_specified_Ngram 
                                                                                                                            )


         train_dataset__signal_amplified_dict = get_thread_level_N_gram_events_adjacent_5bit_dist_dict( pretrained_Ngram_countvectorizer_list = pretrained_Ngram_countvectorizer_list,
                                                                                                        dataset= train_dataset,
                                                                                                        pool = pool_option,
                                                                                                          )


         nodetype_names = ["file", "registry", "network", "process", "thread"] 
         Ngram_edge_feature_names = []
         for pretrained_Ngram_countvectorizer in pretrained_Ngram_countvectorizer_list:
             Ngram_edge_feature_names += pretrained_Ngram_countvectorizer.get_feature_names_out().tolist()

         feature_names = nodetype_names + Ngram_edge_feature_names # yes this order is correct
         X = pd.DataFrame(train_dataset__signal_amplified_dict).T


    else:
         ValueError(f"Invalid graph_embedding_option ({graph_embedding_option})")                  


    X.columns = feature_names
    X.reset_index(inplace = True)
    X.rename(columns = {'index':'data_name'}, inplace = True)
    X.to_csv(os.path.join(this_results_dirpath,"X.csv"))



    # =================================================================================================================================================

    if search_on_train__or__final_test == "final_test":
        # Also prepare for final-test dataset, to later test the best-fitted models on test-set
         # Now apply signal-amplification here (here least conflicts with existing code.)
         if graph_embedding_option == "thread_level__N>1_grams_events__nodetype_5bit":
               final_test_dataset__signal_amplified_dict = get_thread_level_N_gram_events_adjacent_5bit_dist_dict( pretrained_Ngram_countvectorizer_list = pretrained_Ngram_countvectorizer_list,
                                                                                                                   dataset= final_test_dataset,
                                                                                                                   pool = pool_option,
                                                                                                                  )

               

               nodetype_names = ["file", "registry", "network", "process", "thread"] 
               Ngram_edge_feature_names = []
               for pretrained_Ngram_countvectorizer in pretrained_Ngram_countvectorizer_list:
                  Ngram_edge_feature_names += pretrained_Ngram_countvectorizer.get_feature_names_out().tolist()
               feature_names = nodetype_names + Ngram_edge_feature_names # yes this order is correct               
         
               
               final_test_X = pd.DataFrame(final_test_dataset__signal_amplified_dict).T               


         else:
               ValueError(f"Invalid graph_embedding_option ({graph_embedding_option})")

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
                  # benign -------------- # PW:for silketw we dont have benign source level identifier
                  # if "fleschutz" in data_name:
                     # X_grouplist.append("benign_fleschutz")
                  # if "jhochwald" in data_name:
                     # X_grouplist.append("benign_jhochwald")
                  # if "devblackops" in data_name:
                     # X_grouplist.append("benign_devblackops")
                  # if "farag2" in data_name:
                     # X_grouplist.append("benign_farag2")
                  # if "jimbrig" in data_name:
                     # X_grouplist.append("benign_jimbrig")
                  # if "jrussellfreelance" in data_name:
                     # X_grouplist.append("benign_jrussellfreelance")
                  # if "nickrod518" in data_name:
                     # X_grouplist.append("benign_nickrod518")
                  # if "redttr" in data_name:
                     # X_grouplist.append("benign_redttr")
                  # if "sysadmin-survival-kit" in data_name:
                     # X_grouplist.append("benign_sysadmin-survival-kit")
                  # if "stevencohn" in data_name:
                     # X_grouplist.append("benign_stevencohn")
                  # if "ledrago" in data_name:
                     # X_grouplist.append("benign_ledrago")
                  # malware ------------------------------------------
                  if "empire" in data_name:
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
                      X_grouplist.append("benign")

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