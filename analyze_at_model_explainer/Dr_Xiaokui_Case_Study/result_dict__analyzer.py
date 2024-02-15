
'''
JY @ TODO FOR 2023-2-14 

Could perform some intersection operations on 'wrong-direction pushing features'
to identify frequently appearing wrong-direction pushing features and 
** try to interpret with domian knolwedge **
   
'''

# Specific - TODOs
# 1-1. get wrong-direction-pushing-features that appear often in mispredicted malware-samples , by checking intersection-frequency 
# 1-2. get wrong-direction-pushing-features that appear often in mispredicted benign-samples , by checking intersection-frequency    
# 1-3. get wrong-direction-pushing-features that appear often in mispredicted all-samples , by checking intersection-frequency    
# 1-4. check how those wrong-direction-pushing-features did in correctly-predicted samples (did they push towards correct direction or wrong direction on average?)

# 2-1. get correct-direction-pushing-feautres that appear often in correctly-predicted malware-samples , by checking intersection-frequency  
# 2-2. get correct-direction-pushing-feautres that appear often in correctly-predicted benign-samples , by checking intersection-frequency  
# 2-3. get correct-direction-pushing-feautres that appear often in correctly-predicted all-samples , by checking intersection-frequency  
# 2-4. check how those correct-direction-pushing-features did in mispredicted samples (did they push towards correct direction or wrong direction on average?)



      # global correct direction pushing features 
      # global wrong direction pushing features
      # or do it by caegory like mispdicted benign or malware samples or etc
      # 
      # thread level ngram이라는거 remind 
      # overlapping timestamp by thread도 보여주기
      # global Ngram featuee과 thread level ngram feature비교위해

import os
import json
from collections import defaultdict
import pandas as pd
import numpy as np

if __name__ == "__main__":


   Mispredictions_Wrong_Direction_Pushing_Feats_fpath =\
   "/home/jgwak1/tabby/graph_embedding_improvement_JY_git/analyze_at_model_explainer/Dr_Xiaokui_Case_Study/RESULTS/Mispredictions_Wrong_Direction_Pushing_Feats___Best_RF__Full_Dataset_2_Double_Stratified__4gram__sum_pool__final_test__thread_level__N>1_grams_events__nodetype5bit__4gram__sum_pool__2024-02-13_121930.json" 
   
   CorrectPredictions_Correct_Direction_Pushing_Feats_fpath =\
   "/home/jgwak1/tabby/graph_embedding_improvement_JY_git/analyze_at_model_explainer/Dr_Xiaokui_Case_Study/RESULTS/Correct_Predictions_Correct_Direction_Pushing_Feats___Best_RF__Full_Dataset_2_Double_Stratified__4gram__sum_pool__final_test__thread_level__N>1_grams_events__nodetype5bit__4gram__sum_pool__2024-02-13_121930.json" 


   results_save_dirpath = "/home/jgwak1/tabby/graph_embedding_improvement_JY_git/analyze_at_model_explainer/Dr_Xiaokui_Case_Study/RESULTS"

 
   with open(Mispredictions_Wrong_Direction_Pushing_Feats_fpath, 'r') as file:
         Mispredictions_Wrong_Direction_Pushing_Feats__dict = json.load(file)

   with open(CorrectPredictions_Correct_Direction_Pushing_Feats_fpath, 'r') as file:
         CorrectPredictions_Correct_Direction_Pushing_Feats__dict = json.load(file)


   


   # That appear in all mispredicted? (at least half?)
   # ''' Among all mispredicted samples, top N features that pushed towards wrong direction ''' 

   # feature_to_all_mispred_sample_localshap = defaultdict(list)
   # for mispred_sample, nested_dict in Mispredictions_Wrong_Direction_Pushing_Feats__dict.items():
   #      for k, feature_and_localshap in nested_dict.items():
   #          for feature, localshap in feature_and_localshap.items():
   #             feature_to_all_mispred_sample_localshap[feature].append( (mispred_sample, localshap))


   # ---------------------------------------------------------------------------------------------------------------------------------------------------------
   ''' Among mispredicted malware samples, top N features that pushed towards wrong direction ''' 

   feature_to_mispred_malware_sample_localshap = defaultdict(list)

   mispred_malware_sample_cnt = 0

   for mispred_sample, nested_dict in Mispredictions_Wrong_Direction_Pushing_Feats__dict.items():
      
      if mispred_sample.startswith("malware_"):
         mispred_malware_sample_cnt+=1

         for k, feature_and_localshap in nested_dict.items():
               for feature, localshap in feature_and_localshap.items():
                  feature_to_mispred_malware_sample_localshap[feature].append( (mispred_sample, localshap))

   mispred_malware_samples__feature_avg_localshap = dict()
   for feature, mispred_sample_local_shap_tup_list in feature_to_mispred_malware_sample_localshap.items():
      if len(mispred_sample_local_shap_tup_list) > 0: # handle how many
         avg_local_shap = np.mean([x[1] for x in mispred_sample_local_shap_tup_list])
         mispred_malware_samples__feature_avg_localshap[feature] = (avg_local_shap, f"{len(mispred_sample_local_shap_tup_list)}/{mispred_malware_sample_cnt}" )
   # Sort the dictionary by values in ascending order
   sorted_mispred_malware_samples__feature_avg_localshap = dict(sorted(mispred_malware_samples__feature_avg_localshap.items(), key=lambda item: item[1][0]))


   identifier = os.path.split(Mispredictions_Wrong_Direction_Pushing_Feats_fpath)[1].removesuffix(".json")
   identifier = identifier[identifier.find("Best_RF"):]
   with open(os.path.join(results_save_dirpath, f"Among_All_Mispred_Malware_Samples_TopNfeatures_Pushing_To_Wrong_Direction___{identifier}.json"), "w") as json_file:
         json.dump(sorted_mispred_malware_samples__feature_avg_localshap, json_file) 



   # ---------------------------------------------------------------------------------------------------------------------------------------------------------
   ''' Among mispredicted benign samples, top N features that pushed towards wrong direction ''' 

   feature_to_mispred_benign_sample_localshap = defaultdict(list)

   mispred_benign_sample_cnt = 0

   for mispred_sample, nested_dict in Mispredictions_Wrong_Direction_Pushing_Feats__dict.items():
      
      if mispred_sample.startswith("benign_"):
         mispred_benign_sample_cnt += 1

         for k, feature_and_localshap in nested_dict.items():
               for feature, localshap in feature_and_localshap.items():
                  feature_to_mispred_benign_sample_localshap[feature].append( (mispred_sample, localshap))

   mispred_benign_samples__feature_avg_localshap = dict()
   for feature, mispred_sample_local_shap_tup_list in feature_to_mispred_benign_sample_localshap.items():
      if len(mispred_sample_local_shap_tup_list) > 0: # handle how many
         avg_local_shap = np.mean([x[1] for x in mispred_sample_local_shap_tup_list])
         mispred_benign_samples__feature_avg_localshap[feature] = (avg_local_shap, f"{len(mispred_sample_local_shap_tup_list)}/{mispred_benign_sample_cnt}" )
   # Sort the dictionary by values in descending order
   sorted_mispred_benign_samples__feature_avg_localshap = dict(sorted(mispred_benign_samples__feature_avg_localshap.items(), key=lambda item: item[1][0], reverse= True))

   identifier = os.path.split(Mispredictions_Wrong_Direction_Pushing_Feats_fpath)[1].removesuffix(".json")
   identifier = identifier[identifier.find("Best_RF"):]
   with open(os.path.join(results_save_dirpath, f"Among_All_Mispredicted_Benign_Samples_TopNfeatures_Pushing_To_Wrong_Direction___{identifier}.json"), "w") as json_file:
         json.dump(sorted_mispred_benign_samples__feature_avg_localshap, json_file) 

   





   # ---------------------------------------------------------------------------------------------------------------------------------------------------------
   ''' Among correctly-predicted malware samples, top N features that pushed towards wrong direction ''' 
   feature_to_correctpred_malware_sample_localshap = defaultdict(list)

   correctpred_malware_sample_cnt = 0

   for correctpred_sample, nested_dict in CorrectPredictions_Correct_Direction_Pushing_Feats__dict.items():
      
      if correctpred_sample.startswith("malware_"):
         correctpred_malware_sample_cnt += 1

         for k, feature_and_localshap in nested_dict.items():
               for feature, localshap in feature_and_localshap.items():
                  feature_to_correctpred_malware_sample_localshap[feature].append( (correctpred_sample, localshap))

   correctpred_malware_samples__feature_avg_localshap = dict()
   for feature, correctpred_sample_local_shap_tup_list in feature_to_correctpred_malware_sample_localshap.items():
      if len(correctpred_sample_local_shap_tup_list) > 0: # handle how many
         avg_local_shap = np.mean([x[1] for x in correctpred_sample_local_shap_tup_list])
         correctpred_malware_samples__feature_avg_localshap[feature] = (avg_local_shap, f"{len(correctpred_sample_local_shap_tup_list)}/{correctpred_malware_sample_cnt}" )
   # Sort the dictionary by values in descedning order
   sorted_correctpred_malware_samples__feature_avg_localshap = dict(sorted(correctpred_malware_samples__feature_avg_localshap.items(), key=lambda item: item[1][0], reverse= True))


   identifier = os.path.split(CorrectPredictions_Correct_Direction_Pushing_Feats_fpath)[1].removesuffix(".json")
   identifier = identifier[identifier.find("Best_RF"):]
   with open(os.path.join(results_save_dirpath, f"Among_All_CorrectPredicted_Malware_Samples_TopNfeatures_Pushing_To_Correct_Direction___{identifier}.json"), "w") as json_file:
         json.dump(sorted_correctpred_malware_samples__feature_avg_localshap, json_file) 


   # ---------------------------------------------------------------------------------------------------------------------------------------------------------
   ''' Among correctly-predicted benign samples, top N features that pushed towards correct direction ''' 

   feature_to_correctpred_benign_sample_localshap = defaultdict(list)

   correctpred_benign_sample_cnt = 0

   for correctpred_sample, nested_dict in CorrectPredictions_Correct_Direction_Pushing_Feats__dict.items():
      
      if correctpred_sample.startswith("benign_"):
         correctpred_benign_sample_cnt += 1

         for k, feature_and_localshap in nested_dict.items():
               for feature, localshap in feature_and_localshap.items():
                  feature_to_correctpred_benign_sample_localshap[feature].append( (correctpred_sample, localshap))

   correctpred_benign_samples__feature_avg_localshap = dict()
   for feature, correctpred_sample_local_shap_tup_list in feature_to_correctpred_benign_sample_localshap.items():
      if len(correctpred_sample_local_shap_tup_list) > 0: # handle how many
         avg_local_shap = np.mean([x[1] for x in correctpred_sample_local_shap_tup_list])
         correctpred_benign_samples__feature_avg_localshap[feature] = (avg_local_shap, f"{len(correctpred_sample_local_shap_tup_list)}/{correctpred_benign_sample_cnt}" )
   # Sort the dictionary by values in ascending order
   sorted_correctpred_benign_samples__feature_avg_localshap = dict(sorted(correctpred_benign_samples__feature_avg_localshap.items(), key=lambda item: item[1][0]))

   identifier = os.path.split(CorrectPredictions_Correct_Direction_Pushing_Feats_fpath)[1].removesuffix(".json")
   identifier = identifier[identifier.find("Best_RF"):]
   with open(os.path.join(results_save_dirpath, f"Among_All_CorrectPredicted_Benign_Samples_TopNfeatures_Pushing_To_Correct_Direction___{identifier}.json"), "w") as json_file:
         json.dump(sorted_correctpred_benign_samples__feature_avg_localshap, json_file) 



   # x = dict()
   # for k,v in Mispredictions_Wrong_Direction_Pushing_Feats__dict.items():
   #    for k2,v2 in v.items():
   #       x[k] = set(v2.keys())

   # # top x 

   # feats_set = set()
   # for k,v in x.items():
   #    feats_set.update(v)

   # feat_to_sample = defaultdict(set)
   # for feat in feats_set:
   #    for k,v in x.items():
   #       if feat in v:
   #          feat_to_sample[feat].add(k)


