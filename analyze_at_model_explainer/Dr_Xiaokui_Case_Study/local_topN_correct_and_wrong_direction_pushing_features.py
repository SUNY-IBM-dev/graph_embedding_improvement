
'''

TODO: Write a script that, for each prediction, returns Top N
features that pushed towards wrong prediction, and Top N features
that pushed towards correct direction , based on local SHAP explanations. 

– Apply the script to mispredictions, and try get statistics
on which features were most responsible for mispredictions (Dr. Xiaokui Case-Study)

'''


import os
import pandas as pd
import json

if __name__ == "__main__":

   # Set following

   final_test_model_dirpath = \
   "/home/jgwak1/tabby/graph_embedding_improvement_JY_git/graph_embedding_improvement_efforts/Trial_7__Thread_level_N_grams__N_gt_than_1__Similar_to_PriorGraphEmbedding/RESULTS/RandomForest__Full_Dataset_2_Double_Stratified__Best_RF__Full_Dataset_2_Double_Stratified__4gram__sum_pool__final_test__thread_level__N>1_grams_events__nodetype5bit__4gram__sum_pool__2024-02-13_121930"


   samples_of_interest = "Mispredictions" # None, 'Mispredictions' or 'Correct_Predictions' 
   wrong_direction_feats = True
   correct_direction_feats = False
   top_N = None # int or None

   # ======================================================================================================================================

   GlobalSHAP_Importance = pd.read_csv( os.path.join(final_test_model_dirpath, 
                                                      [f for f in os.listdir(final_test_model_dirpath) if f.endswith('Global-SHAP Importance.csv')][0]))

   GlobalSHAP_TestDataset = pd.read_csv( os.path.join(final_test_model_dirpath, 
                                                      [f for f in os.listdir(final_test_model_dirpath) if f.endswith('Global-SHAP Important FeatureNames Test-Dataset.csv')][0]))

   Local_SHAP_vals_TestDataset = pd.read_csv( os.path.join(final_test_model_dirpath, 
                                                      [f for f in os.listdir(final_test_model_dirpath) if f.endswith('Local-SHAP values Test-Dataset.csv')][0]))

   # -------
   Waterfallplots_dirpath = os.path.join( final_test_model_dirpath, 
                                          [f for f in os.listdir(final_test_model_dirpath) if f.startswith('WATERFALL_PLOTS_Local-Explanation_')][0] )
   Mispredictions_Waterfallplots_dirpath = os.path.join(Waterfallplots_dirpath, "Mispredictions")
   Correct_Predictions_Waterfallplots_dirpath = os.path.join(Waterfallplots_dirpath, "Correct_Predictions")
   start_idx = os.listdir(Mispredictions_Waterfallplots_dirpath)[0].index('waterfall_plot_') + len('waterfall_plot_')

   Mispredictions = [ x[start_idx:].removesuffix('.png') for x in os.listdir(Mispredictions_Waterfallplots_dirpath) ]
   Correct_Predictions = [ x[start_idx:].removesuffix('.png') for x in os.listdir(Correct_Predictions_Waterfallplots_dirpath) ]
   
   samples_of_interest__dict = {"Mispredictions": Mispredictions, "Correct_Predictions": Correct_Predictions}

   # -------------------------------------------------------------------------------------------------------------------------------------------------------
   samples_to_check = samples_of_interest__dict[samples_of_interest]



   Local_SHAP_vals_TestDataset.set_index("data_name", inplace = True)
   
   if samples_of_interest == None:
      samples_to_check = list(Local_SHAP_vals_TestDataset.index)
   

   result_dict = dict()

   for test_data in Local_SHAP_vals_TestDataset.index:



      if test_data.startswith("malware"):

         if test_data in samples_to_check:

            # if 'malware', then features that push to correct direction have 'positive' local-shap value
            
            descending_sorted_series = Local_SHAP_vals_TestDataset.loc[test_data].sort_values(ascending = False) # top-1 is highest-positive value
            ascending_sorted_series = Local_SHAP_vals_TestDataset.loc[test_data].sort_values(ascending = True) # top-1 is most-negative value

            correct_direction_features = descending_sorted_series[descending_sorted_series > 0]
            wrong_direction_features = ascending_sorted_series[ascending_sorted_series < 0]

            if top_N == None:
               correct_direction_top_N_features = dict(correct_direction_features)            
               wrong_direction_top_N_features = dict(wrong_direction_features)
            else:
               correct_direction_top_N_features = dict(correct_direction_features[0:top_N])            
               wrong_direction_top_N_features = dict(wrong_direction_features[0:top_N])

            direction_choice_dict = dict()
            if wrong_direction_feats == True:
               direction_choice_dict |= {f"wrong_direction_top_{top_N}_features" :  wrong_direction_top_N_features}
            if correct_direction_feats == True:
               direction_choice_dict |= {f"correct_direction_top_{top_N}_features" :  correct_direction_top_N_features}

            result_dict[test_data] = direction_choice_dict


      elif test_data.startswith("benign"):


         if test_data.startswith("benign_benign_"): # I have no idea why this happened somewhere
            test_data_realname = test_data.removeprefix("benign_")
         else:
            test_data_realname = test_data

         if test_data_realname in samples_to_check:

            # if 'benign', then features that push to correct direction have 'negative' local-shap value
            descending_sorted_series = Local_SHAP_vals_TestDataset.loc[test_data].sort_values(ascending = False) # top-1 is highest-positive value
            ascending_sorted_series = Local_SHAP_vals_TestDataset.loc[test_data].sort_values(ascending = True) # top-1 is most-negative value

            correct_direction_features = ascending_sorted_series[ascending_sorted_series < 0]            
            wrong_direction_features = descending_sorted_series[descending_sorted_series > 0]

            if top_N == None:
               correct_direction_top_N_features = dict(correct_direction_features)            
               wrong_direction_top_N_features = dict(wrong_direction_features)
            else:
               correct_direction_top_N_features = dict(correct_direction_features[0:top_N])            
               wrong_direction_top_N_features = dict(wrong_direction_features[0:top_N])

            
            direction_choice_dict = dict()
            if wrong_direction_feats == True:
               direction_choice_dict |= {f"wrong_direction_top_{top_N}_features" :  wrong_direction_top_N_features}
            if correct_direction_feats == True:
               direction_choice_dict |= {f"correct_direction_top_{top_N}_features" :  correct_direction_top_N_features}


            result_dict[test_data_realname] = direction_choice_dict



      else:
         raise ValueError(f"{test_data} should either start with 'malware' or 'benign'")


         #Local_SHAP_vals_TestDataset.loc[test_data]   
   # ==============================================================================================================================
   results_dirpath = "/home/jgwak1/tabby/graph_embedding_improvement_JY_git/analyze_at_model_explainer/Dr_Xiaokui_Case_Study/RESULTS"

   final_model_identifier = os.path.split(final_test_model_dirpath)[1]
   final_model_identifier = final_model_identifier[final_model_identifier.find("Best_RF"):] # to avoid too-long 


   if wrong_direction_feats == True and correct_direction_feats == False:
      Direction_Desc ="Wrong_Direction"
   elif wrong_direction_feats == False and correct_direction_feats == True:
      Direction_Desc ="Correct_Direction"
   elif wrong_direction_feats == True and correct_direction_feats == True:
      Direction_Desc ="Both"
   else:
      raise ValueError("Need to set either one direction (wrong or correct)")

   with open(os.path.join(results_dirpath, f"{samples_of_interest}_{Direction_Desc}_Pushing_Feats___{final_model_identifier}.json"), "w") as json_file:
         json.dump(result_dict, json_file) 




   result_dict
   print()








