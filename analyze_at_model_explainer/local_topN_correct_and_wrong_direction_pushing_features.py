
'''

TODO: Write a script that, for each prediction, returns Top N
features that pushed towards wrong prediction, and Top N features
that pushed towards correct direction , based on local SHAP explanations. 

– Apply the script to mispredictions, and try get statistics
on which features were most responsible for mispredictions (Dr. Xiaokui Case-Study)

'''


import os
import pandas as pd

if __name__ == "__main__":

   # Set following

   final_test_model_dirpath = \
   "/home/jgwak1/tabby/graph_embedding_improvement_JY_git/graph_embedding_improvement_efforts/Trial_7__Thread_level_N_grams__N_gt_than_1__Similar_to_PriorGraphEmbedding/RESULTS/RandomForest__Full_Dataset_2_Double_Stratified__Best_RF__Full_Dataset_2_Double_Stratified__4gram__sum_pool__final_test__thread_level__N>1_grams_events__nodetype5bit__4gram__sum_pool__2024-02-13_121930"

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
   
   # -------------------------------------------------------------------------------------------------------------------------------------------------------
   samples_of_interest = Mispredictions

   Local_SHAP_vals_TestDataset.set_index("data_name", inplace = True)

   for test_data in Local_SHAP_vals_TestDataset.index:

      if test_data in samples_of_interest:

         if test_data.startswith("malware"):
            
            # if 'malware', then features that push to correct direction have 'positive' local-shap value

            pass

         elif test_data.startswith("benign"):

            # if 'benign', then features that push to correct direction have 'negative' local-shap value

            pass


         else:
            raise ValueError(f"{test_data} should either start with 'malware' or 'benign'")


         Local_SHAP_vals_TestDataset.loc[test_data]   
   

   


   print()