import os
import pandas as pd
from datetime import datetime
import shutil

'''

TODO: explanation_comparisons.py KIND OF WORKING BUT NEEDS DEBUGGING

'''


def predictions_comparisons(graph_embedding__mispredictions__dirpath : str, 
                               no_graph__mispredictions__dirpath: str) -> dict:


   prefix_to_drop = "1-gram SHAP_local_interpretability_waterfall_plot_"
   suffix_to_drop = ".png"

   # ----------------------------------------------------------------------------------------
   # Get correct predictions by both ------------------------------------------------------------------------------------------

   graph_embedding__waterfallplots__dirpath = os.path.split(graph_embedding__mispredictions__dirpath)[0]
   no_graph__waterfallplots__dirpath = os.path.split(no_graph__mispredictions__dirpath)[0]

   graph_embedding__correctpredictions__dirpath = os.path.join(graph_embedding__waterfallplots__dirpath,"Correct_Predictions")
   no_graph__correctpredictions__dirpath = os.path.join(no_graph__waterfallplots__dirpath,"Correct_Predictions")

   graph_embedding__correctpredictions__datanames = \
      { x.removeprefix(prefix_to_drop).removesuffix(suffix_to_drop) for x in os.listdir(graph_embedding__correctpredictions__dirpath) }
   no_graph__correctpredictions__datanames = { x.removeprefix(prefix_to_drop).removesuffix(suffix_to_drop) for x in os.listdir(no_graph__correctpredictions__dirpath) }  

   correctpredictions_by_both = list( graph_embedding__correctpredictions__datanames.intersection(no_graph__correctpredictions__datanames) )

   # ------------------------------------------------------------------------------------------
   # Do misprediction comparisons
   graph_embedding__mispredictions__datanames = \
      { x.removeprefix(prefix_to_drop).removesuffix(suffix_to_drop) for x in os.listdir(graph_embedding__mispredictions__dirpath) }
   no_graph__mispredictions__datanames = { x.removeprefix(prefix_to_drop).removesuffix(suffix_to_drop) for x in os.listdir(no_graph__mispredictions__dirpath) }  


   mispredictions__only_in__graph_embedding = list(graph_embedding__mispredictions__datanames - no_graph__mispredictions__datanames )
   mispredictions__only_in__no_graph = list( no_graph__mispredictions__datanames - graph_embedding__mispredictions__datanames )
   mispredictions__intersecting = list( graph_embedding__mispredictions__datanames.intersection(no_graph__mispredictions__datanames) )

   # ugly 
   graph_embedding_info = os.path.split(os.path.split(os.path.split(graph_embedding__mispredictions__dirpath)[0])[0])[1].removeprefix("sklearn.ensemble._forest.RandomForestClassifier__")
   no_graph_info = os.path.split(os.path.split(os.path.split(no_graph__mispredictions__dirpath)[0])[0])[1].removeprefix("sklearn.ensemble._forest.RandomForestClassifier__")

   print("-"*100, flush= True)
   print(f"\n\nRF + graph-embedding trial info: {graph_embedding_info}", flush = True )
   print(f"RF + flattened-graph trial info: {no_graph_info}\n", flush = True )   
   
   print(f"\n[ correct predictions by both (#{len(correctpredictions_by_both)}) ]\n")
   print(*correctpredictions_by_both, sep="\n")

   print(f"\n[ mispredictions only with 'graph-embedding' (#{len(mispredictions__only_in__graph_embedding)}) ]\n")
   print(*mispredictions__only_in__graph_embedding, sep="\n")

   print(f"\n[ mispredictions only with 'flattened-graph' (#{len(mispredictions__only_in__no_graph)}) ]\n")
   print(*mispredictions__only_in__no_graph, sep="\n")

   print(f"\n[ intersectiong mispredictions (#{len(mispredictions__intersecting)}) ]\n")
   print(*mispredictions__intersecting, sep="\n")
   
   return {"correctpredictions_by_both" : correctpredictions_by_both,
           "mispredictions__intersecting": mispredictions__intersecting, 
           "mispredictions__only_in__graph_embedding": mispredictions__only_in__graph_embedding, 
           "mispredictions__only_in__no_graph": mispredictions__only_in__no_graph}





def get_materials_for_explanation_comparison(graph_embedding__mispredictions__dirpath : str, 
                                             no_graph__mispredictions__dirpath : str) -> dict:

      ''' first get all dirpaths/filepaths/datasets that may contain result of interest (e.g.waterfallplots-dirpath) '''
      # get dirpaths 
      graph_embedding__waterfallplots__dirpath = os.path.split(graph_embedding__mispredictions__dirpath)[0]
      no_graph__waterfallplots__dirpath = os.path.split(no_graph__mispredictions__dirpath)[0]

      graph_embedding__correctpredictions__dirpath = os.path.join(graph_embedding__waterfallplots__dirpath,"Correct_Predictions")
      no_graph__correctpredictions__dirpath = os.path.join(no_graph__waterfallplots__dirpath,"Correct_Predictions")

      graph_embedding__results__dirpath = os.path.split( graph_embedding__waterfallplots__dirpath )[0]
      no_graph__results__dirpath = os.path.split( no_graph__waterfallplots__dirpath )[0]

      # get fnames ad fpaths in preparation of reading in dfs
      graph_embedding__GlobalSHAP_TestDataset_csv__fname = [f for f in os.listdir(graph_embedding__results__dirpath) if "Test-Dataset.csv" in f][0]
      graph_embedding__GlobalSHAP_TestDataset_csv__fpath = os.path.join(graph_embedding__results__dirpath,
                                                                        graph_embedding__GlobalSHAP_TestDataset_csv__fname)
      no_graph__GlobalSHAP_TestDataset_csv__fname = [f for f in os.listdir(no_graph__results__dirpath) if "Test-Dataset.csv" in f][0]
      no_graph__GlobalSHAP_TestDataset_csv__fpath = os.path.join(no_graph__results__dirpath,
                                                               no_graph__GlobalSHAP_TestDataset_csv__fname)
      
      graph_embedding__GlobalSHAP_csv__fname = [f for f in os.listdir(graph_embedding__results__dirpath) if "Global-SHAP Importance.csv" in f][0]
      graph_embedding__GlobalSHAP_csv__fpath = os.path.join(graph_embedding__results__dirpath, graph_embedding__GlobalSHAP_csv__fname)
      no_graph__GlobalSHAP_csv__fname = [f for f in os.listdir(no_graph__results__dirpath) if "Global-SHAP Importance.csv" in f][0]
      no_graph__GlobalSHAP_csv__fpath = os.path.join(no_graph__results__dirpath,no_graph__GlobalSHAP_csv__fname)

      graph_embedding__final_test_results_csv__fname = [f for f in os.listdir(graph_embedding__results__dirpath) if ("Global-SHAP" not in f) and ("WATERFALL" not in f) and ("png" not in f)][0]
      graph_embedding__final_test_results_csv__fpath = os.path.join(graph_embedding__results__dirpath, graph_embedding__final_test_results_csv__fname)
      no_graph__final_test_results_csv__fname = [f for f in os.listdir(no_graph__results__dirpath) if ("Global-SHAP" not in f) and ("WATERFALL" not in f) and ("png" not in f)][0]
      no_graph__final_test_results_csv__fpath = os.path.join(no_graph__results__dirpath, no_graph__final_test_results_csv__fname)

      # Prepare all dfs 

      # -- GlobalSHAP Test-Dataset dataframes
      graph_embedding__GlobalSHAP_TestDataset_df = pd.read_csv(graph_embedding__GlobalSHAP_TestDataset_csv__fpath)
      no_graph__GlobalSHAP_TestDataset_csv__df = pd.read_csv(no_graph__GlobalSHAP_TestDataset_csv__fpath)

      # -- GlobalSHAP dataframes
      graph_embedding__GlobalSHAP_df = pd.read_csv(graph_embedding__GlobalSHAP_csv__fpath)
      no_graph__GlobalSHAP_df = pd.read_csv(no_graph__GlobalSHAP_csv__fpath)

      # -- final test results dataframes
      graph_embedding__final_test_results_df = pd.read_csv(graph_embedding__final_test_results_csv__fpath)
      no_graph__final_test_results_df = pd.read_csv(no_graph__final_test_results_csv__fpath)

      return {
         # dataframes that might be useful
         "graph_embedding__GlobalSHAP_TestDataset_df": graph_embedding__GlobalSHAP_TestDataset_df,
         "no_graph__GlobalSHAP_TestDataset_csv__df": no_graph__GlobalSHAP_TestDataset_csv__df,
         
         "graph_embedding__GlobalSHAP_df": graph_embedding__GlobalSHAP_df,
         "no_graph__GlobalSHAP_df": no_graph__GlobalSHAP_df,
         
         "graph_embedding__final_test_results_df": graph_embedding__final_test_results_df,
         "no_graph__final_test_results_df": no_graph__final_test_results_df,

         # dirpaths that might be useful
         "graph_embedding__mispredictions__dirpath": graph_embedding__mispredictions__dirpath,
         "no_graph__mispredictions__dirpath": no_graph__mispredictions__dirpath,

         "graph_embedding__correctpredictions__dirpath": graph_embedding__correctpredictions__dirpath,
         "no_graph__correctpredictions__dirpath": no_graph__correctpredictions__dirpath,

         "graph_embedding__results__dirpath": graph_embedding__results__dirpath,
         "no_graph__results__dirpath": no_graph__results__dirpath
      }



def produce_explanation_comparisons(predictions_comparisons__dict: dict , 
                                    materials_for_explanation_comparison__dict :dict):
   

      ''' Compare local explanations in various aspects (feature-value, feature-shap, predict-proba, waterfall plot etc)
         for the mispredicted samples (later could do this for correct-predicted samples, but start from here)

         Save the comparison-results to some dir(**)  -- /data/d1/jgwak1/tabby/graph_embedding_improvement_JY_git/analyze_at_model_explainer/COMPARISON_RESULTS
         Give all details 
      '''

      # 1. Create directory for this trial's explanation comparisons
      EXPLANATION_COMPARISONS_DIRPATH = \
         "/data/d1/jgwak1/tabby/graph_embedding_improvement_JY_git/analyze_at_model_explainer/EXPLANATION_COMPARISONS"
      explanation_comparison_trial_dirpath = os.path.join(EXPLANATION_COMPARISONS_DIRPATH,
                                                         f"Explanation_Comparison___@_{datetime.now().strftime('%Y-%m-%d_%H%M%S')}")
      
      if not os.path.exists(EXPLANATION_COMPARISONS_DIRPATH):
               os.makedirs(EXPLANATION_COMPARISONS_DIRPATH)
      os.makedirs(explanation_comparison_trial_dirpath)   

      # 2. Add a explanantion-comparison description file
      with open( os.path.join(explanation_comparison_trial_dirpath, "explanation_comparison_description.txt"), "w" ) as f:
         f.write("explanation comparisons between:\n\n")
         f.write(f"  {materials_for_explanation_comparison__dict['graph_embedding__results__dirpath']}\n")
         f.write("    vs.\n")
         f.write(f"  {materials_for_explanation_comparison__dict['no_graph__results__dirpath']}\n")

      # 3. Create 'explanation comparison' sub-directories for the following 4 categories:
      #      category-1: samples which both graph-embedding and no-graph mis-predicted
      #      category-2: samples which both graph-embedding and no-graph correctly-predicted 
      #      category-3: samples which only graph-embedding mispredicted (no-graph correctly-predicted) 
      #      category-4: samples which only no-graph mispredicted (graph-embedding correctly-predicted)
         
      category_1__dirpath = os.path.join( explanation_comparison_trial_dirpath, "both_mispredicted") 
      category_2__dirpath = os.path.join( explanation_comparison_trial_dirpath, "both_correctly_predicted") 
      category_3__dirpath = os.path.join( explanation_comparison_trial_dirpath, "only_graph_embedding_mispredicted") 
      category_4__dirpath = os.path.join( explanation_comparison_trial_dirpath, "only_no_graph_mispredicted") 
      
      os.makedirs( os.path.join( explanation_comparison_trial_dirpath , category_1__dirpath) )
      os.makedirs( os.path.join( explanation_comparison_trial_dirpath , category_2__dirpath) )
      os.makedirs( os.path.join( explanation_comparison_trial_dirpath , category_3__dirpath) )
      os.makedirs( os.path.join( explanation_comparison_trial_dirpath , category_4__dirpath) )

      # ==============================================================================================================
      # 4. Write out to roduce explanations for each cateogry

      # prep variables and helper func
      graph_embedding__GlobalSHAP_TestDataset_df = materials_for_explanation_comparison__dict["graph_embedding__GlobalSHAP_TestDataset_df"]
      no_graph__GlobalSHAP_TestDataset_csv__df = materials_for_explanation_comparison__dict["no_graph__GlobalSHAP_TestDataset_csv__df"]
      
      
      def get_sumShap_basevalue_predictprobas(sample : str):
      
         # Extract from GlobalSHAP-TestDataset for comparison
         graph_embedding__GlobalSHAP_TestDataset_df__row_of_interest = \
            graph_embedding__GlobalSHAP_TestDataset_df[ graph_embedding__GlobalSHAP_TestDataset_df['data_name'].str.contains(sample) ]

         graph_embedding__sum_of_feature_shaps = graph_embedding__GlobalSHAP_TestDataset_df__row_of_interest['SHAP_sum_of_feature_shaps']
         graph_embedding__base_value = graph_embedding__GlobalSHAP_TestDataset_df__row_of_interest['SHAP_base_value']
         graph_embedding__predict_proba = graph_embedding__GlobalSHAP_TestDataset_df__row_of_interest['predict_proba']


         no_graph__GlobalSHAP_TestDataset_df__row_of_interest = \
            no_graph__GlobalSHAP_TestDataset_csv__df[ no_graph__GlobalSHAP_TestDataset_csv__df['data_name'].str.contains(sample) ]

         no_graph__sum_of_feature_shaps = no_graph__GlobalSHAP_TestDataset_df__row_of_interest['SHAP_sum_of_feature_shaps']
         no_graph__base_value = no_graph__GlobalSHAP_TestDataset_df__row_of_interest['SHAP_base_value']
         no_graph__predict_proba =no_graph__GlobalSHAP_TestDataset_df__row_of_interest['predict_proba']

         return {"graph_embedding__sum_of_feature_shaps": graph_embedding__sum_of_feature_shaps,
                 "graph_embedding__base_value": graph_embedding__base_value,
                 "graph_embedding__predict_proba": graph_embedding__predict_proba,
                 "no_graph__sum_of_feature_shaps": no_graph__sum_of_feature_shaps,
                 "no_graph__base_value": no_graph__base_value,
                 "no_graph__predict_proba": no_graph__predict_proba}
      

      def get_waterfallplots_and_copy(sample : str, explanation_comparision_savedir : str):

         graph_embedding__intersecting_mispred_sample__fname =  [f for f in os.listdir(graph_embedding__mispredictions__dirpath) \
                                                                 if sample in f ][0] 
         graph_embedding__intersecting_mispred_sample__fpath = os.path.join(graph_embedding__mispredictions__dirpath, 
                                                                            graph_embedding__intersecting_mispred_sample__fname)
         shutil.copy(graph_embedding__intersecting_mispred_sample__fpath, 
                     os.path.join(explanation_comparision_savedir, graph_embedding__intersecting_mispred_sample__fname))

         no_graph__intersecting_mispred_sample__fname = [f for f in os.listdir(no_graph__mispredictions__dirpath) \
                                                               if sample in f ][0] 
         no_graph__intersecting_mispred_sample__fpath = os.path.join(no_graph__mispredictions__dirpath, 
                                                                     no_graph__intersecting_mispred_sample__fname)
         shutil.copy(no_graph__intersecting_mispred_sample__fpath, 
                     os.path.join(explanation_comparision_savedir, no_graph__intersecting_mispred_sample__fname))          


      # ---------------------------------------------------------------------------------------------------
      # ---- category-1 -- both_mispredicted
      for sample in predictions_comparisons__dict["mispredictions__intersecting"]: 

         explanation_comparision_savedir = os.path.join(category_1__dirpath, sample)
         os.makedirs(explanation_comparision_savedir)

         # JY @ 2023-12-28: TODO : could also saveout outputs in results_dict to explanation_comparision_savedir
         results_dict = get_sumShap_basevalue_predictprobas(sample)

         get_waterfallplots_and_copy(sample, explanation_comparision_savedir)


         # waterfall plots

         # (do this?) get manually feature values, and feature shaps
         
         # could wrap this into a 


         # TODO: WRITE EVERYTHING TO COMPARISON_RESULTS DIR FOR SMOOTH ANALYSIS


      
      # ---------------------------------------------------------------------------------------------------
      # ---- category-2 -- both_correctly_predicted
      for sample in predictions_comparisons__dict["correctpredictions_by_both"]:
         
         explanation_comparision_savedir = os.path.join(category_2__dirpath, sample)
         os.makedirs(explanation_comparision_savedir)

         results_dict = get_sumShap_basevalue_predictprobas(sample)
         get_waterfallplots_and_copy(sample, explanation_comparision_savedir)


      # ---------------------------------------------------------------------------------------------------
      # ---- category-3 -- only_graph_embedding_mispredicted
      for sample in predictions_comparisons__dict["mispredictions__only_in__graph_embedding"]:

         explanation_comparision_savedir = os.path.join(category_3__dirpath, sample)
         os.makedirs(explanation_comparision_savedir)

         results_dict = get_sumShap_basevalue_predictprobas(sample)
         get_waterfallplots_and_copy(sample, explanation_comparision_savedir)


      # ---------------------------------------------------------------------------------------------------
      # ---- category-4 -- only_no_graph_mispredicted
      for sample in predictions_comparisons__dict["mispredictions__only_in__no_graph"]:

         explanation_comparision_savedir = os.path.join(category_4__dirpath, sample)
         os.makedirs(explanation_comparision_savedir)

         results_dict = get_sumShap_basevalue_predictprobas(sample)
         get_waterfallplots_and_copy(sample, explanation_comparision_savedir)



      # ---------------------------------------------------------------------------------------------------
      return 


if __name__ == "__main__":

   # JY @ 2023-12-28:
   # Refer to : https://docs.google.com/spreadsheets/d/1It7q8PZcwHNsWK9OALr3YT9B09uH_JkuZIiN9Q3sqaM/edit#gid=546272642



   # JY @ 2023-12-28: would make more sense to change following to "results_dirpath" instead of "mispredictions-dirpath"
   #                  but just keep it as it is for now

   # -----------------------------------------------------------------------------------------------------------------------------

   # graph_embedding__mispredictions__dirpath = \
   #    "/data/d1/jgwak1/tabby/graph_embedding_improvement_JY_git/analyze_at_model_explainer/RESULTS/sklearn.ensemble._forest.RandomForestClassifier__Dataset-Case-1__RandomForest_best_hyperparameter_max_case1__final_test__signal_amplified__event_1gram_nodetype_5bit__max__2023-12-21_193418/WATERFALL_PLOTS_Local-Explanation_1gram/Mispredictions"
   # no_graph__mispredictions__dirpath = \
   #    "/data/d1/jgwak1/tabby/graph_embedding_improvement_JY_git/analyze_at_model_explainer/RESULTS/sklearn.ensemble._forest.RandomForestClassifier__Dataset-Case-1__RandomForest_best_hyperparameter_max_case1_nograph__final_test__no_graph_structure__event_1gram_nodetype_5bit__max__2023-12-21_193512/WATERFALL_PLOTS_Local-Explanation_1gram/Mispredictions"


   # graph_embedding__mispredictions__dirpath = \
   #    "/data/d1/jgwak1/tabby/graph_embedding_improvement_JY_git/analyze_at_model_explainer/RESULTS/sklearn.ensemble._forest.RandomForestClassifier__Dataset-Case-1__RandomForest_best_hyperparameter_mean_case1__final_test__signal_amplified__event_1gram_nodetype_5bit__mean__2023-12-21_194437/WATERFALL_PLOTS_Local-Explanation_1gram/Mispredictions"
   # no_graph__mispredictions__dirpath = \
   #    "/data/d1/jgwak1/tabby/graph_embedding_improvement_JY_git/analyze_at_model_explainer/RESULTS/sklearn.ensemble._forest.RandomForestClassifier__Dataset-Case-1__RandomForest_best_hyperparameter_mean_case1_nograph__final_test__no_graph_structure__event_1gram_nodetype_5bit__mean__2023-12-21_194534/WATERFALL_PLOTS_Local-Explanation_1gram/Mispredictions"


   # graph_embedding__mispredictions__dirpath = \
   #    "/data/d1/jgwak1/tabby/graph_embedding_improvement_JY_git/analyze_at_model_explainer/RESULTS/sklearn.ensemble._forest.RandomForestClassifier__Dataset-Case-2__RandomForest_best_hyperparameter_max_case2__final_test__signal_amplified__event_1gram_nodetype_5bit__max__2023-12-21_194005/WATERFALL_PLOTS_Local-Explanation_1gram/Mispredictions"
   # no_graph__mispredictions__dirpath = \
   #    "/data/d1/jgwak1/tabby/graph_embedding_improvement_JY_git/analyze_at_model_explainer/RESULTS/sklearn.ensemble._forest.RandomForestClassifier__Dataset-Case-2__RandomForest_best_hyperparameter_max_case2_nograph__final_test__no_graph_structure__event_1gram_nodetype_5bit__max__2023-12-21_194206/WATERFALL_PLOTS_Local-Explanation_1gram/Mispredictions"

   # -----------------------------------------------------------------------------------------------------------------------------


   graph_embedding__mispredictions__dirpath = \
      "/data/d1/jgwak1/tabby/graph_embedding_improvement_JY_git/analyze_at_model_explainer/RESULTS/sklearn.ensemble._forest.RandomForestClassifier__Dataset-Case-2__RandomForest_best_hyperparameter_mean_case2__final_test__signal_amplified__event_1gram_nodetype_5bit__mean__2023-12-21_194828/WATERFALL_PLOTS_Local-Explanation_1gram/Mispredictions"
   no_graph__mispredictions__dirpath = \
      "/data/d1/jgwak1/tabby/graph_embedding_improvement_JY_git/analyze_at_model_explainer/RESULTS/sklearn.ensemble._forest.RandomForestClassifier__Dataset-Case-2__RandomForest_best_hyperparameter_mean_case2_nograph__final_test__no_graph_structure__event_1gram_nodetype_5bit__mean__2023-12-21_194942/WATERFALL_PLOTS_Local-Explanation_1gram/Mispredictions"


   predictions_comparisons_dict = predictions_comparisons(graph_embedding__mispredictions__dirpath, no_graph__mispredictions__dirpath)
   materials_for_explanation_comparison_dict = get_materials_for_explanation_comparison(graph_embedding__mispredictions__dirpath, no_graph__mispredictions__dirpath)

   produce_explanation_comparisons(predictions_comparisons__dict = predictions_comparisons_dict , 
                                   materials_for_explanation_comparison__dict = materials_for_explanation_comparison_dict)





