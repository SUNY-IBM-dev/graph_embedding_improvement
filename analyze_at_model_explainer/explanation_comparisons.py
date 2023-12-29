import os
import pandas as pd


'''

TODO: More Modularize the code as it is quite messy making it error-prone

'''

if __name__ == "__main__":


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


   graph_embedding__mispredictions__dirpath = \
      "/data/d1/jgwak1/tabby/graph_embedding_improvement_JY_git/analyze_at_model_explainer/RESULTS/sklearn.ensemble._forest.RandomForestClassifier__Dataset-Case-2__RandomForest_best_hyperparameter_mean_case2__final_test__signal_amplified__event_1gram_nodetype_5bit__mean__2023-12-21_194828/WATERFALL_PLOTS_Local-Explanation_1gram/Mispredictions"
   no_graph__mispredictions__dirpath = \
      "/data/d1/jgwak1/tabby/graph_embedding_improvement_JY_git/analyze_at_model_explainer/RESULTS/sklearn.ensemble._forest.RandomForestClassifier__Dataset-Case-2__RandomForest_best_hyperparameter_mean_case2_nograph__final_test__no_graph_structure__event_1gram_nodetype_5bit__mean__2023-12-21_194942/WATERFALL_PLOTS_Local-Explanation_1gram/Mispredictions"


   prefix_to_drop = "1-gram SHAP_local_interpretability_waterfall_plot_"
   suffix_to_drop = ".png"

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
   print(f"\n[ mispredictions only with 'graph-embedding' (#{len(mispredictions__only_in__graph_embedding)}) ]\n")
   print(*mispredictions__only_in__graph_embedding, sep="\n")
   print(f"\n[ mispredictions only with 'flattened-graph' (#{len(mispredictions__only_in__no_graph)}) ]\n")
   print(*mispredictions__only_in__no_graph, sep="\n")
   print(f"\n[ intersectiong mispredictions (#{len(mispredictions__intersecting)}) ]\n")
   print(*mispredictions__intersecting, sep="\n")


   # JY @ 2023-12-28:
   # TODO: Write script to compare in local-explanation level here
   # Refer to : https://docs.google.com/spreadsheets/d/1It7q8PZcwHNsWK9OALr3YT9B09uH_JkuZIiN9Q3sqaM/edit#gid=546272642

   ''' first get all dirpaths/filepaths/datasets that may contain result of interest (e.g.waterfallplots-dirpath) '''
   # get dirpaths 
   graph_embedding__waterfallplots__dirpath = os.path.split(graph_embedding__mispredictions__dirpath)[0]
   no_graph__waterfallplots__dirpath = os.path.split(no_graph__mispredictions__dirpath)[0]

   graph_embedding__correctpredictions__dirpath = os.path.join(graph_embedding__waterfallplots__dirpath,"Correct_Predictions")
   no_graph__correctpredictions__dirpath = os.path.join(no_graph__waterfallplots__dirpath,"Correct_Predictions")

   graph_embedding__results__dirpath = os.path.split( graph_embedding__waterfallplots__dirpath )[0]
   no_graph__results__dirpath = os.path.split( no_graph__waterfallplots__dirpath )[0]

   # get fnames ad fpaths
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



   ''' Compare local explanations in various aspects (feature-value, feature-shap, predict-proba, waterfall plot etc)
       for the mispredicted samples (later could do this for correct-predicted samples, but start from here)

       Save the comparison-results to some dir(**)  -- /data/d1/jgwak1/tabby/graph_embedding_improvement_JY_git/analyze_at_model_explainer/COMPARISON_RESULTS
       Give all details 
   '''
   COMPARISON_RESULTS_DIRPATH = "/data/d1/jgwak1/tabby/graph_embedding_improvement_JY_git/analyze_at_model_explainer/COMPARISON_RESULTS"
   if not os.path.exists(COMPARISON_RESULTS_DIRPATH):
      os.makedirs(COMPARISON_RESULTS_DIRPATH)



   # intersecting mispredictions ----------------------------------------
   for intersecting_mispred_sample in mispredictions__intersecting:

      ### JY @ 2023-12-28: maybe wrap following into a function to be reused

      # Extract from GlobalSHAP-TestDataset for comparison
      graph_embedding__GlobalSHAP_TestDataset_df__row_of_interest = \
         graph_embedding__GlobalSHAP_TestDataset_df[ graph_embedding__GlobalSHAP_TestDataset_df['data_name'].str.contains(intersecting_mispred_sample) ]

      graph_embedding__sum_of_feature_shaps = graph_embedding__GlobalSHAP_TestDataset_df__row_of_interest['SHAP_sum_of_feature_shaps']
      graph_embedding__base_value = graph_embedding__GlobalSHAP_TestDataset_df__row_of_interest['SHAP_base_value']
      graph_embedding__predict_proba = graph_embedding__GlobalSHAP_TestDataset_df__row_of_interest['predict_proba']


      no_graph__GlobalSHAP_TestDataset_df__row_of_interest = \
         no_graph__GlobalSHAP_TestDataset_csv__df[ no_graph__GlobalSHAP_TestDataset_csv__df['data_name'].str.contains(intersecting_mispred_sample) ]

      no_graph__sum_of_feature_shaps = no_graph__GlobalSHAP_TestDataset_df__row_of_interest['SHAP_sum_of_feature_shaps']
      no_graph__base_value = no_graph__GlobalSHAP_TestDataset_df__row_of_interest['SHAP_base_value']
      no_graph__predict_proba =no_graph__GlobalSHAP_TestDataset_df__row_of_interest['predict_proba']

      # waterfall plots

      # (do this?) get manually feature values, and feature shaps
      
      graph_embedding__intersecting_mispred_sample__fpath =  [f for f in os.listdir(graph_embedding__mispredictions__dirpath) \
                                                              if intersecting_mispred_sample in f ][0] 


      no_graph__intersecting_mispred_sample__fpath = [f for f in os.listdir(no_graph__mispredictions__dirpath) \
                                                              if intersecting_mispred_sample in f ][0] 


      # TODO: WRITE EVERYTHING TO COMPARISON_RESULTS DIR FOR SMOOTH ANALYSIS


      pass


   # mispredictions__only_in__graph_embedding ----------------------------------------
   for only_graph_embedding_mispred_sample in mispredictions__only_in__graph_embedding:
      pass


   # mispredictions__only_in__no_graph ----------------------------------------
   for only_no_graph_mispred_sample in mispredictions__only_in__no_graph:
      pass 