import os
import pandas as pd
from datetime import datetime
import shutil

import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import json


def predictions_comparisons(graph_embedding__mispredictions__dirpath : str, 
                               no_graph__mispredictions__dirpath: str) -> dict:


   prefix_to_drop = "1-gram SHAP_local_interpretability_waterfall_plot_"
   suffix_to_drop = ".png"

   possibly_drop = "SUBGRAPH_P3_" # Added by JY @ 2024-1-4: Priti's files contain "SUBGRAPH_P3_" in the middle that should be dropped for matching

   # ----------------------------------------------------------------------------------------
   # Get correct predictions by both ------------------------------------------------------------------------------------------

   graph_embedding__waterfallplots__dirpath = os.path.split(graph_embedding__mispredictions__dirpath)[0]
   no_graph__waterfallplots__dirpath = os.path.split(no_graph__mispredictions__dirpath)[0]

   graph_embedding__correctpredictions__dirpath = os.path.join(graph_embedding__waterfallplots__dirpath,"Correct_Predictions")
   no_graph__correctpredictions__dirpath = os.path.join(no_graph__waterfallplots__dirpath,"Correct_Predictions")

   graph_embedding__correctpredictions__datanames = \
      { x.removeprefix(prefix_to_drop).removesuffix(suffix_to_drop).replace(possibly_drop,"") for x in os.listdir(graph_embedding__correctpredictions__dirpath) }
   no_graph__correctpredictions__datanames = { x.removeprefix(prefix_to_drop).removesuffix(suffix_to_drop).replace(possibly_drop,"") for x in os.listdir(no_graph__correctpredictions__dirpath) }  

   correctpredictions_by_both = list( graph_embedding__correctpredictions__datanames.intersection(no_graph__correctpredictions__datanames) )

   # ------------------------------------------------------------------------------------------
   # Do misprediction comparisons
   graph_embedding__mispredictions__datanames = \
      { x.removeprefix(prefix_to_drop).removesuffix(suffix_to_drop).replace(possibly_drop,"") for x in os.listdir(graph_embedding__mispredictions__dirpath) }
   no_graph__mispredictions__datanames = { x.removeprefix(prefix_to_drop).removesuffix(suffix_to_drop).replace(possibly_drop,"") for x in os.listdir(no_graph__mispredictions__dirpath) }  


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

   print(f"\n[ intersecting mispredictions (#{len(mispredictions__intersecting)}) ]\n")
   print(*mispredictions__intersecting, sep="\n")
   
   return {"correctpredictions_by_both" : correctpredictions_by_both,
           "mispredictions__intersecting": mispredictions__intersecting, 
           "mispredictions__only_in__graph_embedding": mispredictions__only_in__graph_embedding, 
           "mispredictions__only_in__no_graph": mispredictions__only_in__no_graph}





def get_materials_for_explanation_comparison(graph_embedding__mispredictions__dirpath : str, 
                                             no_graph__mispredictions__dirpath : str) -> dict:

      ''' '''

      ''' first get all dirpaths/filepaths/datasets that may contain result of interest (e.g.waterfallplots-dirpath) '''
      # get dirpaths 
      graph_embedding__waterfallplots__dirpath = os.path.split(graph_embedding__mispredictions__dirpath)[0]
      no_graph__waterfallplots__dirpath = os.path.split(no_graph__mispredictions__dirpath)[0]

      graph_embedding__correctpredictions__dirpath = os.path.join(graph_embedding__waterfallplots__dirpath,"Correct_Predictions")
      no_graph__correctpredictions__dirpath = os.path.join(no_graph__waterfallplots__dirpath,"Correct_Predictions")

      graph_embedding__results__dirpath = os.path.split( graph_embedding__waterfallplots__dirpath )[0]
      no_graph__results__dirpath = os.path.split( no_graph__waterfallplots__dirpath )[0]

      # get fnames ad fpaths in preparation of reading in dfs ( modified by JY @ 2024-1-10)
      graph_embedding__GlobalSHAP_TestDataset_csv__fname = [f for f in os.listdir(graph_embedding__results__dirpath) if "Global-SHAP Important FeatureNames Test-Dataset" in f][0]
      graph_embedding__GlobalSHAP_TestDataset_csv__fpath = os.path.join(graph_embedding__results__dirpath,
                                                                        graph_embedding__GlobalSHAP_TestDataset_csv__fname)
      no_graph__GlobalSHAP_TestDataset_csv__fname = [f for f in os.listdir(no_graph__results__dirpath) if "Global-SHAP Important FeatureNames Test-Dataset" in f][0]
      no_graph__GlobalSHAP_TestDataset_csv__fpath = os.path.join(no_graph__results__dirpath,
                                                               no_graph__GlobalSHAP_TestDataset_csv__fname)
      
      # ( Added by JY @ 2024-1-10)
      graph_embedding__LocalSHAP_vals_TestDataset_csv__fname = [f for f in os.listdir(graph_embedding__results__dirpath) if "Local-SHAP values Test-Dataset" in f][0]
      graph_embedding__LocalSHAP_vals_TestDataset_csv__fpath = os.path.join(graph_embedding__results__dirpath,
                                                                            graph_embedding__LocalSHAP_vals_TestDataset_csv__fname)
      no_graph__LocalSHAP_vals_TestDataset_csv__fname = [f for f in os.listdir(no_graph__results__dirpath) if "Local-SHAP values Test-Dataset" in f][0]
      no_graph__LocalSHAP_vals_TestDataset_csv__fpath = os.path.join(no_graph__results__dirpath,
                                                                     no_graph__LocalSHAP_vals_TestDataset_csv__fname)


      graph_embedding__GlobalSHAP_csv__fname = [f for f in os.listdir(graph_embedding__results__dirpath) if "Global-SHAP Importance.csv" in f][0]
      graph_embedding__GlobalSHAP_csv__fpath = os.path.join(graph_embedding__results__dirpath, graph_embedding__GlobalSHAP_csv__fname)
      no_graph__GlobalSHAP_csv__fname = [f for f in os.listdir(no_graph__results__dirpath) if "Global-SHAP Importance.csv" in f][0]
      no_graph__GlobalSHAP_csv__fpath = os.path.join(no_graph__results__dirpath,no_graph__GlobalSHAP_csv__fname)

      # ( modified by JY @ 2024-1-10)
      graph_embedding__final_test_results_csv__fname = [f for f in os.listdir(graph_embedding__results__dirpath) if ("Global-SHAP" not in f) and ("Local-SHAP" not in f) and ("WATERFALL" not in f) and ("png" not in f)][0]
      graph_embedding__final_test_results_csv__fpath = os.path.join(graph_embedding__results__dirpath, graph_embedding__final_test_results_csv__fname)
      no_graph__final_test_results_csv__fname = [f for f in os.listdir(no_graph__results__dirpath) if ("Global-SHAP" not in f) and ("Local-SHAP" not in f) and ("WATERFALL" not in f) and ("png" not in f)][0]
      no_graph__final_test_results_csv__fpath = os.path.join(no_graph__results__dirpath, no_graph__final_test_results_csv__fname)

      # Prepare all dfs 

      # -- GlobalSHAP Test-Dataset dataframes
      graph_embedding__GlobalSHAP_TestDataset_df = pd.read_csv(graph_embedding__GlobalSHAP_TestDataset_csv__fpath)
      no_graph__GlobalSHAP_TestDataset_csv__df = pd.read_csv(no_graph__GlobalSHAP_TestDataset_csv__fpath)

      # -- GlobalSHAP dataframes
      graph_embedding__GlobalSHAP_df = pd.read_csv(graph_embedding__GlobalSHAP_csv__fpath)
      no_graph__GlobalSHAP_df = pd.read_csv(no_graph__GlobalSHAP_csv__fpath)

      # -- LocalSHAP values dataframes
      graph_embedding__LocalSHAP_values_TestDataset_df = pd.read_csv(graph_embedding__LocalSHAP_vals_TestDataset_csv__fpath)
      no_graph__LocalSHAP_values_TestDataset_csv__df = pd.read_csv(no_graph__LocalSHAP_vals_TestDataset_csv__fpath)

      # -- final test results dataframes
      graph_embedding__final_test_results_df = pd.read_csv(graph_embedding__final_test_results_csv__fpath)
      no_graph__final_test_results_df = pd.read_csv(no_graph__final_test_results_csv__fpath)

      return {
         # dataframes that might be useful -- write these out to the trail-subdir 
         "graph_embedding__GlobalSHAP_TestDataset_df": graph_embedding__GlobalSHAP_TestDataset_df,
         "no_graph__GlobalSHAP_TestDataset_csv__df": no_graph__GlobalSHAP_TestDataset_csv__df,
         
         "graph_embedding__GlobalSHAP_df": graph_embedding__GlobalSHAP_df,
         "no_graph__GlobalSHAP_df": no_graph__GlobalSHAP_df,

         # ( Added by JY @ 2024-1-10)
         "graph_embedding__LocalSHAP_values_TestDataset_df": graph_embedding__LocalSHAP_values_TestDataset_df,
         "no_graph__LocalSHAP_values_TestDataset_csv__df": no_graph__LocalSHAP_values_TestDataset_csv__df,


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
         "/home/jgwak1/tabby/graph_embedding_improvement_JY_git/analyze_at_model_explainer/EXPLANATION_COMPARISONS"
      explanation_comparison_trial_dirpath = os.path.join(EXPLANATION_COMPARISONS_DIRPATH,
                                                         f"Explanation_Comparison___@_{datetime.now().strftime('%Y-%m-%d_%H%M%S')}")
      
      if not os.path.exists(EXPLANATION_COMPARISONS_DIRPATH):
               os.makedirs(EXPLANATION_COMPARISONS_DIRPATH)
      os.makedirs(explanation_comparison_trial_dirpath)   

      # 2. Add a explanantion-comparison description file
      with open( os.path.join(explanation_comparison_trial_dirpath, "description_of_explanation_comparison.txt"), "w" ) as f:
         f.write("explanation comparisons between:\n\n")
         f.write(f"  { os.path.split(materials_for_explanation_comparison__dict['graph_embedding__results__dirpath'])[1] } ( f{materials_for_explanation_comparison__dict['graph_embedding__results__dirpath']} ) \n")
         f.write("    vs.\n")
         f.write(f"  { os.path.split(materials_for_explanation_comparison__dict['no_graph__results__dirpath'])[1] }  ( f{materials_for_explanation_comparison__dict['no_graph__results__dirpath']} )\n")


      with open( os.path.join(explanation_comparison_trial_dirpath, "predictions_comparisons__dict.json"), "w" ) as f:
         json.dump(predictions_comparisons__dict, f)


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
      # 4. Save out these, to refer to, as can be needed when analyzing the explanation comparisons.
      materials_for_explanation_comparison__dict["graph_embedding__GlobalSHAP_TestDataset_df"].to_csv(os.path.join(explanation_comparison_trial_dirpath,"graph_embedding__GlobalSHAP_TestDataset_df.csv"))
      materials_for_explanation_comparison__dict["no_graph__GlobalSHAP_TestDataset_csv__df"].to_csv(os.path.join(explanation_comparison_trial_dirpath,"no_graph__GlobalSHAP_TestDataset_csv__df.csv"))

      materials_for_explanation_comparison__dict["graph_embedding__GlobalSHAP_df"].to_csv(os.path.join(explanation_comparison_trial_dirpath,"graph_embedding__GlobalSHAP_df.csv"))
      materials_for_explanation_comparison__dict["no_graph__GlobalSHAP_df"].to_csv(os.path.join(explanation_comparison_trial_dirpath,"no_graph__GlobalSHAP_df.csv"))      

      materials_for_explanation_comparison__dict["graph_embedding__final_test_results_df"].to_csv(os.path.join(explanation_comparison_trial_dirpath,"graph_embedding__final_test_results_df.csv"))
      materials_for_explanation_comparison__dict["no_graph__final_test_results_df"].to_csv(os.path.join(explanation_comparison_trial_dirpath,"no_graph__final_test_results_df.csv"))



      # ( Added by JY @ 2024-1-10)
      materials_for_explanation_comparison__dict["graph_embedding__LocalSHAP_values_TestDataset_df"].to_csv(os.path.join(explanation_comparison_trial_dirpath,"graph_embedding__LocalSHAP_values_TestDataset_df.csv"))
      materials_for_explanation_comparison__dict["no_graph__LocalSHAP_values_TestDataset_csv__df"].to_csv(os.path.join(explanation_comparison_trial_dirpath,"no_graph__LocalSHAP_values_TestDataset_csv__df.csv"))


      # ==============================================================================================================
      # 5. Write out to produce explanations for each cateogry

      # prep variables and helper func
      graph_embedding__GlobalSHAP_TestDataset_df = materials_for_explanation_comparison__dict["graph_embedding__GlobalSHAP_TestDataset_df"]
      no_graph__GlobalSHAP_TestDataset_csv__df = materials_for_explanation_comparison__dict["no_graph__GlobalSHAP_TestDataset_csv__df"]
      
      
      def get_sumShap_basevalue_predictprobas__and_saveout(sample : str,
                                                           explanation_comparision_savedir : str,
                                                           ):
      
         index_columns = ["data_name", "ROW_IDENTIFIER", "SHAP_sum_of_feature_shaps", "SHAP_base_value", "predict_proba", "SUM"]

         # Extract from GlobalSHAP-TestDataset for comparison

         if sample == "notepad++_context_menu": # since '++' causee a problem in str.contains()
            sample = "notepad"
         graph_embedding__GlobalSHAP_TestDataset_df__row_of_interest = \
            graph_embedding__GlobalSHAP_TestDataset_df[ graph_embedding__GlobalSHAP_TestDataset_df['data_name'].str.contains(sample) ]
         
         graph_embedding__GlobalSHAP_TestDataset_df__row_of_interest["ROW_IDENTIFIER"] = "Graph Embedding"
         

         no_graph__GlobalSHAP_TestDataset_df__row_of_interest = \
            no_graph__GlobalSHAP_TestDataset_csv__df[ no_graph__GlobalSHAP_TestDataset_csv__df['data_name'].str.contains(sample) ]
         no_graph__GlobalSHAP_TestDataset_df__row_of_interest["ROW_IDENTIFIER"] = "No Graph"


         # JY @ 2024-1-4: before concatenating, make sure all non-index-column-names are lower-case, to avoid concat going wrong

         graph_embedding__GlobalSHAP_TestDataset_df__row_of_interest.columns = \
            [ colname.lower() if colname not in index_columns else colname for colname in graph_embedding__GlobalSHAP_TestDataset_df__row_of_interest.columns ]

         no_graph__GlobalSHAP_TestDataset_df__row_of_interest.columns = \
            [colname.lower() if colname not in index_columns else colname for colname in no_graph__GlobalSHAP_TestDataset_df__row_of_interest.columns ]


         # save out
         GlobalSHAP_TestDataset_df__rows_of_interest__comparison = \
            pd.concat([no_graph__GlobalSHAP_TestDataset_df__row_of_interest, graph_embedding__GlobalSHAP_TestDataset_df__row_of_interest])
         
         GlobalSHAP_TestDataset_df__rows_of_interest__comparison.set_index( index_columns, inplace = True)

         GlobalSHAP_TestDataset_df__rows_of_interest__comparison.to_csv(os.path.join(explanation_comparision_savedir, "GlobalSHAP_TestDataset_dfs__Row__Comparison.csv"))

         # TODO: histogram comparison w.r.t feature-values as well? -----------------------------------------------------------------------------

         no_graph__feature_columns = [ col for col in no_graph__GlobalSHAP_TestDataset_df__row_of_interest.columns if col not in index_columns + ["SUM"] ]
         plt.figure(figsize=(20, 10)) 
         plt.bar(no_graph__GlobalSHAP_TestDataset_df__row_of_interest[no_graph__feature_columns].columns.tolist(), 
                 no_graph__GlobalSHAP_TestDataset_df__row_of_interest[no_graph__feature_columns].values.tolist()[0])
         plt.xticks(rotation=90, fontsize='xx-small')
         plt.title("No Graph -- Feature-value Distribution")
         plt.savefig( os.path.join( explanation_comparision_savedir,
                                   'No_Graph__Feature-value_Distribution.png') )
         plt.close()

         graph_embedding__feature_columns = [ col for col in graph_embedding__GlobalSHAP_TestDataset_df__row_of_interest.columns if col not in index_columns + ["SUM"] ]
         plt.figure(figsize=(20, 10))
         plt.bar(graph_embedding__GlobalSHAP_TestDataset_df__row_of_interest[graph_embedding__feature_columns].columns.tolist(), 
                 graph_embedding__GlobalSHAP_TestDataset_df__row_of_interest[graph_embedding__feature_columns].values.tolist()[0])
         plt.xticks(rotation=90, fontsize='xx-small')
         plt.title("Graph Embedding -- Feature-value Distribution")
         plt.savefig( os.path.join( explanation_comparision_savedir,
                                   'Graph_Embedding__Feature-value_Distribution.png') )
         plt.close()


         return 
      

      def get_waterfallplots_and_copy(
                                      sample : str, 
                                      graph_embedding__waterfall_plots__dirpath : str,
                                      no_graph__waterfall_plots__dirpath : str,

                                      explanation_comparision_savedir : str,
                                      ):

            graph_embedding__waterfall_plot_sample__fname =  [f for f in os.listdir(graph_embedding__waterfall_plots__dirpath) \
                                                              if sample in f ][0] 
            graph_embedding__waterfall_plot_sample__fpath = os.path.join(graph_embedding__waterfall_plots__dirpath, 
                                                                        graph_embedding__waterfall_plot_sample__fname)
            shutil.copy(graph_embedding__waterfall_plot_sample__fpath, 
                        os.path.join(explanation_comparision_savedir, 
                                     f"graph_embedding__{graph_embedding__waterfall_plot_sample__fname}"))


            no_graph__waterfall_plot_sample__fname = [f for f in os.listdir(no_graph__waterfall_plots__dirpath) \
                                                      if sample in f ][0] 
            no_graph__waterfall_plot_sample__fpath = os.path.join(no_graph__waterfall_plots__dirpath, 
                                                                  no_graph__waterfall_plot_sample__fname)
            shutil.copy(no_graph__waterfall_plot_sample__fpath, 
                        os.path.join(explanation_comparision_savedir, 
                                     f"no_graph__{no_graph__waterfall_plot_sample__fname}"))     
     


      # ---------------------------------------------------------------------------------------------------
      # ---- category-1 -- both_mispredicted
      for sample in predictions_comparisons__dict["mispredictions__intersecting"]: 

         explanation_comparision_savedir = os.path.join(category_1__dirpath, sample)
         os.makedirs(explanation_comparision_savedir)

         # JY @ 2023-12-28: TODO : could also saveout outputs in results_dict to explanation_comparision_savedir
         results_dict = get_sumShap_basevalue_predictprobas__and_saveout(sample, explanation_comparision_savedir )


         get_waterfallplots_and_copy(sample = sample, 
                                     graph_embedding__waterfall_plots__dirpath = materials_for_explanation_comparison__dict['graph_embedding__mispredictions__dirpath'],
                                     no_graph__waterfall_plots__dirpath = materials_for_explanation_comparison__dict['no_graph__mispredictions__dirpath'],
                                     explanation_comparision_savedir= explanation_comparision_savedir)


         # (do this?) get manually feature values, and feature shaps
         
         # TODO: WRITE EVERYTHING TO COMPARISON_RESULTS DIR FOR SMOOTH ANALYSIS


      
      # ---------------------------------------------------------------------------------------------------
      # ---- category-2 -- both_correctly_predicted
      for sample in predictions_comparisons__dict["correctpredictions_by_both"]:
         
         explanation_comparision_savedir = os.path.join(category_2__dirpath, sample)
         os.makedirs(explanation_comparision_savedir)

         results_dict = get_sumShap_basevalue_predictprobas__and_saveout(sample, explanation_comparision_savedir )

         get_waterfallplots_and_copy(sample = sample, 
                                     graph_embedding__waterfall_plots__dirpath = materials_for_explanation_comparison__dict['graph_embedding__correctpredictions__dirpath'],
                                     no_graph__waterfall_plots__dirpath = materials_for_explanation_comparison__dict['no_graph__correctpredictions__dirpath'],
                                     explanation_comparision_savedir= explanation_comparision_savedir)

      # ---------------------------------------------------------------------------------------------------
      # ---- category-3 -- only_graph_embedding_mispredicted
      for sample in predictions_comparisons__dict["mispredictions__only_in__graph_embedding"]:

         explanation_comparision_savedir = os.path.join(category_3__dirpath, sample)
         os.makedirs(explanation_comparision_savedir)

         results_dict = get_sumShap_basevalue_predictprobas__and_saveout(sample, explanation_comparision_savedir )

         get_waterfallplots_and_copy(sample = sample, 
                                     graph_embedding__waterfall_plots__dirpath = materials_for_explanation_comparison__dict['graph_embedding__mispredictions__dirpath'],
                                     no_graph__waterfall_plots__dirpath = materials_for_explanation_comparison__dict['no_graph__correctpredictions__dirpath'],
                                     explanation_comparision_savedir= explanation_comparision_savedir)

      # ---------------------------------------------------------------------------------------------------
      # ---- category-4 -- only_no_graph_mispredicted
      for sample in predictions_comparisons__dict["mispredictions__only_in__no_graph"]:

         explanation_comparision_savedir = os.path.join(category_4__dirpath, sample)
         os.makedirs(explanation_comparision_savedir)

         results_dict = get_sumShap_basevalue_predictprobas__and_saveout(sample, explanation_comparision_savedir )

         get_waterfallplots_and_copy(sample = sample, 
                                     graph_embedding__waterfall_plots__dirpath = materials_for_explanation_comparison__dict['graph_embedding__correctpredictions__dirpath'],
                                     no_graph__waterfall_plots__dirpath = materials_for_explanation_comparison__dict['no_graph__mispredictions__dirpath'],
                                     explanation_comparision_savedir= explanation_comparision_savedir)


      # ---------------------------------------------------------------------------------------------------
      return 


if __name__ == "__main__":


   # TODO: Also incorporate built-in feature-importance for RF

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


   # graph_embedding__mispredictions__dirpath = \
   #    "/data/d1/jgwak1/tabby/graph_embedding_improvement_JY_git/analyze_at_model_explainer/RESULTS/sklearn.ensemble._forest.RandomForestClassifier__Dataset-Case-2__RandomForest_best_hyperparameter_mean_case2__final_test__signal_amplified__event_1gram_nodetype_5bit__mean__2023-12-21_194828/WATERFALL_PLOTS_Local-Explanation_1gram/Mispredictions"
   # no_graph__mispredictions__dirpath = \
   #    "/data/d1/jgwak1/tabby/graph_embedding_improvement_JY_git/analyze_at_model_explainer/RESULTS/sklearn.ensemble._forest.RandomForestClassifier__Dataset-Case-2__RandomForest_best_hyperparameter_mean_case2_nograph__final_test__no_graph_structure__event_1gram_nodetype_5bit__mean__2023-12-21_194942/WATERFALL_PLOTS_Local-Explanation_1gram/Mispredictions"


   # JY @ 2024-1-4 : compare BestRFs of standard-message-passing-1gram-1hop-sumaggr-sumpool vs. baseline of 1gram, on dataset-1
   graph_embedding__mispredictions__dirpath = \
   "/home/jgwak1/tabby/graph_embedding_improvement_JY_git/graph_embedding_improvement_efforts/Trial_7__Thread_level_N_grams__N_gt_than_1__Similar_to_PriorGraphEmbedding/RESULTS/RandomForest__Full_Dataset_2_Double_Stratified__Best_RF__Full_Dataset_2_Double_Stratified__2gram__sum_pool__final_test__thread_level__N>1_grams_events__nodetype5bit__2gram__sum_pool__2024-02-07_092211/WATERFALL_PLOTS_Local-Explanation_1gram/Mispredictions"
   no_graph__mispredictions__dirpath = \
   "/home/jgwak1/tabby/graph_embedding_improvement_JY_git/graph_embedding_improvement_efforts/Baseline_Approaches/RESULTS/RandomForest__Full_Dataset_2_Double_Stratified__Best_RF__Full_Dataset_2_Double_Stratified_2gram__baseline_3__final_test__baseline_3__flattened_graph_Ngram_events__node_type_counts__2gram__2024-02-07_091545/WATERFALL_PLOTS_Local-Explanation_1gram/Mispredictions"


   predictions_comparisons_dict = predictions_comparisons(graph_embedding__mispredictions__dirpath, no_graph__mispredictions__dirpath)
   materials_for_explanation_comparison_dict = get_materials_for_explanation_comparison(graph_embedding__mispredictions__dirpath, no_graph__mispredictions__dirpath)

   produce_explanation_comparisons(predictions_comparisons__dict = predictions_comparisons_dict , 
                                   materials_for_explanation_comparison__dict = materials_for_explanation_comparison_dict)





