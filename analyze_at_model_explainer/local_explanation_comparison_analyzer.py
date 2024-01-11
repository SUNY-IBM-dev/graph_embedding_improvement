'''
TODO:

Automate the procedure for: 
"Local_Explanation_Comparisons -- strong-effect" TAB of 
"https://docs.google.com/spreadsheets/d/10011AKkOJFh2CA8W9-D3OS8VjVz9pg4dRpjRM7fGj1M/edit#gid=1324606877"


# utilize the outputted "Local SHAP Test dataset", "Global SHAP Test dataset" to achieve our goal

MAIN QUESITON
** For each approach (no_graph, graph-embedding), 
      For each feature of interest, across samples, 
      what are the feature-values that are associated with 'pushing towards benign' and 'pushing towards malware'?"


      # magnitude and local shap feature rank 
      # for each node type, show the feature value association with pushing towards which class with the extent


But the limitation here is that the same feature-value might push towards different predictions 
(e.g. for sample-A, feature-A's feature-value 10 might push towards benign prediction, while for sample-B, feature-A's feature-value 10 might push towards malware prediction).
This is mainly due to interactive effects across features 
(e.g. "feature-A: 10 & feature-B: 20" could lead to benign prediction , while "feature-A: 10 & feature-B: 50" could lead to a malware prediction -- think of a decision tree's split point) and non-linearity in machine learning models
i.e. varying contributions of features based on the context of other features within the sample

THEREFORE, JUST TRY THIS BUT DONT TRUST IT TO MUCH 



TODO -- Could count stuff (e.g. incidients of graph-embedding having positive effect for each feature)

'''

import os
import pandas as pd
from collections import defaultdict

if __name__ == "__main__":

   # dataset-1 explanation-comparison (graph-embedding vs. no-graph)
   Explanation_comparison_dirpath = "/home/jgwak1/temp_JY/graph_embedding_improvement_JY_git/analyze_at_model_explainer/EXPLANATION_COMPARISONS/Explanation_Comparison___@_2024-01-08_145910"


   # dataset-2 explanation-comparison (graph-embedding vs. no-graph)
   # Explanation_comparison_dirpath = "/home/jgwak1/temp_JY/graph_embedding_improvement_JY_git/analyze_at_model_explainer/EXPLANATION_COMPARISONS/Explanation_Comparison___@_2024-01-08_151611"

   print(f"{os.path.split(Explanation_comparison_dirpath)[1]}", flush= True)


   # None or 'list with more than 1 element' are acceptable
   features_of_interest = ["file", "thread", "registry", "network", "process"]
   samples_of_interest = None


   # -------------------------------------------------------------
   # Prepare materials
   no_graph__GlobalSHAP_TestDataset = pd.read_csv( os.path.join(Explanation_comparison_dirpath, "no_graph__GlobalSHAP_TestDataset_csv__df.csv") )
   no_graph__Local_SHAP_vals_TestDataset = pd.read_csv( os.path.join(Explanation_comparison_dirpath, "no_graph__LocalSHAP_values_TestDataset_csv__df.csv") )

   graph_embedding__GlobalSHAP_TestDataset = pd.read_csv( os.path.join(Explanation_comparison_dirpath, "graph_embedding__GlobalSHAP_TestDataset_df.csv") )
   graph_embedding__Local_SHAP_vals_TestDataset = pd.read_csv( os.path.join(Explanation_comparison_dirpath, "graph_embedding__LocalSHAP_values_TestDataset_df.csv") )
   # -------------------------------------------------------------

   # check stuff

   assert set(no_graph__GlobalSHAP_TestDataset.columns) == set(graph_embedding__GlobalSHAP_TestDataset.columns), "Discrepancy in columns; Need examination"  
   assert set(no_graph__Local_SHAP_vals_TestDataset.columns) == set(graph_embedding__Local_SHAP_vals_TestDataset.columns), "Discrepancy in columns; Need examination" 

   assert set(no_graph__GlobalSHAP_TestDataset['data_name']) == set(no_graph__Local_SHAP_vals_TestDataset['data_name']) ==\
          set(graph_embedding__GlobalSHAP_TestDataset['data_name']) == set(graph_embedding__Local_SHAP_vals_TestDataset['data_name']), "Discrepancy in data_names; Need examination"                                                          


   assert ( features_of_interest == None ) or ( type(features_of_interest) == list and len(features_of_interest) > 0 ), f"'features_of_interest' should be either a list with at least one feature or None"
   assert ( samples_of_interest == None ) or ( type(samples_of_interest) == list and len(samples_of_interest) > 0 ),f"'samples_of_interest' should be either a list with at least one feature or None"

   non_feature_columns = ['Unnamed: 0', 'data_name', 'SUM', 'predict_proba', 'SHAP_sum_of_feature_shaps', 'SHAP_base_value']
   features_to_check = list( filter(lambda x: x not in non_feature_columns, no_graph__GlobalSHAP_TestDataset.columns) )
   if features_of_interest != None:
      assert set(features_of_interest).issubset(set(no_graph__GlobalSHAP_TestDataset.columns)) and\
             set(features_of_interest).issubset(set(no_graph__Local_SHAP_vals_TestDataset.columns)) and\
             set(features_of_interest).issubset(set(graph_embedding__GlobalSHAP_TestDataset.columns)) and\
             set(features_of_interest).issubset(set(graph_embedding__Local_SHAP_vals_TestDataset.columns)),\
             "Invalid 'features_of_interest' values"
      features_to_check = features_of_interest

   data_names = no_graph__GlobalSHAP_TestDataset['data_name']
   if samples_of_interest != None:
      assert set(samples_of_interest).issubset(set(data_names)),\
             "Invalid 'samples of interest': 'samples_of_interest' must be a subset of 'data_names'"
      data_names = samples_of_interest


   # -------------------------------------------------------------
   no_graph__GlobalSHAP_TestDataset.drop('Unnamed: 0', axis = 1, inplace = True)
   graph_embedding__GlobalSHAP_TestDataset.drop('Unnamed: 0', axis = 1, inplace = True)
   no_graph__Local_SHAP_vals_TestDataset.drop('Unnamed: 0', axis = 1, inplace = True)
   graph_embedding__Local_SHAP_vals_TestDataset.drop('Unnamed: 0', axis = 1, inplace = True)

   no_graph__GlobalSHAP_TestDataset.set_index('data_name', inplace = True)
   graph_embedding__GlobalSHAP_TestDataset.set_index('data_name', inplace = True)
   no_graph__Local_SHAP_vals_TestDataset.set_index('data_name', inplace = True)
   graph_embedding__Local_SHAP_vals_TestDataset.set_index('data_name', inplace = True)

   print(f"\nfeatures_to_check:\n{features_to_check}\n\n", flush=True)
   # -----------------------------------------------------------------------------------------------------------
   # BUT NEED TO BE CAREFUL HERE, IS IT TRIVIALLY POSITIVE OR REALLY POSITIVE
   feature_level__graph_embedding_effect__cnt__dict = {feature: defaultdict(int) for feature in features_to_check}

   sample_level__graph_embedding_effect__cnt__dict = defaultdict(int)

   prediction_changed_by_graph_embedding__cnt = 0

   # -----------------------------------------------------------------------------------------------------------
   # Now automate for analysis

   for i,data_name in enumerate(data_names):
      print("-"*50, flush = True)
      print(f"\n {i+1}. {data_name}\n", flush = True)
      
      for feature in features_to_check:

         # should I also incorporate base-value?
         
         # -- first get label
         if 'benign' in data_name: 
            label = "benign"
         else:
            label = "malware"

         # print(f"feature: {feature} with label {label}", flush = True)


         # get local shap values
         data__no_graph__local_SHAP_value = no_graph__Local_SHAP_vals_TestDataset.loc[data_name, feature]
         data__graph_embedding___local_SHAP_value = graph_embedding__Local_SHAP_vals_TestDataset.loc[data_name, feature]

         # get feature-values
         data__no_graph__feature_value = no_graph__GlobalSHAP_TestDataset.loc[data_name, feature]
         data__graph_embedding___feature_value = graph_embedding__GlobalSHAP_TestDataset.loc[data_name, feature]

         # get rank of this feature in terms of local-shap for the prediction (note rank is determined by pure magnitude, not necessarily towards correct direction)
         # -- confirmed that following feature-ranks and intermediate data matches that of waterfall-plots
         data__no_graph__sorted_absolute_local_shaps__descending_order = no_graph__Local_SHAP_vals_TestDataset.loc[data_name].abs().sort_values(ascending = False)
         data__graph_embedding__sorted_absolute_local_shaps__descending_order = graph_embedding__Local_SHAP_vals_TestDataset.loc[data_name].abs().sort_values(ascending = False)
         
         data__no_graph__sorted_absolute_local_shaps__descending_order__rankings = data__no_graph__sorted_absolute_local_shaps__descending_order.index
         data__graph_embedding__sorted_absolute_local_shaps__descending_order__rankings = data__graph_embedding__sorted_absolute_local_shaps__descending_order.index

         # ranks
         data__no_graph__feature_rank = data__no_graph__sorted_absolute_local_shaps__descending_order__rankings.get_loc(feature)+1 # +1 since starts from 0 (top1 == index0)
         data__graph_embedding__feature_rank = data__graph_embedding__sorted_absolute_local_shaps__descending_order__rankings.get_loc(feature)+1


         # check if this feature is pushing toward the correct direction or not 
         # -- negative local-shap-value : push towards benign (blue-left-pointing-bar) 
         # -- positive local-shap-value : push towards malware (red-right-pointing-bar) 
         no_graph__pushing_towards_correct_direction = None
         graph_embedding__pushing_towards_correct_direction = None
         
         if label == "benign": 

            if data__no_graph__local_SHAP_value < 0: 
               no_graph__pushes_towards = "Benign"
               no_graph__pushing_towards_correct_direction = "Correct"; 
         
            elif data__no_graph__local_SHAP_value > 0: 
               no_graph__pushes_towards = "Malware"
               no_graph__pushing_towards_correct_direction = "Wrong"; 
         
            else: 
               no_graph__pushing_towards_correct_direction = no_graph__pushes_towards = "Null" 


            if data__graph_embedding___local_SHAP_value < 0: 
               graph_embedding_pushes_towards = "Benign"
               graph_embedding__pushing_towards_correct_direction = "Correct"; 
         
            elif data__graph_embedding___local_SHAP_value > 0: 
               graph_embedding_pushes_towards = "Malware"
               graph_embedding__pushing_towards_correct_direction = "Wrong"; 
         
            else: 
               graph_embedding__pushing_towards_correct_direction = graph_embedding_pushes_towards = "Null"


         else: # if malware

            if data__no_graph__local_SHAP_value < 0: 
               no_graph__pushes_towards = "Benign"
               no_graph__pushing_towards_correct_direction = "Wrong"; 
            
            elif data__no_graph__local_SHAP_value > 0: 
               no_graph__pushes_towards = "Malware"
               no_graph__pushing_towards_correct_direction = "Correct";  
            
            else: 
               no_graph__pushing_towards_correct_direction = no_graph__pushes_towards = "Null"


            if data__graph_embedding___local_SHAP_value < 0: 
               graph_embedding_pushes_towards = "Benign" 
               graph_embedding__pushing_towards_correct_direction = "Wrong"
            
            elif data__graph_embedding___local_SHAP_value > 0: 
               graph_embedding_pushes_towards = "Malware"
               graph_embedding__pushing_towards_correct_direction = "Correct"
           
            else: 
               graph_embedding__pushing_towards_correct_direction = graph_embedding_pushes_towards = "Null"

      


         # ----------------------------------------------------------------------------------------------------------------------------
         # start outputting analysis -- 1 : output for each approach
         print(f"* Feature-level analysis ( {data_name} : {feature} ):")
         print(f"'no_graph'  :  '{feature}' feature with value '{data__no_graph__feature_value}'  pushes towards '{no_graph__pushes_towards}',  the '{no_graph__pushing_towards_correct_direction}' direction  with local-shap-value '{data__no_graph__local_SHAP_value}',  the 'top {data__no_graph__feature_rank}-th feature' for prediction", flush = True)         
         print(f"'graph-emb' :  '{feature}' feature with value '{data__graph_embedding___feature_value}'  pushes towards '{graph_embedding_pushes_towards}',  the '{graph_embedding__pushing_towards_correct_direction}' direction  with local-shap-value '{data__graph_embedding___local_SHAP_value}',  the 'top {data__graph_embedding__feature_rank}-th' feature for prediction", flush = True)
         


         # ----------------------------------------------------------------------------------------------------------------------------
         # start outputting analysis -- 2 : compare the two approaches
         # Figure out whether 'graph-embedding' approach had an positive/negative/no-effect towards correct-prediction
         # Compare the local-shap-values of 'no_graph' and 'graph_embedding', 
         # Basically, here write code to automate the visual 'side by side' comparison of two waterfall-plots 

         if label == "benign": 

             if data__no_graph__local_SHAP_value < data__graph_embedding___local_SHAP_value:
                print(f"--> 'graph-embedding' had positive-effect with '{feature}', b/c more pushes towards 'Benign'", flush=True)
                feature_level__graph_embedding_effect__cnt__dict[feature]['positive_effect'] += 1

             elif data__no_graph__local_SHAP_value > data__graph_embedding___local_SHAP_value :
                print(f"--> 'graph-embedding' had negative-effect with '{feature}', b/c less pushes towards 'Benign'", flush=True)
                feature_level__graph_embedding_effect__cnt__dict[feature]['negative_effect'] += 1

             else:
                print(f"--> 'graph-embedding' had ZERO-effect with '{feature}'.")
                feature_level__graph_embedding_effect__cnt__dict[feature]['zero_effect'] += 1


         else: # if malware

             if data__graph_embedding___local_SHAP_value > data__no_graph__local_SHAP_value:
                print(f"--> 'graph-embedding' had positive-effect with '{feature}', b/c it more pushes towards 'Malware'", flush=True)
                feature_level__graph_embedding_effect__cnt__dict[feature]['positive_effect'] += 1

             elif data__graph_embedding___local_SHAP_value < data__no_graph__local_SHAP_value :
                print(f"--> 'graph-embedding' had negative-effect with '{feature}', b/c it less pushes towards 'Malware'", flush=True)
                feature_level__graph_embedding_effect__cnt__dict[feature]['negative_effect'] += 1

             else:
                print(f"--> 'graph-embedding' had ZERO-effect with '{feature}'.")
                feature_level__graph_embedding_effect__cnt__dict[feature]['zero_effect'] += 1


         print("^---- CHECK the degree of effect, by comparing 'local-shap-values' and 'contribution-ranks'", flush=True)
         print("\n")

      # Also could consider sum-of-shaps, here to see overall impact of 'graph-embedding' on this sample, instead of being specific to a feature 
      # Could try to read-in the base-value here from Global-shap-test dataset
      # if sum-shap greater than base-value, predicted as malware --- higher sum-shap means more resembles malware, and vice versa
      # OVERALL

      no_graph__SHAP_base_value = list(set(no_graph__GlobalSHAP_TestDataset['SHAP_base_value']))[0] 
      graph_embedding__SHAP_base_value = list(set(graph_embedding__GlobalSHAP_TestDataset['SHAP_base_value']))[0]

      no_graph__data__SHAP_sum_of_feature_shaps = no_graph__GlobalSHAP_TestDataset.loc[data_name,'SHAP_sum_of_feature_shaps']
      graph_embedding__data__SHAP_sum_of_feature_shaps = graph_embedding__GlobalSHAP_TestDataset.loc[data_name,'SHAP_sum_of_feature_shaps']

      # If sum-of-feature-shap is greater than base-value , the prediction is "Malware" here, 
      # --- baseline is the threshold for model considering sample as 'Malware' 
      if no_graph__data__SHAP_sum_of_feature_shaps > no_graph__SHAP_base_value:
         no_graph__prediction = "Malware"
      else:
         no_graph__prediction = "Benign"

      if graph_embedding__data__SHAP_sum_of_feature_shaps > graph_embedding__SHAP_base_value:
         graph_embedding__prediction = "Malware"
      else:
         graph_embedding__prediction = "Benign"



      print(f"* Sample-level analysis ( {data_name} ):")
      print(f"'no_graph'  -->  Sum-of-Feature-Shaps : {no_graph__data__SHAP_sum_of_feature_shaps}  |  Base-value: {no_graph__SHAP_base_value}  |  Prediction:  {no_graph__prediction}  |  Label: {label}", flush = True)         
      print(f"'graph-emb' -->  Sum-of-Feature-Shaps : {graph_embedding__data__SHAP_sum_of_feature_shaps}  |  Base-value: {graph_embedding__SHAP_base_value}  |  Prediction:  {graph_embedding__prediction}  |  Label: {label}", flush = True)  

      if label == "benign": 
         if no_graph__data__SHAP_sum_of_feature_shaps < graph_embedding__data__SHAP_sum_of_feature_shaps:
            print(f"--> 'graph-embedding' had positive-effect, at sample-level, b/c more pushes towards 'Benign'", flush=True)
            sample_level__graph_embedding_effect__cnt__dict['positive_effect'] += 1

         elif no_graph__data__SHAP_sum_of_feature_shaps > graph_embedding__data__SHAP_sum_of_feature_shaps :
            print(f"--> 'graph-embedding' had negative-effect, at sample-level, b/c less pushes towards 'Benign'", flush=True)
            sample_level__graph_embedding_effect__cnt__dict['negative_effect'] += 1

         else:
            print(f"--> 'graph-embedding' had ZERO-effect, at sample-level.")
            sample_level__graph_embedding_effect__cnt__dict['zero_effect'] += 1

      else: # if malware
         if graph_embedding__data__SHAP_sum_of_feature_shaps > no_graph__data__SHAP_sum_of_feature_shaps:
            print(f"--> 'graph-embedding' had positive-effect, at sample-level, b/c more pushes towards 'Malware'", flush=True)
            sample_level__graph_embedding_effect__cnt__dict['positive_effect'] += 1

         elif graph_embedding__data__SHAP_sum_of_feature_shaps < no_graph__data__SHAP_sum_of_feature_shaps :
            print(f"--> 'graph-embedding' had negative-effect, at sample-level, b/c less pushes towards 'Malware'", flush=True)
            sample_level__graph_embedding_effect__cnt__dict['negative_effect'] += 1

         else:
            print(f"--> 'graph-embedding' had ZERO-effect, at sample-level.")
            sample_level__graph_embedding_effect__cnt__dict['zero_effect'] += 1

      print(f"Prediction-Changed? {no_graph__prediction != graph_embedding__prediction}", flush=True)
      if no_graph__prediction != graph_embedding__prediction:   
         prediction_changed_by_graph_embedding__cnt += 1

      print("-"*50, flush = True)

   # JY @ 2024-1-10
   # BUT NEED TO BE CAREFUL HERE, IS IT TRIVIALLY POSITIVE OR REALLY POSITIVE
   prediction_changed_by_graph_embedding__cnt

   sample_level__graph_embedding_effect__cnt__dict
   feature_level__graph_embedding_effect__cnt__dict

   print()










