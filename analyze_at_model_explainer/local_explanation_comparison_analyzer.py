'''
TODO:

Automate the procedure for: 
"Local_Explanation_Comparisons -- strong-effect" TAB of 
"https://docs.google.com/spreadsheets/d/10011AKkOJFh2CA8W9-D3OS8VjVz9pg4dRpjRM7fGj1M/edit#gid=1324606877"


# utilize the outputted "Local SHAP Test dataset", "Global SHAP Test dataset" to achieve our goal

MAIN QUESITON
** For each approach (no_graph, graph-embedding), 
      For each feature of interest (or features), across samples, 
      what are the feature-values that are associated with 'pushing towards benign' and 'pushing towards malware'?"


      # magnitude and local shap feature rank 
      # for each node type, show the feature value association with pushing towards which class with the extent


But the limitation here is that the same feature-value might push towards different predictions 
(e.g. for sample-A, feature-A's feature-value 10 might push towards benign prediction, while for sample-B, feature-A's feature-value 10 might push towards malware prediction).
This is mainly due to interactive effects across features 
(e.g. "feature-A: 10 & feature-B: 20" could lead to benign prediction , while "feature-A: 10 & feature-B: 50" could lead to a malware prediction -- think of a decision tree's split point) and non-linearity in machine learning models
i.e. varying contributions of features based on the context of other features within the sample

THEREFORE, JUST TRY THIS BUT DONT TRUST IT TO MUCH 
--> JY @ 2024-1-11: but can we still get some tendency?


TODO -- Could count stuff (e.g. incidients of graph-embedding having positive effect for each feature)

'''

import os
import pandas as pd
from collections import defaultdict
import pprint

if __name__ == "__main__":

   # dataset-1 explanation-comparison (graph-embedding vs. no-graph)
   Explanation_comparison_dirpath = "/home/jgwak1/temp_JY/graph_embedding_improvement_JY_git/analyze_at_model_explainer/EXPLANATION_COMPARISONS/Explanation_Comparison___@_2024-01-08_145910"


   # dataset-2 explanation-comparison (graph-embedding vs. no-graph)
   # Explanation_comparison_dirpath = "/home/jgwak1/temp_JY/graph_embedding_improvement_JY_git/analyze_at_model_explainer/EXPLANATION_COMPARISONS/Explanation_Comparison___@_2024-01-08_151611"



   # None or 'list with more than 1 element' are acceptable
   features_of_interest = ["file", "thread", "registry", "network", "process"]
   samples_of_interest = None


   # JY @ 2024-1-11: More specifically, for example, if graph-embedding-local-shap-value is 0.005 and no-graph-local-SHAP-value is 0.004, 
   #                 would it be fair to consider graph-embedding had a positive effect? 
   #                 That small increase (0.001) with graph-embedding could be due to some randomness coming from RF or SHAP-computation.
   #                 Even if it is from graph-embedding, the effect is too marginal. 
   #                 - We want to focus on cases, where graph-embedding likely brought a noticeable (non-marginal) positive or negative effect.
   #                   Therefore, set a threshold. In Waterfall-point, it shows up to 2 decimal points, so 0.01 or 0.005 might be fair. 
   non_marginal_effect__local_shap_diff__Threshold = 0.01

   # JY @ 2024-1-11: similar story for 'sum-of-feature-shaps difference'
   #                 could set the difference-threshold a little higher here compared to the 'local shap difference threshold' ,
   #                 since there is cumulative-effect of local-shap-differences takes place, given this is the 'sum' of feature-shaps
   non_marginal_effect__SUM_of_feature_shaps_diff__Threshold = 0.05



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

   # -----------------------------------------------------------------------------------------------------------
   # Data structures for Further Analysis
   #


   prediction_changed_by_graph_embedding__cnt = 0

   feature_level__graph_embedding_effect__cnt__dict = {feature: defaultdict(int) for feature in features_to_check}
   sample_level__graph_embedding_effect__cnt__dict = defaultdict(int)

   # for some more info
   feature_level__graph_embedding_effect__samples__dict = {feature: {f"marginal_effect ( abs. < {non_marginal_effect__local_shap_diff__Threshold} )": [],
                                                                      "positive_effect": [],
                                                                      "negative_effect": []} 
                                                                     for feature in features_to_check}

   sample_level__graph_embedding_effect__samples__dict = {f"marginal_effect ( abs. < {non_marginal_effect__SUM_of_feature_shaps_diff__Threshold} )": [],
                                                           "positive_effect": [],
                                                           "negative_effect": []}

   # TODO @ 2024-1-11 --
   # Attempt associating feature-values to 'push-towards-benign' or 'push-towards-malware'
   # Yes, there is an issue as interaction-effect and/or model-nonlinearity can play role
   # -> But can I still get feature-value-combinations that are likely to 'push towards benign' or 'push towards malware'?
   #    look into shap functionalities (heatmap etc  --  
   #                                    https://shap-lrjball.readthedocs.io/en/latest/generated/shap.TreeExplainer.html
   #                                    https://shap-lrjball.readthedocs.io/en/latest/generated/shap.explainers.other.TreeGain.html#shap.explainers.other.TreeGain
   #                                    https://shap-lrjball.readthedocs.io/en/latest/api.html#plots
   # )




   # TODO -- This is an experiment; BE CAUTIOUS  cautious in interpretation. -- local-shap-and its extent
   no_graph__feature_value_to_pushes_towards_direction = {feature: {f"push to Neither": [],
                                                                      "pushes toward Benign": [],  # (data-name, shap, rank) -- when sorting use absolute value of local-shap
                                                                                                   #  for easier sorting? 
                                                                      "pushes toward Malware": []} 
                                                                     for feature in features_to_check}
   
   graph_embedding__feature_value_to_pushes_towards_direction = {feature: {f"push to Neither": [],
                                                                      "pushes toward Benign": [],  #  when sorting use absolute value of local-shap
                                                                                                   #  for easier sorting? 
                                                                      "pushes toward Malware": []} 
                                                                     for feature in features_to_check}

   # -----------------------------------------------------------------------------------------------------------
   # Now automate for analysis

   print(f"{os.path.split(Explanation_comparison_dirpath)[1]}", flush= True)
   print(f"\nfeatures_to_check:\n{features_to_check}\n\n", flush=True)
   print(f"non_marginal_effect__local_shap_diff__Threshold: {non_marginal_effect__local_shap_diff__Threshold}", flush=True)
   print(f"non_marginal_effect__SUM_of_feature_shaps_diff__Threshold: {non_marginal_effect__SUM_of_feature_shaps_diff__Threshold}", flush=True)


   for i, data_name in enumerate(data_names):
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


         # --------------------------------------------------------------------------------------------------------------------------------------------------------
         # JY @ 2024-1-11 : Experimental

         if data__no_graph__local_SHAP_value < 0:
                                                                                                # #  when sorting use absolute value of local-shap
            no_graph__feature_value_to_pushes_towards_direction[feature]["pushes toward Benign"].append( { "data_name" : data_name, 
                                                                                                           "local SHAP value": data__no_graph__local_SHAP_value,
                                                                                                           "local feature rank": data__no_graph__feature_rank,
                                                                                                           "feature value": data__no_graph__feature_value } )
               
         elif data__no_graph__local_SHAP_value > 0:

            no_graph__feature_value_to_pushes_towards_direction[feature]["pushes toward Malware"].append( { "data_name" : data_name, 
                                                                                                           "local SHAP value": data__no_graph__local_SHAP_value,
                                                                                                           "local feature rank": data__no_graph__feature_rank,
                                                                                                           "feature value": data__no_graph__feature_value } )

         else:
            no_graph__feature_value_to_pushes_towards_direction[feature]["pushes toward Neither"].append( { "data_name" : data_name, 
                                                                                                           "local SHAP value": data__no_graph__local_SHAP_value,
                                                                                                           "local feature rank": data__no_graph__feature_rank,
                                                                                                           "feature value": data__no_graph__feature_value } )



         if data__graph_embedding___local_SHAP_value < 0:
                                                                                                # #  when sorting use absolute value of local-shap
            graph_embedding__feature_value_to_pushes_towards_direction[feature]["pushes toward Benign"].append( 
                                                                                                         { "data_name" : data_name, 
                                                                                                           "local SHAP value": data__graph_embedding___local_SHAP_value,
                                                                                                           "local feature rank": data__graph_embedding__feature_rank,
                                                                                                           "feature value": data__graph_embedding___feature_value } )
               
         elif data__graph_embedding___local_SHAP_value > 0:

            graph_embedding__feature_value_to_pushes_towards_direction[feature]["pushes toward Malware"].append( { "data_name" : data_name, 
                                                                                                           "local SHAP value": data__graph_embedding___local_SHAP_value,
                                                                                                           "local feature rank": data__graph_embedding__feature_rank,
                                                                                                           "feature value": data__graph_embedding___feature_value } )

         else:
            graph_embedding__feature_value_to_pushes_towards_direction[feature]["pushes toward Neither"].append( { "data_name" : data_name, 
                                                                                                           "local SHAP value": data__graph_embedding___local_SHAP_value,
                                                                                                           "local feature rank": data__graph_embedding__feature_rank,
                                                                                                           "feature value": data__graph_embedding___feature_value } )


         

         # --------------------------------------------------------------------------------------------------------------------------------------------------------

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


         local_shap__abs_diff = abs(data__no_graph__local_SHAP_value - data__graph_embedding___local_SHAP_value)

         if local_shap__abs_diff < non_marginal_effect__local_shap_diff__Threshold:
            print(f"--> 'graph-embedding' had marginal effect ( abs. < {non_marginal_effect__local_shap_diff__Threshold} local-shap-difference ) with '{feature}'.")
            feature_level__graph_embedding_effect__cnt__dict[feature][f"marginal_effect ( abs. < {non_marginal_effect__local_shap_diff__Threshold} )"] += 1
            feature_level__graph_embedding_effect__samples__dict[feature][f"marginal_effect ( abs. < {non_marginal_effect__local_shap_diff__Threshold} )"].append( (data_name, local_shap__abs_diff) )

         else: # non-marginal (noticeable) effect

            if label == "benign": 
   
               if data__no_graph__local_SHAP_value < data__graph_embedding___local_SHAP_value:
                  print(f"--> 'graph-embedding' had positive-effect with '{feature}', b/c more pushes towards 'Benign'", flush=True)
                  feature_level__graph_embedding_effect__cnt__dict[feature]['positive_effect'] += 1
                  feature_level__graph_embedding_effect__samples__dict[feature]['positive_effect'].append( (data_name, local_shap__abs_diff) )


               elif data__no_graph__local_SHAP_value > data__graph_embedding___local_SHAP_value :
                  print(f"--> 'graph-embedding' had negative-effect with '{feature}', b/c less pushes towards 'Benign'", flush=True)
                  feature_level__graph_embedding_effect__cnt__dict[feature]['negative_effect'] += 1
                  feature_level__graph_embedding_effect__samples__dict[feature]['negative_effect'].append( (data_name, local_shap__abs_diff) )

            else: # if label == malware

             if data__graph_embedding___local_SHAP_value > data__no_graph__local_SHAP_value:
                print(f"--> 'graph-embedding' had positive-effect with '{feature}', b/c it more pushes towards 'Malware'", flush=True)
                feature_level__graph_embedding_effect__cnt__dict[feature]['positive_effect'] += 1
                feature_level__graph_embedding_effect__samples__dict[feature]['positive_effect'].append( (data_name, local_shap__abs_diff) )
 


             elif data__graph_embedding___local_SHAP_value < data__no_graph__local_SHAP_value :
                print(f"--> 'graph-embedding' had negative-effect with '{feature}', b/c it less pushes towards 'Malware'", flush=True)
                feature_level__graph_embedding_effect__cnt__dict[feature]['negative_effect'] += 1
                feature_level__graph_embedding_effect__samples__dict[feature]['negative_effect'].append( (data_name, local_shap__abs_diff) )




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


      sum_of_feature_shaps__abs_diff = abs(no_graph__data__SHAP_sum_of_feature_shaps - graph_embedding__data__SHAP_sum_of_feature_shaps)

      if sum_of_feature_shaps__abs_diff < non_marginal_effect__SUM_of_feature_shaps_diff__Threshold:
         print(f"--> 'graph-embedding' had marginal-effect ( abs. < {non_marginal_effect__SUM_of_feature_shaps_diff__Threshold} sum-of-feature-shaps difference ), at sample-level.")
         sample_level__graph_embedding_effect__cnt__dict[f"marginal_effect ( abs. < {non_marginal_effect__SUM_of_feature_shaps_diff__Threshold} )"] += 1
         sample_level__graph_embedding_effect__samples__dict[f"marginal_effect ( abs. < {non_marginal_effect__SUM_of_feature_shaps_diff__Threshold} )"].append( (data_name, sum_of_feature_shaps__abs_diff) )


      else: # non-marginal (noticeable) effect

         if label == "benign": 

            if no_graph__data__SHAP_sum_of_feature_shaps < graph_embedding__data__SHAP_sum_of_feature_shaps:
               print(f"--> 'graph-embedding' had positive-effect, at sample-level, b/c more pushes towards 'Benign'", flush=True)
               sample_level__graph_embedding_effect__cnt__dict['positive_effect'] += 1
               sample_level__graph_embedding_effect__samples__dict['positive_effect'].append( (data_name, sum_of_feature_shaps__abs_diff)  )

            elif no_graph__data__SHAP_sum_of_feature_shaps > graph_embedding__data__SHAP_sum_of_feature_shaps :
               print(f"--> 'graph-embedding' had negative-effect, at sample-level, b/c less pushes towards 'Benign'", flush=True)
               sample_level__graph_embedding_effect__cnt__dict['negative_effect'] += 1
               sample_level__graph_embedding_effect__samples__dict['negative_effect'].append( (data_name, sum_of_feature_shaps__abs_diff) )

         else: # if malware

            if graph_embedding__data__SHAP_sum_of_feature_shaps > no_graph__data__SHAP_sum_of_feature_shaps:
               print(f"--> 'graph-embedding' had positive-effect, at sample-level, b/c more pushes towards 'Malware'", flush=True)
               sample_level__graph_embedding_effect__cnt__dict['positive_effect'] += 1
               sample_level__graph_embedding_effect__samples__dict['positive_effect'].append( (data_name, sum_of_feature_shaps__abs_diff) )

            elif graph_embedding__data__SHAP_sum_of_feature_shaps < no_graph__data__SHAP_sum_of_feature_shaps :
               print(f"--> 'graph-embedding' had negative-effect, at sample-level, b/c less pushes towards 'Malware'", flush=True)
               sample_level__graph_embedding_effect__cnt__dict['negative_effect'] += 1
               sample_level__graph_embedding_effect__samples__dict['negative_effect'].append( (data_name, sum_of_feature_shaps__abs_diff) )


      print(f"Prediction-Changed? {no_graph__prediction != graph_embedding__prediction}", flush=True)
      if no_graph__prediction != graph_embedding__prediction:   
         prediction_changed_by_graph_embedding__cnt += 1

      print("-"*50, flush = True)

   # JY @ 2024-1-10
   # BUT NEED TO BE CAREFUL HERE, IS IT TRIVIALLY POSITIVE OR REALLY POSITIVE
   prediction_changed_by_graph_embedding__cnt

   feature_level__graph_embedding_effect__cnt__dict
   sample_level__graph_embedding_effect__cnt__dict
   

   feature_level__graph_embedding_effect__samples__dict__sorted = {feature: {effect_category: sorted(list_of_tuples, key=lambda x: x[1], reverse=True) 
                                                                             for effect_category, list_of_tuples in effect_dict.items()} 
                                                                   for feature, effect_dict in feature_level__graph_embedding_effect__samples__dict.items()}
                                                         
   sample_level__graph_embedding_effect__samples__dict__sorted = { k: sorted(v, key = lambda x: x[1], reverse=True) for k,v in sample_level__graph_embedding_effect__samples__dict.items() }
   

   print("="*150, flush=True)
   print("="*150, flush=True)
   print("\n[ Additional info ]:\n")

   print(f"prediction_changed_by_graph_embedding__cnt: {prediction_changed_by_graph_embedding__cnt}\n", flush=True)
   
   print(f"feature_level__graph_embedding_effect__cnt__dict:\n", flush=True)
   pprint.pprint(feature_level__graph_embedding_effect__cnt__dict, indent=2)
   print("\n")

   print(f"sample_level__graph_embedding_effect__cnt__dict:\n", flush=True)
   pprint.pprint(sample_level__graph_embedding_effect__cnt__dict, indent=2)
   print("\n")

   print(f"feature_level__graph_embedding_effect__samples__dict__sorted:\n", flush=True)
   pprint.pprint(feature_level__graph_embedding_effect__samples__dict__sorted, indent=2)
   print("\n")

   print(f"sample_level__graph_embedding_effect__samples__dict__sorted:\n", flush=True)
   pprint.pprint(sample_level__graph_embedding_effect__samples__dict__sorted, indent=2)
   print("\n")

   # experimental : be cautious

   # maybe could plot it?
   print(f"\n** FOR THE FOLLOWING RESULTS, BE CONSERVATIVE IN MAKING ANY CONCLUSIONS!\n(This is mapping of a 'single feature-value' to its 'pushing direction and extent' (expressed by it's local SHAP value); THIS DOES NOT TAKE INTO ACCOUNT the potential effects of feature-interactions or model(RF) non-linearity)", flush = True)
   print("--> Could extend/incorporate to 'shap_interaction_values' which TreeSHAP provides (e.g. can I get feature-value combinations with strongest joint-effect on pushing towards positive/negative? )\n", flush=True)

   no_graph__feature_value_to_pushes_towards_direction__sorted_by_localshsap = {feature: {direction_category: sorted(direction_dict, key=lambda x: abs(x['local SHAP value']), 
                                                                                                              reverse=True) 
                                                                               for direction_category, direction_dict in direction_dict.items()} 
                                                                               for feature, direction_dict in no_graph__feature_value_to_pushes_towards_direction.items()}


   graph_embedding__feature_value_to_pushes_towards_direction__sorted_by_localshsap = {feature: {direction_category: sorted(direction_dict, key=lambda x: abs(x['local SHAP value']), 
                                                                                                              reverse=True) 
                                                                                       for direction_category, direction_dict in direction_dict.items()} 
                                                                                       for feature, direction_dict in graph_embedding__feature_value_to_pushes_towards_direction.items()}


   no_graph__feature_value_to_pushes_towards_direction__sorted_by_local_rank = {feature: {direction_category: sorted(direction_dict, key=lambda x: x["local feature rank"])                                                                                                               
                                                                                          for direction_category, direction_dict in direction_dict.items()} 
                                                                                          for feature, direction_dict in no_graph__feature_value_to_pushes_towards_direction.items()}


   graph_embedding__feature_value_to_pushes_towards_direction__sorted_by_local_rank = {feature:  {direction_category: sorted(direction_dict, key=lambda x: x["local feature rank"])  
                                                                                       for direction_category, direction_dict in direction_dict.items()} 
                                                                                       for feature, direction_dict in graph_embedding__feature_value_to_pushes_towards_direction.items()}



   print(f"no_graph__feature_value_to_pushes_towards_direction__sorted_by_local_rank:\n", flush=True)
   pprint.pprint(no_graph__feature_value_to_pushes_towards_direction__sorted_by_local_rank, indent=2)
   print("\n")


   print(f"graph_embedding__feature_value_to_pushes_towards_direction__sorted_by_local_rank:\n", flush=True)
   pprint.pprint(graph_embedding__feature_value_to_pushes_towards_direction__sorted_by_local_rank, indent=2)
   print("\n")   

   print()