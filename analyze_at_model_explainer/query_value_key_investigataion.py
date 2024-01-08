import os
import pandas as pd

if __name__ == "__main__":

   output_parentdirpath =\
       "/data/d1/jgwak1/tabby/graph_embedding_improvement_JY_git/analyze_at_model_explainer/EXPLANATION_COMPARISONS/Explanation_Comparison___@_2024-01-04_121751"

   # https://docs.google.com/spreadsheets/d/161t7tQQ3Rk6gUSEfVBfuR_ons6FwgNqzLIUc1cFrR3c/edit#gid=922262486

   no_graph__GlobalSHAP_TestDataset_csv__df__csvpath =\
        "/data/d1/jgwak1/tabby/graph_embedding_improvement_JY_git/analyze_at_model_explainer/EXPLANATION_COMPARISONS/Explanation_Comparison___@_2024-01-04_121751/no_graph__GlobalSHAP_TestDataset_csv__df.csv"

   graph_embedding__GlobalSHAP_TestDataset_df__csvpath = \
      "/data/d1/jgwak1/tabby/graph_embedding_improvement_JY_git/analyze_at_model_explainer/EXPLANATION_COMPARISONS/Explanation_Comparison___@_2024-01-04_121751/graph_embedding__GlobalSHAP_TestDataset_df.csv"
   
   graph_embedding__feature_to_compare__lowercase = "QueryValueKey"
   no_graph__feature_to_compare__lowercase = "queryvaluekey"


   graph_embedding__GlobalSHAP_TestDataset_df = pd.read_csv(graph_embedding__GlobalSHAP_TestDataset_df__csvpath)
   no_graph__GlobalSHAP_TestDataset_csv__df = pd.read_csv(no_graph__GlobalSHAP_TestDataset_csv__df__csvpath)


   no_graph__data_name_fix = [x.replace("SUBGRAPH_P3_","") for x in no_graph__GlobalSHAP_TestDataset_csv__df['data_name']]
   no_graph__data_name_fix = [x.replace('benign_', "") if 'malware' in x else x for x in no_graph__data_name_fix]

   no_graph__GlobalSHAP_TestDataset_csv__df['data_name'] = no_graph__data_name_fix


   graph_embedding__GlobalSHAP_TestDataset__feature_col = graph_embedding__GlobalSHAP_TestDataset_df[['data_name',graph_embedding__feature_to_compare__lowercase]]
   no_graph__GlobalSHAP_TestDataset__feature_col = no_graph__GlobalSHAP_TestDataset_csv__df[['data_name', no_graph__feature_to_compare__lowercase]]



   merged_df = pd.merge(graph_embedding__GlobalSHAP_TestDataset__feature_col, no_graph__GlobalSHAP_TestDataset__feature_col, 
                        on='data_name', how='outer')

   merged_df.to_csv(os.path.join(output_parentdirpath, f"graph_embedding_{graph_embedding__feature_to_compare__lowercase}__no_graph__{no_graph__feature_to_compare__lowercase}.csv" ))
   print()