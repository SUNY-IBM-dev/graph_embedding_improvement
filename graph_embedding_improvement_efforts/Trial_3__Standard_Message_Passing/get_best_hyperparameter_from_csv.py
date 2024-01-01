import pandas as pd
import pprint

if __name__ == "__main__":
    

    up_to_row = None
    if not (type(up_to_row) == int) and not ( up_to_row == None ):
      raise ValueError("'up_to_row' should only either be integer or None")

    dataset_csvpath =\
    "/data/d1/jgwak1/tabby/graph_embedding_improvement_JY_git/graph_embedding_improvement_efforts/Trial_3__Standard_Message_Passing/RESULTS/sklearn.ensemble._forest.RandomForestClassifier__Dataset-Case-1__RandomForest_searchspace_1__10_FoldCV__search_on_train__standard_message_passing_graph_embedding__1hops__sum_aggr__sum_pool__2023-12-29_060125/sklearn.ensemble._forest.RandomForestClassifier__Dataset-Case-1__RandomForest_searchspace_1__10_FoldCV__search_on_train__standard_message_passing_graph_embedding__1hops__sum_aggr__sum_pool__2023-12-29_060125.csv"
    # -----------------------------------------------------------------------------------------------------------------------------------------------------------

    tuning_df = pd.read_csv(dataset_csvpath)
 
    if up_to_row:
         print(f"up_to_row: {up_to_row}", flush=True)
         tuning_df = tuning_df[:up_to_row + 1]
 
    if "Avg_Val_F1" in tuning_df.columns:     
       F1_colname = "Avg_Val_F1"
       Accuracy_colname = "Avg_Val_Accuracy"
    if "Final_Test_F1" in tuning_df.columns:    
       F1_colname = "Final_Test_F1"
       Accuracy_colname = "Final_Test_Accuracy"

    # get the best rows for both metrics
    best_F1_row_idx = tuning_df[F1_colname].idxmax()
    best_Acc_row_idx = tuning_df[Accuracy_colname].idxmax()

    # datas
    print(f"tuning_df: {tuning_df}", flush=True)
    print("", flush = True)
    print(f"tuning_df.loc[best_F1_row_idx][F1_colname]: {tuning_df.loc[best_F1_row_idx][F1_colname]}", flush=True)    
    print(f"tuning_df['{F1_colname}'].mean(): {tuning_df[F1_colname].mean()}", flush=True)    
    print("", flush = True)
    print(f"tuning_df.loc[best_Acc_row_idx][Accuracy_colname]: {tuning_df.loc[best_Acc_row_idx][Accuracy_colname]}", flush=True)    
    print(f"tuning_df['{Accuracy_colname}'].mean(): {tuning_df[Accuracy_colname].mean()}", flush=True)    

    # Added by JY @ 2024-1-1 : 
    # Get hyperparameters that correspond to Best-Avg-F1.

    if F1_colname == "Avg_Val_F1":
      print("\n", flush = True)
      print("Best Avg_Val_F1 Hyperparamters:\n", flush = True)

      # get names of hyperparameters 
      hyperparameter_names = [x for x in tuning_df.columns if not ("Unnamed" in x) and not ("Val" in x) ]

      best_hyperparameters_dict = dict( tuning_df.loc[best_F1_row_idx, hyperparameter_names] )

      pprint.pprint(best_hyperparameters_dict)

      print()




    # other hyperparameters