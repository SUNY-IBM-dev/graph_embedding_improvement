import pandas as pd

if __name__ == "__main__":
    
   # -----------------------------------------------------------------------------------------------------------------------------------------------

   # [Dataset-1 -- XGB--search-on-train , event-1gram] -- running
   #  up_to_row = 6562
   #  treatment_csvpath = "/home/jgwak1/SUNYIBM/gnn_v1_615/option_3_StratKfoldCV/sklearn.ensemble._gb.GradientBoostingClassifier__Dataset_1_B148_M148__XGBoost_searchspace_1__10_FoldCV__search_on_train__signal_amplified__event_1gram__max__2023-08-08_092817.csv"
   #  no_treatment_csvpath = "/home/jgwak1/SUNYIBM/gnn_v1_615/option_3_StratKfoldCV/sklearn.ensemble._gb.GradientBoostingClassifier__Dataset_1_B148_M148__XGBoost_searchspace_1__10_FoldCV__search_on_train__no_graph_structure__event_1gram__max__2023-08-08_092740.csv"


   # -----------------------------------------------------------------------------------------------------------------------------------------------

   # [Dataset-1 -- XGB--search-on-all , event-1gram] 
   #  up_to_row = 6562
   #  treatment_csvpath = "/home/jgwak1/SUNYIBM/gnn_v1_615/option_3_StratKfoldCV/sklearn.ensemble._gb.GradientBoostingClassifier__Dataset_1_B148_M148__XGBoost_searchspace_1__10_FoldCV__search_on_all__signal_amplified__event_1gram__max__2023-08-07_173505.csv"
   #  no_treatment_csvpath = "/home/jgwak1/SUNYIBM/gnn_v1_615/option_3_StratKfoldCV/sklearn.ensemble._gb.GradientBoostingClassifier__Dataset_1_B148_M148__XGBoost_searchspace_1__10_FoldCV__search_on_all__no_graph_structure__event_1gram__max__2023-08-07_173520.csv"


   # -----------------------------------------------------------------------------------------------------------------------------------------------

   # [Dataset-2 -- XGB--search-on-train , event-1gram] -- running
   #  up_to_row = 6562
   #  treatment_csvpath = "/home/jgwak1/SUNYIBM/gnn_v1_615/option_3_StratKfoldCV/sklearn.ensemble._gb.GradientBoostingClassifier__Dataset_2_B239_M239__XGBoost_searchspace_1__10_FoldCV__search_on_train__signal_amplified__event_1gram__max__2023-08-08_033236.csv"
   #  no_treatment_csvpath = "/home/jgwak1/SUNYIBM/gnn_v1_615/option_3_StratKfoldCV/sklearn.ensemble._gb.GradientBoostingClassifier__Dataset_2_B239_M239__XGBoost_searchspace_1__10_FoldCV__search_on_train__no_graph_structure__event_1gram__max__2023-08-08_024849.csv"

   # -----------------------------------------------------------------------------------------------------------------------------------------------
   # [Dataset-2 -- XGB--search-on-all , event-1gram] 

   #  up_to_row = 6562
   #  treatment_csvpath = "/home/jgwak1/SUNYIBM/gnn_v1_615/option_3_StratKfoldCV/sklearn.ensemble._gb.GradientBoostingClassifier__Dataset_2_B239_M239__XGBoost_searchspace_1__10_FoldCV__search_on_all__signal_amplified__event_1gram__max__2023-08-07_173554.csv"
   #  no_treatment_csvpath = "/home/jgwak1/SUNYIBM/gnn_v1_615/option_3_StratKfoldCV/sklearn.ensemble._gb.GradientBoostingClassifier__Dataset_2_B239_M239__XGBoost_searchspace_1__10_FoldCV__search_on_all__no_graph_structure__event_1gram__max__2023-08-07_173612.csv"


   # -----------------------------------------------------------------------------------------------------------------------------------------------


   # [Dataset-1 -- XGB--search-on-all , event-1gram + 5-bit-node-type + 41-bit-adhoc-node-pattern ]
   #  up_to_row = 6562
   #  treatment_csvpath = "/home/jgwak1/SUNYIBM/gnn_v1_615/option_3_StratKfoldCV/sklearn.ensemble._gb.GradientBoostingClassifier__Dataset_1_B148_M148__XGBoost_searchspace_1__10_FoldCV__search_on_all__signal_amplified__event_1gram_nodetype_5bit_and_Ahoc_Identifier__max__2023-08-08_160147.csv"
   #  no_treatment_csvpath = "/home/jgwak1/SUNYIBM/gnn_v1_615/option_3_StratKfoldCV/sklearn.ensemble._gb.GradientBoostingClassifier__Dataset_1_B148_M148__XGBoost_searchspace_1__10_FoldCV__search_on_all__no_graph_structure__event_1gram_nodetype_5bit_and_Ahoc_Identifier__max__2023-08-08_160216.csv"


   # [Dataset-2 -- XGB--search-on-train , event-1gram + 5-bit-node-type + 41-bit-adhoc-node-pattern ]
   #  up_to_row = 6562
   #  treatment_csvpath = "/home/jgwak1/SUNYIBM/gnn_v1_615/option_3_StratKfoldCV/sklearn.ensemble._gb.GradientBoostingClassifier__Dataset_2_B239_M239__XGBoost_searchspace_1__10_FoldCV__search_on_train__signal_amplified__event_1gram_nodetype_5bit_and_Ahoc_Identifier__max__2023-08-08_191552.csv"
   #  no_treatment_csvpath = "/home/jgwak1/SUNYIBM/gnn_v1_615/option_3_StratKfoldCV/sklearn.ensemble._gb.GradientBoostingClassifier__Dataset_2_B239_M239__XGBoost_searchspace_1__10_FoldCV__search_on_train__no_graph_structure__event_1gram_nodetype_5bit_and_Ahoc_Identifier__max__2023-08-08_191523.csv"
 
   # [Dataset-1 -- XGB--search-on-train , event-1gram + 5-bit-node-type + 41-bit-adhoc-node-pattern ] IS ON "OCELOT" 
   # [Dataset-2 -- XGB--search-on-all , event-1gram + 5-bit-node-type + 41-bit-adhoc-node-pattern ] IS ON "OCELOT"


  # -----------------------------------------------------------------------------------------------------------------------------------------------



   # [Dataset-2 -- RF--search-on-all , event-1gram + 5-bit-node-type ]
   #  up_to_row = 6913
   #  treatment_csvpath = "/home/jgwak1/SUNYIBM/gnn_v1_615/option_3_StratKfoldCV/sklearn.ensemble._forest.RandomForestClassifier__Dataset_2_B239_M239__RandomForest_searchspace_1__10_FoldCV__search_on_all__signal_amplified__event_1gram_nodetype_5bit__max__2023-08-11_001756.csv"
   #  no_treatment_csvpath = "/home/jgwak1/SUNYIBM/gnn_v1_615/option_3_StratKfoldCV/sklearn.ensemble._forest.RandomForestClassifier__Dataset_2_B239_M239__RandomForest_searchspace_1__10_FoldCV__search_on_all__no_graph_structure__event_1gram_nodetype_5bit__max__2023-08-11_001815.csv"

   # [Dataset-2 -- RF--search-on-train , event-1gram + 5-bit-node-type ]

    up_to_row = 6913
    treatment_csvpath = "/home/jgwak1/SUNYIBM/gnn_v1_615/option_3_StratKfoldCV/sklearn.ensemble._forest.RandomForestClassifier__Dataset_2_B239_M239__RandomForest_searchspace_1__10_FoldCV__search_on_train__signal_amplified__event_1gram_nodetype_5bit__max__2023-08-11_001710.csv"
    no_treatment_csvpath = "/home/jgwak1/SUNYIBM/gnn_v1_615/option_3_StratKfoldCV/sklearn.ensemble._forest.RandomForestClassifier__Dataset_2_B239_M239__RandomForest_searchspace_1__10_FoldCV__search_on_train__no_graph_structure__event_1gram_nodetype_5bit__max__2023-08-11_001730.csv"
 



                                                #  RF - Dataset-1 -- search-on-all
                                                #  up_to_row = 973
                                                #  treatment_csvpath = "/data/d1/jgwak1/stratkfold/sklearn.ensemble._forest.RandomForestClassifier__RandomForest_searchspace_1__10_FoldCV__search_on_all__signal_amplified__event_1gram_nodetype_5bit__max__2023-07-27_010658.csv"
                                                #  no_treatment_csvpath = "/data/d1/jgwak1/stratkfold/sklearn.ensemble._forest.RandomForestClassifier__RandomForest_searchspace_1__10_FoldCV__search_on_all__no_graph_structure__event_1gram_nodetype_5bit__max__2023-07-27_010646.csv"

                                                #  RF - Dataset-2 -- search-on-all
                                                #  up_to_row = 973
                                                #  treatment_csvpath = "/data/d1/jgwak1/stratkfold/sklearn.ensemble._forest.RandomForestClassifier__Dataset_2_B239_M239__RandomForest_searchspace_1__10_FoldCV__search_on_all__signal_amplified__event_1gram_nodetype_5bit__max__2023-07-29_002453.csv"
                                                #  no_treatment_csvpath = "/data/d1/jgwak1/stratkfold/sklearn.ensemble._forest.RandomForestClassifier__Dataset_2_B239_M239__RandomForest_searchspace_1__10_FoldCV__search_on_all__no_graph_structure__event_1gram_nodetype_5bit__max__2023-07-29_002526.csv"


    # -----------------------------------------------------------------------------------------------------------------------------------------------------------

    treatment_df = pd.read_csv(treatment_csvpath)
    no_treatment_df = pd.read_csv(no_treatment_csvpath)    

    if up_to_row:
         print(f"up_to_row: {up_to_row}", flush=True)
         treatment_df = treatment_df[:up_to_row + 1]
         no_treatment_df = no_treatment_df[:up_to_row + 1]

    if "Avg_Val_F1" in treatment_df.columns and "Avg_Val_Accuracy" in no_treatment_df.columns:
      
       F1_colname = "Avg_Val_F1"
       Accuracy_colname = "Avg_Val_Accuracy"
    
    if "Final_Test_F1" in treatment_df.columns and "Final_Test_Accuracy" in no_treatment_df.columns:    

       F1_colname = "Final_Test_F1"
       Accuracy_colname = "Final_Test_Accuracy"

    # datas
    print(f"treatment_csvpath: {treatment_csvpath}", flush=True)
    print(f"no_treatment_csvpath: {no_treatment_csvpath}", flush=True)
    print("")

    print(f"no_treatment_df['{F1_colname}'].max(): {no_treatment_df[F1_colname].max()}", flush=True)
    print(f"treatment_df['{F1_colname}'].max(): {treatment_df[F1_colname].max()}", flush=True)    
    print(f"no_treatment_df['{F1_colname}'].mean(): {no_treatment_df[F1_colname].mean()}", flush=True)
    print(f"treatment_df['{F1_colname}'].mean(): {treatment_df[F1_colname].mean()}", flush=True)    
    print("")
    # Accuracy
    print(f"no_treatment_df['{Accuracy_colname}'].max(): {no_treatment_df[Accuracy_colname].max()}", flush=True)
    print(f"treatment_df['{Accuracy_colname}'].max(): {treatment_df[Accuracy_colname].max()}", flush=True)    
    print(f"no_treatment_df['{Accuracy_colname}'].mean(): {no_treatment_df[Accuracy_colname].mean()}", flush=True)
    print(f"treatment_df['{Accuracy_colname}'].mean(): {treatment_df[Accuracy_colname].mean()}", flush=True)    
