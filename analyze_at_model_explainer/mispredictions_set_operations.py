import os

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
