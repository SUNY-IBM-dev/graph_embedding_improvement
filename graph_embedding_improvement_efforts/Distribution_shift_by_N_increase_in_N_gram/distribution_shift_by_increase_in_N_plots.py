import pandas as pd 
import matplotlib.pyplot as plt
import pathlib
current_fpath = pathlib.Path(__file__).resolve()



def plot_distribution_shift_histograms( Ngram_approach_choice : dict,
                                        Dataset_name : str,
                                        Tuned_on_Entire_or_Train : str):
   # if Metric not in {'Avg_Val_Accuracy', 'Avg_Val_F1'}:
   #    raise ValueError(f"metric should be either 'Avg_Val_Accuracy' or 'Avg_Val_F1'")

   color_iter = iter(['blue','red','green'])
   models = ""
   for Ngram_desc, Tuning_csvpath in Ngram_approach_choice.items():

      Tuning_df = pd.read_csv(Tuning_csvpath)
      AvgVal_Accuracies = Tuning_df['Avg_Val_Accuracy']

      color = next(color_iter)
      models += f"{Ngram_desc}_"
      plt.hist( AvgVal_Accuracies, 
                bins= 50, density=False, alpha=0.7, color= color, label= f"{Ngram_desc}(#{len(AvgVal_Accuracies)})" )
            # bins=len(Baseline_Ngram__Entire_Full_Dataset_1__AvgVal_Accuracy), density=True, alpha=0.7, color='blue', label='Histogram')

   # e.g. Dist_for__Baseline_2gram_VS_Thread_2gram__Entire_Full_Dataset_1
   plot_fname = f"Dists_for__{models}__{Dataset_name}_{Tuned_on_Entire_or_Train}" 

   plt.title(f"{Dataset_name}", fontsize=11, pad=10)
   plt.xlabel('Avg.Val.Accuracy', fontsize=11, labelpad=10)
   plt.ylabel(f'Count of Hyperparam. Sets', fontsize=11, labelpad=10)
   plt.legend(fontsize=10)
   plt.savefig(f"{current_fpath.parent}/{plot_fname}__AvgVal_Accuracy.png")
   # plt.tight_layout()
   plt.close()




   color_iter = iter(['blue','red','green'])
   models = ""
   for Ngram_desc, Tuning_csvpath in Ngram_approach_choice.items():

      Tuning_df = pd.read_csv(Tuning_csvpath)
      AvgVal_F1s = Tuning_df['Avg_Val_F1']

      color = next(color_iter)
      models += f"{Ngram_desc}_"
      plt.hist( AvgVal_F1s, 
                bins= 50, density=False, alpha=0.7, color= color, label= f"{Ngram_desc}(#{len(AvgVal_F1s)})" )
            # bins=len(Baseline_Ngram__Entire_Full_Dataset_1__AvgVal_Accuracy), density=True, alpha=0.7, color='blue', label='Histogram')

   # e.g. Dist_for__Baseline_2gram_VS_Thread_2gram__Entire_Full_Dataset_1
   plot_fname = f"Dists_for__{models}__{Dataset_name}_{Tuned_on_Entire_or_Train}" 

   plt.title(f"{Dataset_name}", fontsize=11, pad=10)
   plt.xlabel('Avg.Val.F1', fontsize=11, labelpad=10)
   plt.ylabel(f'Count of Hyperparam. Sets', fontsize=11, labelpad=10)
   plt.legend(fontsize=10)
   plt.savefig(f"{current_fpath.parent}/{plot_fname}__AvgVal_F1.png")
 
   
   plt.close()



if __name__ == "__main__":

   # Dataset_1 1-gram ( tuning complete & plots created )    
   Baseline_1gram__Entire_Full_Dataset_1_Tuning_csvpath =\
      "/home/jgwak1/tabby/graph_embedding_improvement_JY_git/graph_embedding_improvement_efforts/Baseline_Approaches/RESULTS/RandomForest__Full_Dataset_1_NoTraceUIDupdated__RandomForest_searchspace_1__10_FoldCV__search_on_all__baseline_3__flattened_graph_Ngram_events__node_type_counts__1gram__2024-02-05_000556/RandomForest__Full_Dataset_1_NoTraceUIDupdated__RandomForest_searchspace_1__10_FoldCV__search_on_all__baseline_3__flattened_graph_Ngram_events__node_type_counts__1gram__2024-02-05_000556.csv"
   Thread_1gram__Entire_Full_Dataset_1_Tuning_csvpath = \
      "/home/jgwak1/tabby/graph_embedding_improvement_JY_git/graph_embedding_improvement_efforts/Trial_7__Thread_level_N_grams__N_gt_than_1__Similar_to_PriorGraphEmbedding/RESULTS/RandomForest__Full_Dataset_1_NoTraceUIDupdated__RandomForest_searchspace_1__10_FoldCV__search_on_all__thread_level__N>1_grams_events__nodetype5bit__1gram__sum_pool__only_train_specified_Ngram_True__2024-02-05_000813/RandomForest__Full_Dataset_1_NoTraceUIDupdated__RandomForest_searchspace_1__10_FoldCV__search_on_all__thread_level__N>1_grams_events__nodetype5bit__1gram__sum_pool__only_train_specified_Ngram_True__2024-02-05_000813.csv"

   # Dataset_1 2-gram ( tuning complete & plots created ) 
   Baseline_2gram__Entire_Full_Dataset_1_Tuning_csvpath = \
      "/home/jgwak1/tabby/graph_embedding_improvement_JY_git/graph_embedding_improvement_efforts/Baseline_Approaches/RESULTS/RandomForest__Full_Dataset_1_NoTraceUIDupdated__RandomForest_searchspace_1__10_FoldCV__search_on_all__baseline_3__flattened_graph_Ngram_events__node_type_counts__2gram__2024-02-05_000419/RandomForest__Full_Dataset_1_NoTraceUIDupdated__RandomForest_searchspace_1__10_FoldCV__search_on_all__baseline_3__flattened_graph_Ngram_events__node_type_counts__2gram__2024-02-05_000419.csv" 
   Thread_2gram__Entire_Full_Dataset_1_Tuning_csvpath = \
      "/home/jgwak1/tabby/graph_embedding_improvement_JY_git/graph_embedding_improvement_efforts/Trial_7__Thread_level_N_grams__N_gt_than_1__Similar_to_PriorGraphEmbedding/RESULTS/RandomForest__Full_Dataset_1_NoTraceUIDupdated__RandomForest_searchspace_1__10_FoldCV__search_on_all__thread_level__N>1_grams_events__nodetype5bit__2gram__sum_pool__only_train_specified_Ngram_True__2024-02-05_000246/RandomForest__Full_Dataset_1_NoTraceUIDupdated__RandomForest_searchspace_1__10_FoldCV__search_on_all__thread_level__N>1_grams_events__nodetype5bit__2gram__sum_pool__only_train_specified_Ngram_True__2024-02-05_000246.csv"

   # Dataset_1 4-gram (  ) 
   Baseline_4gram__Entire_Full_Dataset_1_Tuning_csvpath = \
      "/home/jgwak1/tabby/PW_NON_TRACE_COMMAND_DATASET/Full_dataset_results/RandomForest__Dataset-Case-1__RandomForest_searchspace_1__10_FoldCV__search_on_all__baseline_3__flattened_graph_Ngram_events__node_type_counts__4gram__2024-02-05_114013/RandomForest__Dataset-Case-1__RandomForest_searchspace_1__10_FoldCV__search_on_all__baseline_3__flattened_graph_Ngram_events__node_type_counts__4gram__2024-02-05_114013.csv" 
   Thread_4gram__Entire_Full_Dataset_1_Tuning_csvpath = \
      "/home/jgwak1/tabby/graph_embedding_improvement_JY_git/graph_embedding_improvement_efforts/Trial_7__Thread_level_N_grams__N_gt_than_1__Similar_to_PriorGraphEmbedding/RESULTS/RandomForest__Full_Dataset_1_NoTraceUIDupdated__RandomForest_searchspace_1__10_FoldCV__search_on_all__thread_level__N>1_grams_events__nodetype5bit__4gram__sum_pool__only_train_specified_Ngram_True__2024-02-05_121751/RandomForest__Full_Dataset_1_NoTraceUIDupdated__RandomForest_searchspace_1__10_FoldCV__search_on_all__thread_level__N>1_grams_events__nodetype5bit__4gram__sum_pool__only_train_specified_Ngram_True__2024-02-05_121751.csv"

   # Dataset_2 1-gram ( tuning complete & plots created  ) 
   Baseline_1gram__Entire_Full_Dataset_2_Tuning_csvpath = \
      "/home/jgwak1/tabby/graph_embedding_improvement_JY_git/graph_embedding_improvement_efforts/Baseline_Approaches/RESULTS/RandomForest__Full_Dataset_2_NoTraceUIDupdated__RandomForest_searchspace_1__10_FoldCV__search_on_all__baseline_3__flattened_graph_Ngram_events__node_type_counts__1gram__2024-02-05_001844/RandomForest__Full_Dataset_2_NoTraceUIDupdated__RandomForest_searchspace_1__10_FoldCV__search_on_all__baseline_3__flattened_graph_Ngram_events__node_type_counts__1gram__2024-02-05_001844.csv" 
   Thread_1gram__Entire_Full_Dataset_2_Tuning_csvpath = \
      "/home/jgwak1/tabby/graph_embedding_improvement_JY_git/graph_embedding_improvement_efforts/Trial_7__Thread_level_N_grams__N_gt_than_1__Similar_to_PriorGraphEmbedding/RESULTS/RandomForest__Full_Dataset_2_NoTraceUIDupdated__RandomForest_searchspace_1__10_FoldCV__search_on_all__thread_level__N>1_grams_events__nodetype5bit__1gram__sum_pool__only_train_specified_Ngram_True__2024-02-05_002314/RandomForest__Full_Dataset_2_NoTraceUIDupdated__RandomForest_searchspace_1__10_FoldCV__search_on_all__thread_level__N>1_grams_events__nodetype5bit__1gram__sum_pool__only_train_specified_Ngram_True__2024-02-05_002314.csv"

   # Dataset_2 2-gram (  ) 
   Baseline_2gram__Entire_Full_Dataset_2_Tuning_csvpath = \
      "/home/jgwak1/tabby/graph_embedding_improvement_JY_git/graph_embedding_improvement_efforts/Baseline_Approaches/RESULTS/RandomForest__Full_Dataset_2_NoTraceUIDupdated__RandomForest_searchspace_1__10_FoldCV__search_on_all__baseline_3__flattened_graph_Ngram_events__node_type_counts__2gram__2024-02-05_002008/RandomForest__Full_Dataset_2_NoTraceUIDupdated__RandomForest_searchspace_1__10_FoldCV__search_on_all__baseline_3__flattened_graph_Ngram_events__node_type_counts__2gram__2024-02-05_002008.csv" 
   Thread_2gram__Entire_Full_Dataset_2_Tuning_csvpath = \
      "/home/jgwak1/tabby/graph_embedding_improvement_JY_git/graph_embedding_improvement_efforts/Trial_7__Thread_level_N_grams__N_gt_than_1__Similar_to_PriorGraphEmbedding/RESULTS/RandomForest__Full_Dataset_2_NoTraceUIDupdated__RandomForest_searchspace_1__10_FoldCV__search_on_all__thread_level__N>1_grams_events__nodetype5bit__2gram__sum_pool__only_train_specified_Ngram_True__2024-02-05_002353/RandomForest__Full_Dataset_2_NoTraceUIDupdated__RandomForest_searchspace_1__10_FoldCV__search_on_all__thread_level__N>1_grams_events__nodetype5bit__2gram__sum_pool__only_train_specified_Ngram_True__2024-02-05_002353.csv"

   # Dataset_2 4-gram (  ) 
   Baseline_4gram__Entire_Full_Dataset_2_Tuning_csvpath = \
      "/home/jgwak1/tabby/PW_NON_TRACE_COMMAND_DATASET/Full_dataset_results/RandomForest__Dataset-Case-2__RandomForest_searchspace_1__10_FoldCV__search_on_all__baseline_3__flattened_graph_Ngram_events__node_type_counts__4gram__2024-02-05_113957/RandomForest__Dataset-Case-2__RandomForest_searchspace_1__10_FoldCV__search_on_all__baseline_3__flattened_graph_Ngram_events__node_type_counts__4gram__2024-02-05_113957.csv" 
   Thread_4gram__Entire_Full_Dataset_2_Tuning_csvpath = \
      "/home/jgwak1/tabby/graph_embedding_improvement_JY_git/graph_embedding_improvement_efforts/Trial_7__Thread_level_N_grams__N_gt_than_1__Similar_to_PriorGraphEmbedding/RESULTS/RandomForest__Full_Dataset_2_NoTraceUIDupdated__RandomForest_searchspace_1__10_FoldCV__search_on_all__thread_level__N>1_grams_events__nodetype5bit__4gram__sum_pool__only_train_specified_Ngram_True__2024-02-05_122301/RandomForest__Full_Dataset_2_NoTraceUIDupdated__RandomForest_searchspace_1__10_FoldCV__search_on_all__thread_level__N>1_grams_events__nodetype5bit__4gram__sum_pool__only_train_specified_Ngram_True__2024-02-05_122301.csv"


   Baseline_Ngrams__Entire_Full_Dataset_1_Tuning_csvpaths = {
      "Baseline_1gram": Baseline_1gram__Entire_Full_Dataset_1_Tuning_csvpath,
      "Baseline_2gram": Baseline_2gram__Entire_Full_Dataset_1_Tuning_csvpath,
      "Baseline_4gram": Baseline_4gram__Entire_Full_Dataset_1_Tuning_csvpath,
   }

   Baseline_Ngrams__Entire_Full_Dataset_2_Tuning_csvpaths = {
      "Baseline_1gram": Baseline_1gram__Entire_Full_Dataset_2_Tuning_csvpath,
      "Baseline_2gram": Baseline_2gram__Entire_Full_Dataset_2_Tuning_csvpath,
      "Baseline_4gram": Baseline_4gram__Entire_Full_Dataset_2_Tuning_csvpath,
   }

   Thread_Ngrams__Entire_Full_Dataset_1_Tuning_csvpaths = {
      "Thread_1gram": Thread_1gram__Entire_Full_Dataset_1_Tuning_csvpath,
      "Thread_2gram": Thread_2gram__Entire_Full_Dataset_1_Tuning_csvpath,
      "Thread_4gram": Thread_4gram__Entire_Full_Dataset_1_Tuning_csvpath,
   }

   Thread_Ngrams__Entire_Full_Dataset_2_Tuning_csvpaths = {
      "Thread_1gram": Thread_1gram__Entire_Full_Dataset_2_Tuning_csvpath,
      "Thread_2gram": Thread_2gram__Entire_Full_Dataset_2_Tuning_csvpath,
      "Thread_4gram": Thread_4gram__Entire_Full_Dataset_2_Tuning_csvpath,
   }





   # ----------------------------------------------- 
   # Set the following
   Ngram_approach_choice = Thread_Ngrams__Entire_Full_Dataset_2_Tuning_csvpaths
   Dataset_name = "Full Dataset-2"        # e.g. Full Dataset-1
   Tuned_on_Entire_or_Train = "Entire"    # 'Entire' or 'Train'
   # -------------------------------------------------------------------------------------------------------------------------



   plot_distribution_shift_histograms(
                                       Ngram_approach_choice,
                                       Dataset_name,
                                       Tuned_on_Entire_or_Train
                                      )







   # plt.hist(Baseline_Ngram__Entire_Full_Dataset_1__AvgVal_Accuracy, 
   #          bins= 50, density=False, alpha=0.7, color='blue', label='Baseline 2-gram')
   #          # bins=len(Baseline_Ngram__Entire_Full_Dataset_1__AvgVal_Accuracy), density=True, alpha=0.7, color='blue', label='Histogram')

   # plt.hist(Thread_Ngram__Entire_Full_Dataset_1__AvgVal_Accuracy, 
   #          bins= 50, density=False, alpha=0.7, color='red', label='Thread 2-gram')

   # plt.title('Full Dataset-1', fontsize=11, pad=10)
   # plt.xlabel('Avg.Val.Accuracy', fontsize=11, labelpad=10)
   # plt.ylabel('Count of Hyperparam. Sets (Total 6913)', fontsize=11, labelpad=10)
   # plt.legend(fontsize=10)
   # plt.savefig(f"{current_fpath.parent}/Dist_for__BaselineNgram_VS_ThreadNgram__Entire_Full_Dataset_1__AvgVal_Accuracy.png")
   # # plt.tight_layout()

   # plt.close()

   # plt.hist(Baseline_Ngram__Entire_Full_Dataset_1__AvgVal_F1, 
   #          bins=50, density=False, alpha=0.7, color='blue', label='Baseline 2-gram')
   # plt.hist(Thread_Ngram__Entire_Full_Dataset_1__AvgVal_F1, 
   #          bins=50, density=False, alpha=0.7, color='red', label='Thread 2-gram')

   # plt.title('Full Dataset-1', fontsize=11, pad=10)
   # plt.xlabel('Avg.Val.F1', fontsize=11, labelpad=10)
   # plt.ylabel('Count of Hyperparam. Sets (Total 6913)', fontsize=11, labelpad=10)
   # plt.legend(fontsize=10)
   # # plt.tight_layout()
   # plt.savefig(f"{current_fpath.parent}/Dist_for__BaselineNgram_VS_ThreadNgram__Entire_Full_Dataset_1__AvgVal_F1.png")
   # plt.close()
