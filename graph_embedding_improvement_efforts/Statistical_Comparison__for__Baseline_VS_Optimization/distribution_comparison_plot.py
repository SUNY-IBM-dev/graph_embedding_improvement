import pandas as pd 
import matplotlib.pyplot as plt
import pathlib
current_fpath = pathlib.Path(__file__).resolve()



def plot_distribution_comparison_histograms(Baseline_AvgVal_Accuracies,
                                             Optimization_AvgVal_Accuracies,
                                             
                                             Baseline_AvgVal_F1s,
                                             Optimization_AvgVal_F1s,
                                             # -----------------------------------------------
                                             Baseline_label, # e.g. Baseline 2-gram
                                             Optimization_label, # e.g. Thread 2-gram

                                             Dataset_name, # e.g. Full Dataset-1
                                             Tuned_on_Entire_or_Train = "Entire"

                                             ):

   # e.g. Dist_for__Baseline_2gram_VS_Thread_2gram__Entire_Full_Dataset_1
   plot_fname = f"Dist_for__{Baseline_label}_VS_{Optimization_label}__{Dataset_name}_{Tuned_on_Entire_or_Train}" 


   all_possible_hyperparam_sets_num = 6912
   available_hyperparam_sets_num = min( len(Baseline_AvgVal_Accuracies), len(Optimization_AvgVal_Accuracies) )
   if available_hyperparam_sets_num != all_possible_hyperparam_sets_num:
      print(f"Either one did not finish tuning all {all_possible_hyperparam_sets_num} hyperparameter-sets.\nSo plotting based on the shorter length {available_hyperparam_sets_num}", flush = True)

   plt.hist(Baseline_AvgVal_Accuracies, 
            bins= 50, density=False, alpha=0.7, color='blue', label= Baseline_label )
            # bins=len(Baseline_Ngram__Entire_Full_Dataset_1__AvgVal_Accuracy), density=True, alpha=0.7, color='blue', label='Histogram')

   plt.hist(Optimization_AvgVal_Accuracies, 
            bins= 50, density=False, alpha=0.7, color='red', label= Optimization_label)


   plt.title(f"{Dataset_name}", fontsize=11, pad=10)
   plt.xlabel('Avg.Val.Accuracy', fontsize=11, labelpad=10)
   plt.ylabel(f'Count of Hyperparam. Sets (Total {available_hyperparam_sets_num})', fontsize=11, labelpad=10)
   plt.legend(fontsize=10)
   plt.savefig(f"{current_fpath.parent}/{plot_fname}__AvgVal_Accuracy.png")
   # plt.tight_layout()

   plt.close()


   plt.hist(Baseline_AvgVal_F1s, 
            bins=50, density=False, alpha=0.7, color='blue', label='Baseline 2-gram')
   plt.hist(Optimization_AvgVal_F1s, 
            bins=50, density=False, alpha=0.7, color='red', label='Thread 2-gram')

   plt.title(f"{Dataset_name}", fontsize=11, pad=10)
   plt.xlabel('Avg.Val.F1', fontsize=11, labelpad=10)
   plt.ylabel(f'Count of Hyperparam. Sets (Total {available_hyperparam_sets_num})', fontsize=11, labelpad=10)
   plt.legend(fontsize=10)
   # plt.tight_layout()
   
   plt.savefig(f"{current_fpath.parent}/{plot_fname}__AvgVal_F1.png")
   
   plt.close()



if __name__ == "__main__":

   # Dataset_1 2gram ( created )
   Baseline_2gram__Entire_Full_Dataset_1_Tuning_csvpath = \
      "/home/jgwak1/tabby/graph_embedding_improvement_JY_git/graph_embedding_improvement_efforts/Baseline_Approaches/RESULTS/RandomForest__Full_Dataset_1_NoTraceUIDupdated__RandomForest_searchspace_1__10_FoldCV__search_on_all__baseline_3__flattened_graph_Ngram_events__node_type_counts__2gram__2024-02-05_000419/RandomForest__Full_Dataset_1_NoTraceUIDupdated__RandomForest_searchspace_1__10_FoldCV__search_on_all__baseline_3__flattened_graph_Ngram_events__node_type_counts__2gram__2024-02-05_000419.csv" 
   Thread_2gram__Entire_Full_Dataset_1_Tuning_csvpath = \
      "/home/jgwak1/tabby/graph_embedding_improvement_JY_git/graph_embedding_improvement_efforts/Trial_7__Thread_level_N_grams__N_gt_than_1__Similar_to_PriorGraphEmbedding/RESULTS/RandomForest__Full_Dataset_1_NoTraceUIDupdated__RandomForest_searchspace_1__10_FoldCV__search_on_all__thread_level__N>1_grams_events__nodetype5bit__2gram__sum_pool__only_train_specified_Ngram_True__2024-02-05_000246/RandomForest__Full_Dataset_1_NoTraceUIDupdated__RandomForest_searchspace_1__10_FoldCV__search_on_all__thread_level__N>1_grams_events__nodetype5bit__2gram__sum_pool__only_train_specified_Ngram_True__2024-02-05_000246.csv"

   Baseline_label = "Baseline 2-gram"     # e.g. Baseline 2-gram
   Optimization_label = "Thread 2-gram"   # e.g. Thread 2-gram
   Dataset_name = "Full Dataset-1"        # e.g. Full Dataset-1
   Tuned_on_Entire_or_Train = "Entire"    # 'Entire' or 'Train'
   # -------------------------------------------------------------------------------------------------------------------------

   Baseline_Tuning_df = pd.read_csv(Baseline_2gram__Entire_Full_Dataset_1_Tuning_csvpath)
   Optimization_Tuning_df = pd.read_csv(Thread_2gram__Entire_Full_Dataset_1_Tuning_csvpath)


   Baseline__AvgVal_Accuracies = Baseline_Tuning_df['Avg_Val_Accuracy']
   Baseline__AvgVal_F1s = Baseline_Tuning_df['Avg_Val_F1']   

   Optimization__AvgVal_Accuracies = Optimization_Tuning_df['Avg_Val_Accuracy']
   Optimization__AvgVal_F1s = Optimization_Tuning_df['Avg_Val_F1']

   plot_distribution_comparison_histograms(
                                           Baseline_AvgVal_Accuracies= Baseline__AvgVal_Accuracies,
                                           Optimization_AvgVal_Accuracies= Optimization__AvgVal_Accuracies,
                                           
                                           Baseline_AvgVal_F1s= Baseline__AvgVal_F1s,
                                           Optimization_AvgVal_F1s= Optimization__AvgVal_F1s,
                                           
                                           # -----------------------------------------------
                                           Baseline_label = Baseline_label, # e.g. Baseline 2-gram
                                           Optimization_label = Optimization_label, # e.g. Thread 2-gram

                                           Dataset_name = Dataset_name, # e.g. Full Dataset-1
                                           Tuned_on_Entire_or_Train = Tuned_on_Entire_or_Train,


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