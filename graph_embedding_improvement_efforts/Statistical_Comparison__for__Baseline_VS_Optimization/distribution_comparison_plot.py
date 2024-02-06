import pandas as pd 
import matplotlib.pyplot as plt

if __name__ == "__main__":

   # Dataset_1 2gram
   Baseline_Ngram__Entire_Full_Dataset_1_Tuning_csvpath = \
      "/home/jgwak1/tabby/graph_embedding_improvement_JY_git/graph_embedding_improvement_efforts/Baseline_Approaches/RESULTS/RandomForest__Full_Dataset_1_NoTraceUIDupdated__RandomForest_searchspace_1__10_FoldCV__search_on_all__baseline_3__flattened_graph_Ngram_events__node_type_counts__2gram__2024-02-05_000419/RandomForest__Full_Dataset_1_NoTraceUIDupdated__RandomForest_searchspace_1__10_FoldCV__search_on_all__baseline_3__flattened_graph_Ngram_events__node_type_counts__2gram__2024-02-05_000419.csv" 
   Thread_Ngram__Entire_Full_Dataset_1_Tuning_csvpath = \
      "/home/jgwak1/tabby/graph_embedding_improvement_JY_git/graph_embedding_improvement_efforts/Trial_7__Thread_level_N_grams__N_gt_than_1__Similar_to_PriorGraphEmbedding/RESULTS/RandomForest__Full_Dataset_1_NoTraceUIDupdated__RandomForest_searchspace_1__10_FoldCV__search_on_all__thread_level__N>1_grams_events__nodetype5bit__2gram__sum_pool__only_train_specified_Ngram_True__2024-02-05_000246/RandomForest__Full_Dataset_1_NoTraceUIDupdated__RandomForest_searchspace_1__10_FoldCV__search_on_all__thread_level__N>1_grams_events__nodetype5bit__2gram__sum_pool__only_train_specified_Ngram_True__2024-02-05_000246.csv"

   Baseline_Ngram__Entire_Full_Dataset_1_Tuning_df = pd.read_csv(Baseline_Ngram__Entire_Full_Dataset_1_Tuning_csvpath)
   Thread_Ngram__Entire_Full_Dataset_1_Tuning_df = pd.read_csv(Thread_Ngram__Entire_Full_Dataset_1_Tuning_csvpath)




   print()

   Baseline_Ngram__Entire_Full_Dataset_1__AvgVal_Accuracy = Baseline_Ngram__Entire_Full_Dataset_1_Tuning_df['Avg_Val_Accuracy']
   Baseline_Ngram__Entire_Full_Dataset_1__AvgVal_F1 = Baseline_Ngram__Entire_Full_Dataset_1_Tuning_df['Avg_Val_F1']   

   Thread_Ngram__Entire_Full_Dataset_1__AvgVal_Accuracy = Thread_Ngram__Entire_Full_Dataset_1_Tuning_df['Avg_Val_Accuracy']
   Thread_Ngram__Entire_Full_Dataset_1__AvgVal_F1 = Thread_Ngram__Entire_Full_Dataset_1_Tuning_df['Avg_Val_F1']

   # First Check normal distribution

   plt.hist(Baseline_Ngram__Entire_Full_Dataset_1__AvgVal_Accuracy, 
            bins= 50, density=False, alpha=0.7, color='blue', label='Baseline 2-gram')
            # bins=len(Baseline_Ngram__Entire_Full_Dataset_1__AvgVal_Accuracy), density=True, alpha=0.7, color='blue', label='Histogram')

   plt.hist(Thread_Ngram__Entire_Full_Dataset_1__AvgVal_Accuracy, 
            bins= 50, density=False, alpha=0.7, color='red', label='Thread 2-gram')

   plt.title('Full Dataset-1', fontsize=11, pad=10)
   plt.xlabel('Avg.Val.Accuracy', fontsize=11, labelpad=10)
   plt.ylabel('Count of Hyperparam. Sets (Total 6913)', fontsize=11, labelpad=10)
   plt.legend(fontsize=10)
   plt.savefig("/home/jgwak1/tabby/graph_embedding_improvement_JY_git/graph_embedding_improvement_efforts/Statistical_Comparison__for__Baseline_VS_Optimization/Dist_for__BaselineNgram_VS_ThreadNgram__Entire_Full_Dataset_1__AvgVal_Accuracy.png")
   # plt.tight_layout()

   plt.close()


   plt.hist(Baseline_Ngram__Entire_Full_Dataset_1__AvgVal_F1, 
            bins=50, density=False, alpha=0.7, color='blue', label='Baseline 2-gram')
   plt.hist(Thread_Ngram__Entire_Full_Dataset_1__AvgVal_F1, 
            bins=50, density=False, alpha=0.7, color='red', label='Thread 2-gram')

   plt.title('Full Dataset-1', fontsize=11, pad=10)
   plt.xlabel('Avg.Val.F1', fontsize=11, labelpad=10)
   plt.ylabel('Count of Hyperparam. Sets (Total 6913)', fontsize=11, labelpad=10)
   plt.legend(fontsize=10)
   # plt.tight_layout()
   
   plt.savefig("/home/jgwak1/tabby/graph_embedding_improvement_JY_git/graph_embedding_improvement_efforts/Statistical_Comparison__for__Baseline_VS_Optimization/Dist_for__BaselineNgram_VS_ThreadNgram__Entire_Full_Dataset_1__AvgVal_F1.png")
   
   plt.close()



   def plot_distribution_comparison_histograms(Baseline_AvgVal_Accuracies,
                                               Optimization_AvgVal_Accuracies,
                                               
                                               Baseline_AvgVal_F1s,
                                               Optimization_AvgVal_F1s,

                                               plot_fname,
                                               ):


      plt.hist(Baseline_AvgVal_Accuracies, 
               bins= 50, density=False, alpha=0.7, color='blue', label='Baseline 2-gram')
               # bins=len(Baseline_Ngram__Entire_Full_Dataset_1__AvgVal_Accuracy), density=True, alpha=0.7, color='blue', label='Histogram')

      plt.hist(Optimization_AvgVal_Accuracies, 
               bins= 50, density=False, alpha=0.7, color='red', label='Thread 2-gram')

      plt.title('Full Dataset-1', fontsize=11, pad=10)
      plt.xlabel('Avg.Val.Accuracy', fontsize=11, labelpad=10)
      plt.ylabel('Count of Hyperparam. Sets (Total 6913)', fontsize=11, labelpad=10)
      plt.legend(fontsize=10)
      plt.savefig(f"/home/jgwak1/tabby/graph_embedding_improvement_JY_git/graph_embedding_improvement_efforts/Statistical_Comparison__for__Baseline_VS_Optimization/{plot_fname}__AvgVal_Accuracy.png")
      # plt.tight_layout()

      plt.close()


      plt.hist(Baseline_AvgVal_F1s, 
               bins=50, density=False, alpha=0.7, color='blue', label='Baseline 2-gram')
      plt.hist(Optimization_AvgVal_F1s, 
               bins=50, density=False, alpha=0.7, color='red', label='Thread 2-gram')

      plt.title('Full Dataset-2', fontsize=11, pad=10)
      plt.xlabel('Avg.Val.F1', fontsize=11, labelpad=10)
      plt.ylabel('Count of Hyperparam. Sets (Total 6913)', fontsize=11, labelpad=10)
      plt.legend(fontsize=10)
      # plt.tight_layout()
      
      plt.savefig(f"/home/jgwak1/tabby/graph_embedding_improvement_JY_git/graph_embedding_improvement_efforts/Statistical_Comparison__for__Baseline_VS_Optimization/{plot_fname}__AvgVal_F1.png")
      
      plt.close()


