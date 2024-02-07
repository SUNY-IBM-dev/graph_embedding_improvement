# Statistical Hypothesis Testing for Mean/Max-Avg.Val.Scores(Acc/F1) of Baseline vs. Optimization
#
#     Baseline's distribution of all average-validation-scores(accuracy / F1)  
#        VS.
#     Optimization's distribution of all average-validation-scores(accuracy / F1)
#
#
#     In my comparison between the baseline and optimization approaches, 
#     the data is considered "paired" because each observation in one group (baseline) 
#     is directly related or matched to an observation in the other group (optimization).      
#
#
#     - If normal-distribution, can try paired parameteric hypothesis-testing like: "pair-t-test"
#     - If NOT normal-distribution, can try paired non-parameteric hypothesis-testing like: "Wilcoxon signed-rank test"
#
#     Compare the overall performance of the baseline and optimization approaches across all data splits and model configurations. 
#     This is comparing the the distributions of 'all Avg.Val.Acc/F1' obtained from the two approaches.
#
#
#     [ References ]:
#        https://towardsdatascience.com/statistical-tests-for-comparing-machine-learning-and-baseline-performance-4dfc9402e46f
#        https://machinelearningmastery.com/statistical-significance-tests-for-comparing-machine-learning-algorithms/
#        https://machinelearningmastery.com/mcnemars-test-for-machine-learning/
#
#
# 
#
#     P-value

import pandas as pd 
import matplotlib.pyplot as plt
from scipy.stats import norm

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
