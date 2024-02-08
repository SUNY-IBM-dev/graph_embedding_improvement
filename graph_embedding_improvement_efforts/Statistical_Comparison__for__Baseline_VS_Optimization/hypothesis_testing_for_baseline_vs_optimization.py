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
import os
from scipy.stats import wilcoxon

# import bootstrapped.bootstrap as bs
# import bootstrapped.stats_functions as bs_stats
import numpy as np

from scipy.stats import friedmanchisquare

# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.permutation_test.html
from scipy.stats import permutation_test

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

   # ----------------------------------------------- 
   # Set the following

   Baseline_Tuning_csvpath = Baseline_2gram__Entire_Full_Dataset_1_Tuning_csvpath
   Optimization_Tuning_csvpath = Thread_2gram__Entire_Full_Dataset_1_Tuning_csvpath

   statistic_choice = "mean_statistic"
   # -------------------------------------------------------------------------------------------------------------------------


   Baseline_Tuning_df = pd.read_csv(Baseline_Tuning_csvpath)
   Optimization_Tuning_df = pd.read_csv(Optimization_Tuning_csvpath)
   print()

   Baseline_Tuning__AvgVal_Accuracy = Baseline_Tuning_df['Avg_Val_Accuracy']
   Baseline_Tuning__AvgVal_F1 = Baseline_Tuning_df['Avg_Val_F1']   

   Optimization_Tuning__AvgVal_Accuracy = Optimization_Tuning_df['Avg_Val_Accuracy']
   Optimization_Tuning__AvgVal_F1 = Optimization_Tuning_df['Avg_Val_F1']


   all_possible_hyperparam_sets_num = 6912
   available_hyperparam_sets_num = min( len(Baseline_Tuning__AvgVal_Accuracy), len(Optimization_Tuning__AvgVal_Accuracy) )
   if available_hyperparam_sets_num != all_possible_hyperparam_sets_num:
      tuning_incomplete = True
      print(f"Either one did not finish tuning all {all_possible_hyperparam_sets_num} hyperparameter-sets.\nSo doing based on the shorter length {available_hyperparam_sets_num}", flush = True)

   # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.permutation_test.html
   # Assuming a two-sided test (alternative='two-sided'), you can change it if needed

   # Better understand this -- WHAT WOULD BE A GOOD STATISTIC FOR OUR PROBLEM?
   def mean_statistic(x, y, axis):
      return np.mean(x, axis=axis) - np.mean(y, axis=axis)

   def sum_of_ranks(y_true, y_pred):
      ranks = np.argsort(np.concatenate((y_true, y_pred)))
      rank_sum_baseline = ranks[:len(y_true)].sum()
      rank_sum_optimization = ranks[len(y_true):].sum()
      return rank_sum_optimization - rank_sum_baseline

   # Your test statistic function (replace this with your own)
   def anova_f_statistic(*groups):
      return np.var(np.concatenate(groups)) / np.mean([np.var(group) for group in groups])
   
   def correlation_coefficient(x, y):
       return np.corrcoef(x, y)[0, 1]


   statistic_dict = {"mean_statistic": mean_statistic,
   "sum_of_ranks": sum_of_ranks,
   "anova_f_statistic": anova_f_statistic,
   "correlation_coefficient": correlation_coefficient, }

   result = permutation_test( ( Baseline_Tuning__AvgVal_F1[:available_hyperparam_sets_num] ,  
                                Optimization_Tuning__AvgVal_F1[:available_hyperparam_sets_num] ) ,
                                statistic_dict[statistic_choice], 
                                alternative='two-sided')


   print(f"{os.path.split(Baseline_Tuning_csvpath)[1]}\nvs\n{os.path.split(Optimization_Tuning_csvpath)[1]}")
   if result.pvalue < 0.05:
      print(f"AvgVal_F1: The difference is statistically significant (permutation_test-{statistic_choice}).\np-value: {result.pvalue}")
   else:
      print(f"AvgVal_F1: The difference is not statistically significant (permutation_test-{statistic_choice}).\np-value: {result.pvalue}")

   result = permutation_test( ( Baseline_Tuning__AvgVal_Accuracy[:available_hyperparam_sets_num] ,  
                                Optimization_Tuning__AvgVal_Accuracy[:available_hyperparam_sets_num] ) ,
                                statistic_dict[statistic_choice], 
                                alternative='two-sided')
   if result.pvalue < 0.05:
      print(f"AvgVal_Accuracy: The difference is statistically significant (permutation_test-{statistic_choice}).\np-value: {result.pvalue}")
   else:
      print(f"AvgVal_Accuracy: The difference is not statistically significant (permutation_test-{statistic_choice}).\np-value: {result.pvalue}")


   # results = bs.bootstrap_ab(Baseline_Tuning__AvgVal_Accuracy[:available_hyperparam_sets_num], 
   #                           Optimization_Tuning__AvgVal_Accuracy[:available_hyperparam_sets_num], 
   #                           stat_func=bs_stats.median, compare_func=bs_stats.difference)

   # # Print the results
   # print(f"Bootstrapped 95% CI for the median difference: {results}")
   # print(f"P-value: {results.p_value}")

   # # Check for statistical significance (using a common significance level of 0.05)
   # if results.p_value < 0.05:
   #    print("The difference is statistically significant.")
   # else:
   #    print("The difference is not statistically significant.")

   # results = bs.bootstrap_ab(Baseline_Tuning__AvgVal_F1[:available_hyperparam_sets_num], 
   #                           Optimization_Tuning__AvgVal_F1[:available_hyperparam_sets_num], 
   #                           stat_func=bs_stats.median, compare_func=bs_stats.difference)

   # # Print the results
   # print(f"Bootstrapped 95% CI for the median difference: {results}")
   # print(f"P-value: {results.p_value}")

   # # Check for statistical significance (using a common significance level of 0.05)
   # if results.p_value < 0.05:
   #    print("The difference is statistically significant.")
   # else:
   #    print("The difference is not statistically significant.")



   ''' The Friedman test is a non-parametric test used to determine if there are any statistically significant differences between the means of three or more paired groups.  
       SO NOT APPLICALE
   '''

   # # Perform the Friedman test
   # statistic, p_value = friedmanchisquare(Baseline_Tuning__AvgVal_Accuracy[:available_hyperparam_sets_num], 
   #                                        Optimization_Tuning__AvgVal_Accuracy[:available_hyperparam_sets_num])

   # # Print the results
   # print(f"Friedman test statistic: {statistic}")
   # print(f"P-value: {p_value}")

   # # Check for statistical significance (using a common significance level of 0.05)
   # if p_value < 0.05:
   #    print("There are significant differences among the groups.")
   # else:
   #    print("There are no significant differences among the groups.")


   # # Perform the Friedman test
   # statistic, p_value = friedmanchisquare(Baseline_Tuning__AvgVal_F1[:available_hyperparam_sets_num], 
   #                                        Optimization_Tuning__AvgVal_F1[:available_hyperparam_sets_num])

   # # Print the results
   # print(f"Friedman test statistic: {statistic}")
   # print(f"P-value: {p_value}")

   # # Check for statistical significance (using a common significance level of 0.05)
   # if p_value < 0.05:
   #    print("There are significant differences among the groups.")
   # else:
   #    print("There are no significant differences among the groups.")      



   ''' The Wilcoxon signed-rank test is paired non-parametric but problem is it has 'symmetry assumption'.
       Our Avg.Val.Accuracies and Avg.Val.F1-scores are multimodal distribution.  
       SO NOT APPLICALE
   '''

   # # It turns out nose are NOT normal distribution, but rather multi-modal distribution
   # # So apply paired-Non-parameteric statistical signficiance test

   # # Perform the Wilcoxon signed-rank test  --- Try different parameters?
   # # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wilcoxon.html
   # statistic, p_value = wilcoxon(Baseline_Tuning__AvgVal_Accuracy[:available_hyperparam_sets_num], 
   #                               Optimization_Tuning__AvgVal_Accuracy[:available_hyperparam_sets_num],
   #                               zero_method = "wilcox",
   #                               # correction = "true",
   #                               mode = "exact")

   # # Print the results
   # print(f"Wilcoxon signed-rank statistic (Avg.Val.Acc): {statistic}")
   # print(f"P-value: {p_value}")

   # # Check for statistical significance (using a common significance level of 0.05)
   # if p_value < 0.05:
   #    print("The difference is statistically significant.")
   # else:
   #    print("The difference is not statistically significant.")




   # # Perform the Wilcoxon signed-rank test
   # statistic, p_value = wilcoxon(Baseline_Tuning__AvgVal_F1[:available_hyperparam_sets_num], 
   #                               Optimization_Tuning__AvgVal_F1[:available_hyperparam_sets_num],

   #                               zero_method = "wilcox",
   #                               # correction = "true",
   #                               mode = "auto")

   # # Print the results
   # print(f"Wilcoxon signed-rank statistic (Avg.Val.F1): {statistic}")
   # print(f"P-value: {p_value}")

   # # Check for statistical significance (using a common significance level of 0.05)
   # if p_value < 0.05:
   #    print("The difference is statistically significant.")
   # else:
   #    print("The difference is not statistically significant.")
