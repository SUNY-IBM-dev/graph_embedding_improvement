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
# 
#     - If normal-distribution, can try paired parameteric hypothesis-testing like: "pair-t-test"
#     - If NOT normal-distribution, can try paired non-parameteric hypothesis-testing like: ""
#
#     [ References ]:
#        https://towardsdatascience.com/statistical-tests-for-comparing-machine-learning-and-baseline-performance-4dfc9402e46f
#        https://machinelearningmastery.com/statistical-significance-tests-for-comparing-machine-learning-algorithms/
#        https://machinelearningmastery.com/mcnemars-test-for-machine-learning/
#
#

# P-value

import pandas as pd 

if __name__ == "__main__":

   print()