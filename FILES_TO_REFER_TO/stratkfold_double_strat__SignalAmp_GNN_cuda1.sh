#!/bin/bash

# for K could try 10 sometime instead of 5, when narrowed down.

# RUN bash stratkfold_cuda1.sh &> stratkfold_cuda1_exec.txt
#`seq -s ' ' 42 1 50`\

cuda_num=1

CUBLAS_WORKSPACE_CONFIG=:16:8 python3 /home/jgwak1/SUNYIBM/gnn_v1_615/option_3_StratKfoldCV/stratkfold_double_strat__SignalAmp_GNN.py\
                                                                                             --model_cls GNN_Signal_Amplification__ver2\
                                                                                             --K 10\
                                                                                             --num_epochs 1000\
                                                                                             --batch_size 32\
                                                                                             --device cuda:$cuda_num\
                                                                                             --data_split_ratio 1 0 0\
                                                                                             --best_criteria train_loss\
                                                                                             --save_bestmodel_fitted_on_whole_data f

