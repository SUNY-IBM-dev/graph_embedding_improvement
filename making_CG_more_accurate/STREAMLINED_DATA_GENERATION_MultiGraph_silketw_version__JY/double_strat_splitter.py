import os
from sklearn.model_selection import train_test_split
from collections import Counter
import pandas as pd
import pathlib
import shutil



if __name__ == "__main__":

     # Full Dataset-1 	All Benign, Mawlare Pickles -------------------------------------------------------------------------------------------
   # #   # Sources of all samples
   #   Src__Benign_ALL_Pickles_dirpath = "/home/jgwak1/tabby/graph_embedding_improvement_JY_git/making_CG_more_accurate/PW_NON_TRACE_COMMAND_DATASET/Benign_Case1/Indices/Processed_Benign_ONLY_TaskName_edgeattr_full_set_5bit"
   #   Src__Malware_ALL_Pickles_dirpath = "/home/jgwak1/tabby/graph_embedding_improvement_JY_git/making_CG_more_accurate/PW_NON_TRACE_COMMAND_DATASET/Malware_Case1/Indices/Processed_Malware_ONLY_TaskName_edgeattr_full_set_5bit"

   #   # Destinations of double-stratified splitted samples
   #   Dest__Benign_Train_dirpath = "/home/jgwak1/tabby/graph_embedding_improvement_JY_git/making_CG_more_accurate/PW_NON_TRACE_COMMAND_DATASET/Benign_Case1/Full_train_set__double_strat"
   #   Dest__Benign_Test_dirpath = "/home/jgwak1/tabby/graph_embedding_improvement_JY_git/making_CG_more_accurate/PW_NON_TRACE_COMMAND_DATASET/Benign_Case1/Full_test_set__double_strat"

   #   Dest__Malware_Train_dirpath = "/home/jgwak1/tabby/graph_embedding_improvement_JY_git/making_CG_more_accurate/PW_NON_TRACE_COMMAND_DATASET/Malware_Case1/Full_train_set__double_strat"
   #   Dest__Malware_Test_dirpath = "/home/jgwak1/tabby/graph_embedding_improvement_JY_git/making_CG_more_accurate/PW_NON_TRACE_COMMAND_DATASET/Malware_Case1/Full_test_set__double_strat"


     # Full Dataset-2 	All Benign, Mawlare Pickles -------------------------------------------------------------------------------------------
     # Sources of all samples
     Src__Benign_ALL_Pickles_dirpath = "/home/jgwak1/tabby/graph_embedding_improvement_JY_git/making_CG_more_accurate/PW_NON_TRACE_COMMAND_DATASET/Benign_Case2/Indices/Processed_Benign_ONLY_TaskName_edgeattr_full_set_5bit"
     Src__Malware_ALL_Pickles_dirpath = "/home/jgwak1/tabby/graph_embedding_improvement_JY_git/making_CG_more_accurate/PW_NON_TRACE_COMMAND_DATASET/Malware_Case2/Indices/Processed_Malware_ONLY_TaskName_edgeattr_full_set_5bit"

     # Destinations of double-stratified splitted samples
     Dest__Benign_Train_dirpath = "/home/jgwak1/tabby/graph_embedding_improvement_JY_git/making_CG_more_accurate/PW_NON_TRACE_COMMAND_DATASET/Benign_Case2/Full_train_set__double_strat"
     Dest__Benign_Test_dirpath = "/home/jgwak1/tabby/graph_embedding_improvement_JY_git/making_CG_more_accurate/PW_NON_TRACE_COMMAND_DATASET/Benign_Case2/Full_test_set__double_strat"

     Dest__Malware_Train_dirpath = "/home/jgwak1/tabby/graph_embedding_improvement_JY_git/making_CG_more_accurate/PW_NON_TRACE_COMMAND_DATASET/Malware_Case2/Full_train_set__double_strat"
     Dest__Malware_Test_dirpath = "/home/jgwak1/tabby/graph_embedding_improvement_JY_git/making_CG_more_accurate/PW_NON_TRACE_COMMAND_DATASET/Malware_Case2/Full_test_set__double_strat"



     #-----------------------------------------------------------------------------------------------------------------------------------
     if (not os.path.exists(Dest__Benign_Train_dirpath)) or (not os.path.exists(Dest__Benign_Test_dirpath)) or\
        (not os.path.exists(Dest__Malware_Train_dirpath)) or (not os.path.exists(Dest__Malware_Test_dirpath)):
        raise RuntimeError("All Destination Paths need to exist")
     if ( len(os.listdir(Dest__Benign_Train_dirpath)) >  0 ) or ( len(os.listdir(Dest__Benign_Test_dirpath)) > 0 ) or\
        ( len(os.listdir(Dest__Malware_Train_dirpath)) >  0 ) or ( len(os.listdir(Dest__Malware_Test_dirpath)) > 0 ): 
        raise RuntimeError("All Destination Paths must be empty in the beginning")

     #-----------------------------------------------------------------------------------------------------------------------------------


     Benign_ALL_Pickle_fnames = [f for f in os.listdir( Src__Benign_ALL_Pickles_dirpath ) if "Processed" in f ]
     Malware_ALL_Pickle_fnames = [f for f in os.listdir( Src__Malware_ALL_Pickles_dirpath ) if "Processed" in f ]

     
     ALL_Pickle_fnames = Benign_ALL_Pickle_fnames + Malware_ALL_Pickle_fnames

     # JY @ 2023-06-27
     #      Now we want to acheive stratified-K-Fold split based on not only label but also datasource.
     #      So in each split, we want the label+datasource combination to be consistent.
     #      e.g. Across folds, we want ratio of benign-fleschutz, benign-jhochwald, ... malware-empire, malware-nishang,.. to be consistent
     #      For this, first generate a group-list.
     X_grouplist = []


     for data_name in ALL_Pickle_fnames:
            # benign --------------
            if "powershell-master" in data_name:
               X_grouplist.append("benign_fleschutz")
            elif "jhochwald" in data_name:
               X_grouplist.append("benign_jhochwald")
            elif "devblackops" in data_name:
               X_grouplist.append("benign_devblackops")
            elif "farag2" in data_name:
               X_grouplist.append("benign_farag2")
            elif "jimbrig" in data_name:
               X_grouplist.append("benign_jimbrig")
            elif "jrussellfreelance" in data_name:
               X_grouplist.append("benign_jrussellfreelance")
            elif "nickrod518" in data_name:
               X_grouplist.append("benign_nickrod518")
            elif "redttr" in data_name:
               X_grouplist.append("benign_redttr")
            elif "sysadmin-survival-kit" in data_name:
               X_grouplist.append("benign_sysadmin-survival-kit")
            elif "stevencohn" in data_name:
               X_grouplist.append("benign_stevencohn")
            elif "ledragox" in data_name:
               X_grouplist.append("benign_ledrago")
            elif "floriantim" in data_name: # Added by JY @ 2024-2-3 : pattern in dataset-2
               X_grouplist.append("benign_floriantim")
            elif "nickbeau" in data_name: # Added by JY @ 2024-2-3 : pattern in dataset-2
               X_grouplist.append("benign_nickbeau")
            # malware ------------------------------------------
            elif "empire" in data_name:
               X_grouplist.append("malware_empire")
            elif "invoke_obfuscation" in data_name:
               X_grouplist.append("malware_invoke_obfuscation")
            elif "nishang" in data_name:
               X_grouplist.append("malware_nishang")
            elif "poshc2" in data_name:
               X_grouplist.append("malware_poshc2")
            elif "mafia" in data_name:
               X_grouplist.append("malware_mafia")
            elif "offsec" in data_name:
               X_grouplist.append("malware_offsec")
            elif "powershellery" in data_name:
               X_grouplist.append("malware_powershellery")
            elif "psbits" in data_name:
               X_grouplist.append("malware_psbits")
            elif "pt_toolkit" in data_name:
               X_grouplist.append("malware_pt_toolkit")
            elif "randomps" in data_name:
               X_grouplist.append("malware_randomps")
            elif "smallposh" in data_name:
               X_grouplist.append("malware_smallposh")
            elif "asyncrat" in data_name: #PW: need to change depends on all types e.g., malware_rest
               X_grouplist.append("malware_asyncrat")
            elif "bumblebee" in data_name:
               X_grouplist.append("malware_bumblebee")
            elif "cobalt_strike" in data_name:
               X_grouplist.append("malware_cobalt_strike")
            elif "coinminer" in data_name:
               X_grouplist.append("malware_coinminer")
            elif "gozi" in data_name:
               X_grouplist.append("malware_gozi")
            elif "guloader" in data_name:
               X_grouplist.append("malware_guloader")
            elif "netsupport" in data_name:
               X_grouplist.append("malware_netsupport")
            elif "netwalker" in data_name:
               X_grouplist.append("malware_netwalker")
            elif "nw0rm" in data_name:
               X_grouplist.append("malware_nw0rm")
            elif "quakbot" in data_name:
               X_grouplist.append("malware_quakbot")
            elif "quasarrat" in data_name:
               X_grouplist.append("malware_quasarrat")
            elif "rest" in data_name:
               X_grouplist.append("malware_rest")
            elif "metasploit" in data_name:
               X_grouplist.append("malware_metasploit")                       
            # if "recollected" in data_name:
               # X_grouplist.append("malware_recollected")
            else:
               raise ValueError(f"unidentifeid pattern in {data_name}")

      # Before applying double check 

     labels = []
     data_sources = []
     for pickle_fname, labels_and_datasource in list(zip(ALL_Pickle_fnames, X_grouplist)):
         label, datasource = labels_and_datasource.split('_', 1)
         assert label in pickle_fname, f"Problem: '{label}' is not in {pickle_fname}"

         # exception for 'fleschutz'
         if datasource == "fleschutz":
            assert "powershell-master" in pickle_fname, f"Problem: 'powershell-master' (i.e. fleschutz) is not in {pickle_fname}"
         else:
            assert datasource in pickle_fname, f"Problem: '{datasource}' is not in {pickle_fname}"
         labels.append(label)
         data_sources.append(datasource)

      # correctness of X_grouplist can be checked by following
      # list(zip(X, [data_name for data_name in X.index], y, X_grouplist))


     # JY @ 2024-2-5 : 
     #                  Try to use the 'stratify = X_grouplist' as well, but just put all the count-1 ones to train-set
     #                  Then check the csv in the end

     #less_than_5_grouplist = [ x[0] for x in Counter(X_grouplist).items() if x[1] < 5 ]
     count_1_grouplist = [ x[0] for x in Counter(X_grouplist).items() if x[1] == 1 ]

     
     filtered__ALL_Pickle_fnames = [item for item in ALL_Pickle_fnames if not any(sub in item for sub in count_1_grouplist)]
     
     if len(count_1_grouplist) >= 1:
       dropped_out_count_1_pickle_fnames = [item for item in ALL_Pickle_fnames if all(sub in item for sub in count_1_grouplist)]

     filtered__X_grouplist = [item for item in X_grouplist if not any(sub in item for sub in count_1_grouplist) ]


     X_train, X_test, y_train, y_test = train_test_split(
                                                         filtered__ALL_Pickle_fnames , # X
                                                         
                                                         filtered__X_grouplist,  # y
                                                         
                                                         test_size=0.2, # 80% vs 20%

                                                         shuffle = True,  # Important -- to not end up with samples alphabetically similar in each
                                                                          #              this can give randomness in which activites within 
                                                                          # #            same label-datasource end up in train and test
                                                                          #              Otherwise, it will be in alphabetiacal order 
                                                                          #               such as add-* activities from benign-farag ending up in train
                                                                          #               and enumerate-* activities from benign-farage ending up in test
                                                                          #               WHICH WE DON'T WANT. SO SHUFFLE
                                                                          #              
                                                                          #               In other words, did not do "label-datasource-activity" level stratification
                                                                          #               But by 'shuffling', we can avoid alphabetically splitting samples within 'label-datasource'
                                                                          #               and give some randomness in splitting activities.
                                                         
                                                         stratify = filtered__X_grouplist, # BUT COMMENTED BACK IN AFTER EXPLICIT HANDLING OF COUNT-1 SAMPLES --> commenting out solved the Error of "The least populated class in y has only 1 member, which is too few. The minimum number of groups for any class cannot be less than 2."      # https://stackoverflow.com/questions/43179429/scikit-learn-error-the-least-populated-class-in-y-has-only-1-member
                                                                                           #
                                                                                           # stratify: array-like, default=None
                                                                                           #           If not None, data is split in a stratified fashion, using this as the class labels. Read more in the User Guide.     
                                                                                           #           https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
                                                         
                                                         random_state=42
                                                         )
     # Add the count -1 smaples to train set
     if len(count_1_grouplist) >= 1:
         X_train += dropped_out_count_1_pickle_fnames
         y_train += count_1_grouplist



     # Now check if stratification worked

     len(X_train) / len(ALL_Pickle_fnames)
     len(X_test) / len(ALL_Pickle_fnames)


     X_train_benign_num = len([x for x in X_train if 'benign' in x ])
     X_test_benign_num = len([x for x in X_test if 'benign' in x ])
     X_train_benign_num / (X_train_benign_num + X_test_benign_num)


     X_train_malware_num = len([x for x in X_train if 'malware' in x ])
     X_test_malware_num = len([x for x in X_test if 'malware' in x ])
     X_train_malware_num / (X_train_malware_num + X_test_malware_num)

     print()



     # Use Counter to get counts 
     train_label_datasource__counter = Counter(y_train)
     train_label_datasource__count_df = pd.DataFrame(train_label_datasource__counter, index = ['train_label_datasource_count',])
     # Calculate ratios
     train_label_datasource__total_count = sum(train_label_datasource__counter.values())
     train_label_datasource__ratios = {key: count / train_label_datasource__total_count for key, count in train_label_datasource__counter.items()}
     train_label_datasource_ratio_df = pd.DataFrame(train_label_datasource__ratios, index = ['train_label_datasource_ratio',])


     # Use Counter to get counts 
     test_label_datasource__counter = Counter(y_test)
     test_label_datasource__count_df = pd.DataFrame(test_label_datasource__counter, index = ['test_label_datasource_count',])
     # Calculate ratios
     test_label_datasource__total_count = sum(test_label_datasource__counter.values())
     test_label_datasource__ratios = {key: count / test_label_datasource__total_count for key, count in test_label_datasource__counter.items()}
     test_label_datasource__ratio_df = pd.DataFrame(test_label_datasource__ratios, index = ['test_label_datasource__ratios',])

     # ------------------------------------------------------------------------------------------------------------------------------------
     
     train_and_test_label_datasource_count_comparison_df = pd.merge(train_label_datasource__count_df,
                                                                    test_label_datasource__count_df, how = 'outer')
     train_and_test_label_datasource_count_comparison_df.set_index(pd.Index(['train_label_datasource_count', 'test_label_datasource_count']), inplace= True)


     train_and_test_label_datasource_ratio_comparison_df = pd.merge(train_label_datasource_ratio_df,
                                                                    test_label_datasource__ratio_df, how = 'outer')
     train_and_test_label_datasource_ratio_comparison_df.set_index(pd.Index(['train_label_datasource_ratio', 'test_label_datasource__ratio']), inplace= True)

     # ------------------------------------------------------------------------------------------------------------------------------------

     train_and_test_label_datasource_ratio_count_comparison_df = pd.concat([train_and_test_label_datasource_ratio_comparison_df, 
                                                                            train_and_test_label_datasource_count_comparison_df])

     current_fpath = pathlib.Path(__file__).resolve()
     train_and_test_label_datasource_ratio_count_comparison_df.to_csv( os.path.join(current_fpath.parent, "train_and_test_label_datasource_ratio_count_comparison.csv" )  )


     # ------------------------------------------------------------------------------------------------------------------------------------

     # Now save out X_train and X_test to respective dirs

     for train_pickle_fname in X_train:

         if "benign" in train_pickle_fname:
             shutil.copyfile(src= os.path.join(Src__Benign_ALL_Pickles_dirpath, train_pickle_fname),
                             dst= os.path.join(Dest__Benign_Train_dirpath, train_pickle_fname) )

         elif "malware" in train_pickle_fname:
             shutil.copyfile(src= os.path.join(Src__Malware_ALL_Pickles_dirpath, train_pickle_fname),
                             dst= os.path.join(Dest__Malware_Train_dirpath, train_pickle_fname) )
         else:
            raise ValueError("No 'malware' or 'benign' in pickle ")


     for test_pickle_fname in X_test:

         if "benign" in test_pickle_fname:
             shutil.copyfile(src= os.path.join(Src__Benign_ALL_Pickles_dirpath, test_pickle_fname),
                             dst= os.path.join(Dest__Benign_Test_dirpath, test_pickle_fname) )

         elif "malware" in test_pickle_fname:
             shutil.copyfile(src= os.path.join(Src__Malware_ALL_Pickles_dirpath, test_pickle_fname),
                             dst= os.path.join(Dest__Malware_Test_dirpath, test_pickle_fname) )
         else:
            raise ValueError("No 'malware' or 'benign' in pickle ")
         

     print("check ") 
     list(zip(X_train,y_train))

     assert set(os.listdir(Dest__Benign_Test_dirpath) + os.listdir(Dest__Malware_Test_dirpath)) == set(X_test)
     assert set(os.listdir(Dest__Benign_Train_dirpath) + os.listdir(Dest__Malware_Train_dirpath)) == set(X_train)
