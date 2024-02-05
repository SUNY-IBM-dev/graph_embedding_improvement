import os
from sklearn.model_selection import train_test_split



if __name__ == "__main__":



     Src__Benign_ALL_Pickles_dirpath = ""
     Src__Malware_ALL_Pickles_dirpath = ""

     Dest__Benign_Train_dirpath = ""
     Dest__Benign_Test_dirpath = ""

     Dest__Malware_Train_dirpath = ""
     Dest__Malware_Test_dirpath = ""

     #-----------------------------------------------------------------------------------------------------------------------------------

     # JY @ 2023-06-27
     #      Now we want to acheive stratified-K-Fold split based on not only label but also datasource.
     #      So in each split, we want the label+datasource combination to be consistent.
     #      e.g. Across folds, we want ratio of benign-fleschutz, benign-jhochwald, ... malware-empire, malware-nishang,.. to be consistent
     #      For this, first generate a group-list.
     X_grouplist = []

     data_names = X['data_name']

     for data_name in data_names:
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


      # correctness of X_grouplist can be checked by following
      # list(zip(X, [data_name for data_name in X.index], y, X_grouplist))

      X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)


      dataset_cv = StratifiedKFold(n_splits=K, 
                                    shuffle = True, 
                                    random_state=np.random.RandomState(seed=split_shuffle_seed))

      # Model-id Prefix

      #################################################################################################################
      k_validation_results = []
      k_results = defaultdict(list)
      avg_validation_results = {}

      for train_idx, validation_idx in dataset_cv.split(X, y = X_grouplist):  # JY @ 2023-06-27 : y = X_grouplist is crucial for double-stratification
