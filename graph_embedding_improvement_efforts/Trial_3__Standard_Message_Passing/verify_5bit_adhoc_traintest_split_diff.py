import os 

if __name__ == "__main__":

   # dataset-1
   dataset_1_benign_train__5bit = set(os.listdir("/data/d1/jgwak1/tabby/SILKETW_DATASET_NEW/Silketw_benign_train_test_data_case1/offline_train/Processed_Benign_ONLY_TaskName_edgeattr"))
   dataset_1_benign_train__adhoc = set(os.listdir("/data/d1/jgwak1/tabby/SILKETW_DATASET_NEW/Benign_case1/train"))

   dataset_1_malware_train__5bit = set(os.listdir("/data/d1/jgwak1/tabby/SILKETW_DATASET_NEW/Silketw_malware_train_test_data_case1/offline_train/Processed_Malware_ONLY_TaskName_edgeattr"))
   dataset_1_malware_train__adhoc = set(os.listdir("/data/d1/jgwak1/tabby/SILKETW_DATASET_NEW/Malware_case1/train"))

   dataset_1_benign_test__5bit = set(os.listdir("/data/d1/jgwak1/tabby/SILKETW_DATASET_NEW/Silketw_benign_train_test_data_case1/offline_test/Processed_Benign_ONLY_TaskName_edgeattr"))
   dataset_1_benign_test__adhoc = set(os.listdir("/data/d1/jgwak1/tabby/SILKETW_DATASET_NEW/Benign_case1/test"))

   dataset_1_malware_test__5bit = set(os.listdir("/data/d1/jgwak1/tabby/SILKETW_DATASET_NEW/Silketw_malware_train_test_data_case1/offline_test/Processed_Malware_ONLY_TaskName_edgeattr"))
   dataset_1_malware_test__adhoc = set(os.listdir("/data/d1/jgwak1/tabby/SILKETW_DATASET_NEW/Malware_case1/test"))

   # dataset-2 
   dataset_2_benign_train__5bit = set(os.listdir("/data/d1/jgwak1/tabby/SILKETW_DATASET_NEW/Silketw_benign_train_test_data_case1_case2/offline_train/Processed_Benign_ONLY_TaskName_edgeattr"))
   dataset_2_benign_train__adhoc = set(os.listdir("/data/d1/jgwak1/tabby/SILKETW_DATASET_NEW/Benign_case2/train"))

   dataset_2_malware_train__5bit = set(os.listdir("/data/d1/jgwak1/tabby/SILKETW_DATASET_NEW/Silketw_malware_train_test_data_case1_case2/offline_train/Processed_Malware_ONLY_TaskName_edgeattr"))
   dataset_2_malware_train__adhoc = set(os.listdir("/data/d1/jgwak1/tabby/SILKETW_DATASET_NEW/Malware_case2/train"))

   dataset_2_benign_test__5bit = set(os.listdir("/data/d1/jgwak1/tabby/SILKETW_DATASET_NEW/Silketw_benign_train_test_data_case1_case2/offline_test/Processed_Benign_ONLY_TaskName_edgeattr"))
   dataset_2_benign_test__adhoc = set(os.listdir("/data/d1/jgwak1/tabby/SILKETW_DATASET_NEW/Benign_case2/test"))

   dataset_2_malware_test__5bit = set(os.listdir("/data/d1/jgwak1/tabby/SILKETW_DATASET_NEW/Silketw_malware_train_test_data_case1_case2/offline_test/Processed_Malware_ONLY_TaskName_edgeattr"))
   dataset_2_malware_test__adhoc = set(os.listdir("/data/d1/jgwak1/tabby/SILKETW_DATASET_NEW/Malware_case2/test"))



   print()
   # dataset-1

   print( dataset_1_benign_train__5bit.difference(dataset_1_benign_train__adhoc) )
   print( dataset_1_malware_train__5bit.difference(dataset_1_malware_train__adhoc) )

   print(dataset_1_benign_test__5bit.difference(dataset_1_benign_test__adhoc) )
   print(dataset_1_malware_test__5bit.difference(dataset_1_malware_test__adhoc))

   # dataset-2 

   print(dataset_2_benign_train__5bit.difference(dataset_2_benign_train__adhoc))
   print(dataset_2_malware_train__5bit.difference(dataset_2_malware_train__adhoc))

   print(dataset_2_benign_test__5bit.difference(dataset_2_benign_test__adhoc))
   print(dataset_2_malware_test__5bit.difference(dataset_2_malware_test__adhoc))
