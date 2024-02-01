'''

1.
split 
'/data/d1/jgwak1/tabby/graph_embedding_improvement_JY_git/making_CG_more_accurate/Subgraphs/Dataset_3_Benign/Indices/Processed_Benign_ONLY_TaskName_edgeattr' 
same as in 
/data/d1/jgwak1/tabby/Graph_embedding_aka_signal_amplification_files/Non_trace_commad_benign_dataset/train/Processed_Benign_ONLY_TaskName_edgeattr
/data/d1/jgwak1/tabby/Graph_embedding_aka_signal_amplification_files/Non_trace_commad_benign_dataset/test/Processed_Benign_ONLY_TaskName_edgeattr

2. 
split 
/data/d1/jgwak1/tabby/graph_embedding_improvement_JY_git/making_CG_more_accurate/Subgraphs/Dataset_3_Malware/Indices/Processed_Malware_ONLY_TaskName_edgeattr
same as in 
/data/d1/jgwak1/tabby/Graph_embedding_aka_signal_amplification_files/Non_trace_command_malware_dataset/train/Processed_Malware_ONLY_TaskName_edgeattr
/data/d1/jgwak1/tabby/Graph_embedding_aka_signal_amplification_files/Non_trace_command_malware_dataset/test/Processed_Malware_ONLY_TaskName_edgeattr
'''

import os
import shutil

if __name__ == "__main__":


   original_split_benign_train__dpath = "/data/d1/jgwak1/tabby/Graph_embedding_aka_signal_amplification_files/Non_trace_commad_benign_dataset/train/Processed_Benign_ONLY_TaskName_edgeattr"
   original_split_malware_train__dpath = "/data/d1/jgwak1/tabby/Graph_embedding_aka_signal_amplification_files/Non_trace_command_malware_dataset/train/Processed_Malware_ONLY_TaskName_edgeattr"

   original_split_benign_test__dpath = "/data/d1/jgwak1/tabby/Graph_embedding_aka_signal_amplification_files/Non_trace_commad_benign_dataset/test/Processed_Benign_ONLY_TaskName_edgeattr"
   original_split_malware_test__dpath = "/data/d1/jgwak1/tabby/Graph_embedding_aka_signal_amplification_files/Non_trace_command_malware_dataset/test/Processed_Malware_ONLY_TaskName_edgeattr"
   

   all_benign_pickles_to_distribute__dpath = "/data/d1/jgwak1/tabby/graph_embedding_improvement_JY_git/making_CG_more_accurate/Subgraphs/Dataset_3_Benign/Indices/Processed_Benign_ONLY_TaskName_edgeattr"
   all_malware_pickles_to_distribute__dpath = "/data/d1/jgwak1/tabby/graph_embedding_improvement_JY_git/making_CG_more_accurate/Subgraphs/Dataset_3_Malware/Indices/Processed_Malware_ONLY_TaskName_edgeattr"
 
   destination_benign_train__dpath = "/data/d1/jgwak1/tabby/graph_embedding_improvement_JY_git/making_CG_more_accurate/Subgraphs/Dataset_3_Benign/train"
   destination_malware_train__dpath = "/data/d1/jgwak1/tabby/graph_embedding_improvement_JY_git/making_CG_more_accurate/Subgraphs/Dataset_3_Malware/train"

   destination_benign_test__dpath = "/data/d1/jgwak1/tabby/graph_embedding_improvement_JY_git/making_CG_more_accurate/Subgraphs/Dataset_3_Benign/test"
   destination_malware_test__dpath = "/data/d1/jgwak1/tabby/graph_embedding_improvement_JY_git/making_CG_more_accurate/Subgraphs/Dataset_3_Malware/test"   

   # ----------------------------------------------------------------------------------------
   original_split_benign_train__fnames = [f for f in os.listdir(original_split_benign_train__dpath) if 'Processed' in f]
   original_split_malware_train__fnames = [f for f in os.listdir(original_split_malware_train__dpath) if 'Processed' in f]

   original_split_benign_test__fnames = [f for f in os.listdir(original_split_benign_test__dpath) if 'Processed' in f]
   original_split_malware_test__fnames = [f for f in os.listdir(original_split_malware_test__dpath) if 'Processed' in f]


   # distribute benign_train
   for benign_train_fname in original_split_benign_train__fnames:

      src = os.path.join(all_benign_pickles_to_distribute__dpath, benign_train_fname)
      dest = os.path.join(destination_benign_train__dpath, benign_train_fname)
      shutil.copy(source = src, 
                  destination = dest)
      print(f"{benign_train_fname} copied successfully from\n{src}\nto\n{dest}.", flush = True)


   # distribute benign_test
   for benign_test_fname in original_split_benign_test__fnames:

      src = os.path.join(all_benign_pickles_to_distribute__dpath, benign_test_fname)
      dest = os.path.join(destination_benign_test__dpath, benign_test_fname)
      shutil.copy(source = src, 
                  destination = dest)
      print(f"{benign_test_fname} copied successfully from\n{src}\nto\n{dest}.", flush = True)      


   # distribute malware_train
   for malware_train_fname in original_split_malware_train__fnames:

      src = os.path.join(all_malware_pickles_to_distribute__dpath, malware_train_fname)
      dest = os.path.join(destination_malware_train__dpath, malware_train_fname)
      shutil.copy(source = src, 
                  destination = dest)
      print(f"{malware_train_fname} copied successfully from\n{src}\nto\n{dest}.", flush = True)      


   # distribute malware_test
   for malware_test_fname in original_split_malware_test__fnames:
      src = os.path.join(all_malware_pickles_to_distribute__dpath, malware_test_fname)
      dest = os.path.join(destination_malware_test__dpath, malware_test_fname)
      shutil.copy(source = src, 
                  destination = dest)
      print(f"{malware_test_fname} copied successfully from\n{src}\nto\n{dest}.", flush = True)      


   print("done", flush=True)