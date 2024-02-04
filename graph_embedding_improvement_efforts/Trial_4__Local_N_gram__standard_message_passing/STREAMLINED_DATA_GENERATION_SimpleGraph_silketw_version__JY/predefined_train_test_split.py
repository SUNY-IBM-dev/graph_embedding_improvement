
''' 
# JY @ 2024-1-2:

Based on the "Predefined Split" information, 
Copy pickle files from Source-Directories to their Target-Directories.

(code written to be compatible with how Priti manually splitted for dataset-2)

'''

import os
import shutil

if __name__ == "__main__":

   fnames_to_ignore = ["train_pickles_addition_record.txt",
                       "test_pickles_addition_record.txt"]

   # ========================================================================================================
   # PREDEFINED SPLITS (ONLY FOR SPLITS INFORMATION **) 
   #  Predefined Split Dataset-1   
   PredefinedSplit__Dataset_case1__benign_train__pickles_dirpath = "/home/jgwak1/tabby/PW_NON_TRACE_COMMAND_DATASET/Benign_Case1/train/Processed_Benign_ONLY_TaskName_edgeattr"
   PredefinedSplit__Dataset_case1__benign_final_test__pickles_dirpath = "/home/jgwak1/tabby/PW_NON_TRACE_COMMAND_DATASET/Benign_Case1/test/Processed_Benign_ONLY_TaskName_edgeattr"
   
   PredefinedSplit__Dataset_case1__malware_train__pickles_dirpath = "/home/jgwak1/tabby/PW_NON_TRACE_COMMAND_DATASET/Malware_Case1/train/Processed_Malware_ONLY_TaskName_edgeattr"
   PredefinedSplit__Dataset_case1__malware_final_test__pickles_dirpath = "/home/jgwak1/tabby/PW_NON_TRACE_COMMAND_DATASET/Malware_Case1/test/Processed_Malware_ONLY_TaskName_edgeattr"   
   
   #  Predefined Split Dataset-2
   PredefinedSplit__Dataset_case1_case2__benign_train__pickles_dirpath = "/home/jgwak1/tabby/PW_NON_TRACE_COMMAND_DATASET/Benign_Case2/train/Processed_Benign_ONLY_TaskName_edgeattr"
   PredefinedSplit__Dataset_case1_case2__benign_final_test__pickles_dirpath = "/home/jgwak1/tabby/PW_NON_TRACE_COMMAND_DATASET/Benign_Case2/test/Processed_Benign_ONLY_TaskName_edgeattr"
   
   PredefinedSplit__Dataset_case1_case2__malware_train__pickles_dirpath = "/home/jgwak1/tabby/PW_NON_TRACE_COMMAND_DATASET/Malware_Case2/train/Processed_Malware_ONLY_TaskName_edgeattr"
   PredefinedSplit__Dataset_case1_case2__malware_final_test__pickles_dirpath = "/home/jgwak1/tabby/PW_NON_TRACE_COMMAND_DATASET/Malware_Case2/test/Processed_Malware_ONLY_TaskName_edgeattr"      
   # ========================================================================================================
   # SOURCE DIRECTORIES (of pickle files to copy from)

   # Dataset-1
   Source__Dataset_case1__benign_ALL__pickles_dirpath = "/home/jgwak1/tabby/graph_embedding_improvement_JY_git/graph_embedding_improvement_efforts/Trial_4__Local_N_gram__standard_message_passing/Subgraphs__SimpleGraph/NON_TRACE_COMMAND_DATASET/Benign_Case1/Indices/Processed_Benign_ONLY_TaskName_edgeattr"
   Source__Dataset_case2__benign_ALL__pickles_dirpath = "/home/jgwak1/tabby/graph_embedding_improvement_JY_git/graph_embedding_improvement_efforts/Trial_4__Local_N_gram__standard_message_passing/Subgraphs__SimpleGraph/NON_TRACE_COMMAND_DATASET/Benign_Case2/Indices/Processed_Benign_ONLY_TaskName_edgeattr"

   # Dataset-2   
   Source__Dataset_case1__malware_ALL__pickles_dirpath = "/home/jgwak1/tabby/graph_embedding_improvement_JY_git/graph_embedding_improvement_efforts/Trial_4__Local_N_gram__standard_message_passing/Subgraphs__SimpleGraph/NON_TRACE_COMMAND_DATASET/Malware_Case1/Indices/Processed_Malware_ONLY_TaskName_edgeattr"
   Source__Dataset_case2__malware_ALL__pickles_dirpath = "/home/jgwak1/tabby/graph_embedding_improvement_JY_git/graph_embedding_improvement_efforts/Trial_4__Local_N_gram__standard_message_passing/Subgraphs__SimpleGraph/NON_TRACE_COMMAND_DATASET/Malware_Case2/Indices/Processed_Malware_ONLY_TaskName_edgeattr"

   # ========================================================================================================
   # TARGET DIRECTORIES (of pickle files to copy to)
   
   # Dataset-1   
   Target__Dataset_case1__benign_train__pickles_dirpath = "/home/jgwak1/tabby/graph_embedding_improvement_JY_git/graph_embedding_improvement_efforts/Trial_4__Local_N_gram__standard_message_passing/Subgraphs__SimpleGraph/NON_TRACE_COMMAND_DATASET/Benign_Case1/train"
   Target__Dataset_case1__benign_final_test__pickles_dirpath = "/home/jgwak1/tabby/graph_embedding_improvement_JY_git/graph_embedding_improvement_efforts/Trial_4__Local_N_gram__standard_message_passing/Subgraphs__SimpleGraph/NON_TRACE_COMMAND_DATASET/Benign_Case1/test"
   
   Target__Dataset_case1__malware_train__pickles_dirpath = "/home/jgwak1/tabby/graph_embedding_improvement_JY_git/graph_embedding_improvement_efforts/Trial_4__Local_N_gram__standard_message_passing/Subgraphs__SimpleGraph/NON_TRACE_COMMAND_DATASET/Malware_Case1/train"
   Target__Dataset_case1__malware_final_test__pickles_dirpath = "/home/jgwak1/tabby/graph_embedding_improvement_JY_git/graph_embedding_improvement_efforts/Trial_4__Local_N_gram__standard_message_passing/Subgraphs__SimpleGraph/NON_TRACE_COMMAND_DATASET/Malware_Case1/test"
   
   # Dataset-2
   Target__Dataset_case1_case2__benign_train__pickles_dirpath = "/home/jgwak1/tabby/graph_embedding_improvement_JY_git/graph_embedding_improvement_efforts/Trial_4__Local_N_gram__standard_message_passing/Subgraphs__SimpleGraph/NON_TRACE_COMMAND_DATASET/Benign_Case2/train" 
   Target__Dataset_case1_case2__benign_final_test__pickles_dirpath = "/home/jgwak1/tabby/graph_embedding_improvement_JY_git/graph_embedding_improvement_efforts/Trial_4__Local_N_gram__standard_message_passing/Subgraphs__SimpleGraph/NON_TRACE_COMMAND_DATASET/Benign_Case2/test" 
   
   Target__Dataset_case1_case2__malware_train__pickles_dirpath = "/home/jgwak1/tabby/graph_embedding_improvement_JY_git/graph_embedding_improvement_efforts/Trial_4__Local_N_gram__standard_message_passing/Subgraphs__SimpleGraph/NON_TRACE_COMMAND_DATASET/Malware_Case2/train" 
   Target__Dataset_case1_case2__malware_final_test__pickles_dirpath = "/home/jgwak1/tabby/graph_embedding_improvement_JY_git/graph_embedding_improvement_efforts/Trial_4__Local_N_gram__standard_message_passing/Subgraphs__SimpleGraph/NON_TRACE_COMMAND_DATASET/Malware_Case2/test" 

   # *******************************************************************************************************
   # *******************************************************************************************************
   # Perform copying over for Dataset-1 

   # 1. get predefined splits
   PredefinedSplit__Dataset_case1__benign_train__pickles = [f for f in os.listdir(PredefinedSplit__Dataset_case1__benign_train__pickles_dirpath) \
                                                            if f not in fnames_to_ignore]
   PredefinedSplit__Dataset_case1__benign_final_test__pickles = [f for f in os.listdir(PredefinedSplit__Dataset_case1__benign_final_test__pickles_dirpath) \
                                                                 if f not in fnames_to_ignore]
   PredefinedSplit__Dataset_case1__malware_train__pickles = [f for f in os.listdir(PredefinedSplit__Dataset_case1__malware_train__pickles_dirpath) \
                                                             if f not in fnames_to_ignore]
   PredefinedSplit__Dataset_case1__malware_final_test__pickles = [f for f in os.listdir(PredefinedSplit__Dataset_case1__malware_final_test__pickles_dirpath) \
                                                                 if f not in fnames_to_ignore]

   # 2. for each sample(pickle) in dataset-1 sources, copy over to target-dirs, based on predefined split
   
   Source__Dataset_case1__benign_ALL__pickles = [f for f in os.listdir(Source__Dataset_case1__benign_ALL__pickles_dirpath) \
                                                 if f not in fnames_to_ignore]
   Source__Dataset_case1__malware_ALL__pickles = [f for f in os.listdir(Source__Dataset_case1__malware_ALL__pickles_dirpath) \
                                                 if f not in fnames_to_ignore]   

   for benign_pickle in Source__Dataset_case1__benign_ALL__pickles:
      
      if benign_pickle in PredefinedSplit__Dataset_case1__benign_train__pickles:
         shutil.copyfile(src= os.path.join(Source__Dataset_case1__benign_ALL__pickles_dirpath, benign_pickle),
                         dst= os.path.join(Target__Dataset_case1__benign_train__pickles_dirpath, benign_pickle)
                         )

      elif benign_pickle in PredefinedSplit__Dataset_case1__benign_final_test__pickles:
         shutil.copyfile(src= os.path.join(Source__Dataset_case1__benign_ALL__pickles_dirpath, benign_pickle),
                         dst= os.path.join(Target__Dataset_case1__benign_final_test__pickles_dirpath, benign_pickle)
                         )
      else:
         print(f"{benign_pickle} not in either \n{PredefinedSplit__Dataset_case1__benign_train__pickles_dirpath}\n{PredefinedSplit__Dataset_case1__benign_final_test__pickles_dirpath}\n", flush = True)
      

   for malware_pickle in Source__Dataset_case1__malware_ALL__pickles:

      if malware_pickle in PredefinedSplit__Dataset_case1__malware_train__pickles:
         shutil.copyfile(src= os.path.join(Source__Dataset_case1__malware_ALL__pickles_dirpath, malware_pickle),
                         dst= os.path.join(Target__Dataset_case1__malware_train__pickles_dirpath, malware_pickle)
                         )

      elif malware_pickle in PredefinedSplit__Dataset_case1__malware_final_test__pickles:
         shutil.copyfile(src= os.path.join(Source__Dataset_case1__malware_ALL__pickles_dirpath, malware_pickle),
                         dst= os.path.join(Target__Dataset_case1__malware_final_test__pickles_dirpath, malware_pickle)
                         )
      else:
         print(f"{malware_pickle} not in either \n{PredefinedSplit__Dataset_case1__malware_train__pickles_dirpath}\n{PredefinedSplit__Dataset_case1__malware_final_test__pickles_dirpath}\n", flush = True)
      

   # *******************************************************************************************************
   # *******************************************************************************************************
   # Perform copying over for Dataset-2 (case-1, case-2)
   # 1. get predefined splits
   PredefinedSplit__Dataset_case1_case2__benign_train__pickles = [f for f in os.listdir(PredefinedSplit__Dataset_case1_case2__benign_train__pickles_dirpath) \
                                                                  if f not in fnames_to_ignore]
   PredefinedSplit__Dataset_case1_case2__benign_final_test__pickles = [f for f in os.listdir(PredefinedSplit__Dataset_case1_case2__benign_final_test__pickles_dirpath) \
                                                                       if f not in fnames_to_ignore]
   PredefinedSplit__Dataset_case1_case2__malware_train__pickles = [f for f in os.listdir(PredefinedSplit__Dataset_case1_case2__malware_train__pickles_dirpath) \
                                                                   if f not in fnames_to_ignore]
   PredefinedSplit__Dataset_case1_case2__malware_final_test__pickles = [f for f in os.listdir(PredefinedSplit__Dataset_case1_case2__malware_final_test__pickles_dirpath) \
                                                                        if f not in fnames_to_ignore]

   # 2. for each sample(pickle) in dataset-1 sources, copy over to target-dirs, based on predefined split

   Source__Dataset_case1__benign_ALL__pickles = [f for f in os.listdir(Source__Dataset_case1__benign_ALL__pickles_dirpath) \
                                                 if f not in fnames_to_ignore]
   Source__Dataset_case2__benign_ALL__pickles = [f for f in os.listdir(Source__Dataset_case2__benign_ALL__pickles_dirpath) \
                                                 if f not in fnames_to_ignore]   
   
   Source__Dataset_case1__malware_ALL__pickles = [f for f in os.listdir(Source__Dataset_case1__malware_ALL__pickles_dirpath) \
                                                 if f not in fnames_to_ignore]   
   Source__Dataset_case2__malware_ALL__pickles = [f for f in os.listdir(Source__Dataset_case2__malware_ALL__pickles_dirpath) \
                                                 if f not in fnames_to_ignore]      


   for benign_pickle in Source__Dataset_case1__benign_ALL__pickles + Source__Dataset_case2__benign_ALL__pickles:
      
      # figure out whether benign_pickle comes from case1 or case2
      if benign_pickle in Source__Dataset_case1__benign_ALL__pickles:
         source_pickle_dirpath = Source__Dataset_case1__benign_ALL__pickles_dirpath
      elif benign_pickle in Source__Dataset_case2__benign_ALL__pickles:
         source_pickle_dirpath = Source__Dataset_case2__benign_ALL__pickles_dirpath

      # copy 
      if benign_pickle in PredefinedSplit__Dataset_case1_case2__benign_train__pickles:
         shutil.copyfile(src= os.path.join(source_pickle_dirpath, benign_pickle),
                         dst= os.path.join(Target__Dataset_case1_case2__benign_train__pickles_dirpath, benign_pickle)
                         )
         

      elif benign_pickle in PredefinedSplit__Dataset_case1_case2__benign_final_test__pickles:
         shutil.copyfile(src= os.path.join(source_pickle_dirpath, benign_pickle),
                         dst= os.path.join(Target__Dataset_case1_case2__benign_final_test__pickles_dirpath, benign_pickle)
                         )
      else:
         print(f"{benign_pickle} not in either \n{PredefinedSplit__Dataset_case1_case2__benign_train__pickles_dirpath}\n{PredefinedSplit__Dataset_case1_case2__benign_final_test__pickles_dirpath}\n", flush = True)
      

   for malware_pickle in Source__Dataset_case1__malware_ALL__pickles + Source__Dataset_case2__malware_ALL__pickles:

      # figure out whether malware_pickle comes from case1 or case2
      if malware_pickle in Source__Dataset_case1__malware_ALL__pickles:
         source_pickle_dirpath = Source__Dataset_case1__malware_ALL__pickles_dirpath
      elif malware_pickle in Source__Dataset_case2__malware_ALL__pickles:
         source_pickle_dirpath = Source__Dataset_case2__malware_ALL__pickles_dirpath

      # copy 
      if malware_pickle in PredefinedSplit__Dataset_case1_case2__malware_train__pickles:
         shutil.copyfile(src= os.path.join(source_pickle_dirpath, malware_pickle),
                         dst= os.path.join(Target__Dataset_case1_case2__malware_train__pickles_dirpath, malware_pickle)
                         )

      elif malware_pickle in PredefinedSplit__Dataset_case1_case2__malware_final_test__pickles:
         shutil.copyfile(src= os.path.join(source_pickle_dirpath, malware_pickle),
                         dst= os.path.join(Target__Dataset_case1_case2__malware_final_test__pickles_dirpath, malware_pickle)
                         )
      else:
         print(f"{malware_pickle} not in either \n{PredefinedSplit__Dataset_case1_case2__malware_train__pickles_dirpath}\n{PredefinedSplit__Dataset_case1_case2__malware_final_test__pickles_dirpath}\n", flush = True)
      

   # *******************************************************************************************************
   # *******************************************************************************************************
   # Verification
   assert set(PredefinedSplit__Dataset_case1__benign_train__pickles) == set(os.listdir(Target__Dataset_case1__benign_train__pickles_dirpath))
   assert set(PredefinedSplit__Dataset_case1__benign_final_test__pickles) == set(os.listdir(Target__Dataset_case1__benign_final_test__pickles_dirpath))

   assert set(PredefinedSplit__Dataset_case1__malware_train__pickles) == set(os.listdir(Target__Dataset_case1__malware_train__pickles_dirpath))
   assert set(PredefinedSplit__Dataset_case1__malware_final_test__pickles) == set(os.listdir(Target__Dataset_case1__malware_final_test__pickles_dirpath))

   assert set(PredefinedSplit__Dataset_case1_case2__benign_train__pickles) == set(os.listdir(Target__Dataset_case1_case2__benign_train__pickles_dirpath))
   assert set(PredefinedSplit__Dataset_case1_case2__benign_final_test__pickles) == set(os.listdir(Target__Dataset_case1_case2__benign_final_test__pickles_dirpath))

   assert set(PredefinedSplit__Dataset_case1_case2__malware_train__pickles) == set(os.listdir(Target__Dataset_case1_case2__malware_train__pickles_dirpath))
   assert set(PredefinedSplit__Dataset_case1_case2__malware_final_test__pickles) == set(os.listdir(Target__Dataset_case1_case2__malware_final_test__pickles_dirpath))


