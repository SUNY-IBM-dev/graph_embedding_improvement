import os
import re
import shutil

'''
1. First Run "TrainTest_Split_helper" to get all 
   "MALWARE_TEST_INDEX_LIST.txt", "MALWARE_TRAIN_INDEX_LIST", "MALWARE_TEST_INDEX_LIST" and
   "BENIGN_ALL_PICKLE_FILES_LIST.txt", "BENIGN_TRAIN_PICKLE_FILES_LIST.txt", "BENIGN_TEST_PICKLE_FILES_LIST.txt".

2. Then run this file, which distributes the malware-train-pickle-files, malware-test-pickle-files and benign-train-pickle-files, benign-test-pickle-files
   to under "OFFLINE_TRAINING.../Processed_..Benign" and "NOT_USED_FOR_OFFLINE_TRAINING.../Processed_..Benign" 

'''




if __name__ == "__main__":

   # Set Parameters -----------------------------------------------------------------------------------------------------------------------------------------
   
   #benign_source_dirpath = "/data/d1/jgwak1/tabby/NEW_Projection3_Datasets_Based_On_Modified_CG_Code__20230124/Processed_Benign" 
   #malware_source_dirpath = "/data/d1/jgwak1/tabby/NEW_Projection3_Datasets_Based_On_Modified_CG_Code__20230124/Processed_Malware"
   #train_dest_dir = "/data/d1/jgwak1/tabby/NEW_Projection3_Datasets_Based_On_Modified_CG_Code__20230124/OFFLINE_TRAINING__BenignTrain540_BenignUser13_MalTrain412"
   #test_dest_dir = "/data/d1/jgwak1/tabby/NEW_Projection3_Datasets_Based_On_Modified_CG_Code__20230124/NOT_USED_FOR_OFFLINE_TRAINING__BenignTest293_BenignUser16_MalTest100"

   source_benign_dirpath = "/data/d1/jgwak1/tabby/NEW_Projection3_Datasets_Based_On_Modified_CG_Code__20230124/Processed_Benign_ONLY_TaskName_edgeattr" 
   source_malware_dirpath = "/data/d1/jgwak1/tabby/NEW_Projection3_Datasets_Based_On_Modified_CG_Code__20230124/Processed_Malware_ONLY_TaskName_edgeattr"
   dest_train_dir = "/data/d1/jgwak1/tabby/NEW_Projection3_Datasets_Based_On_Modified_CG_Code__20230124/OFFLINE_TRAINING__BenignTrain666_BenignUser57_MalTrain416__edgeattr_only_TaskName"
   dest_test_dir = "/data/d1/jgwak1/tabby/NEW_Projection3_Datasets_Based_On_Modified_CG_Code__20230124/NOT_USED_FOR_OFFLINE_TRAINING__BenignTest167_BenignUser15_MalTest104__edgeattr_only_TaskName"


   
   benign_train_pickle_files_list_path = "/data/d1/jgwak1/tabby/NEW_Projection3_Datasets_Based_On_Modified_CG_Code__20230124/BENIGN_TRAIN_PICKLE_FILES_LIST.txt"
   benign_test_pickle_files_list_path = "/data/d1/jgwak1/tabby/NEW_Projection3_Datasets_Based_On_Modified_CG_Code__20230124/BENIGN_TEST_PICKLE_FILES_LIST.txt"
   malware_train_pickle_files_list_path = "/data/d1/jgwak1/tabby/NEW_Projection3_Datasets_Based_On_Modified_CG_Code__20230124/MALWARE_TRAIN_PICKLE_FILES_LIST.txt"
   malware_test_pickle_files_list_path = "/data/d1/jgwak1/tabby/NEW_Projection3_Datasets_Based_On_Modified_CG_Code__20230124/MALWARE_TEST_PICKLE_FILES_LIST.txt"  
   #-----------------------------
   # READ ALL NECESSARY PICKLE-FILE LISTS
   with open(benign_train_pickle_files_list_path, "r") as rf:
      benign_train_pickle_files_list = [ line.strip() for line in rf.readlines() ]

   with open(benign_test_pickle_files_list_path, "r") as rf:
      benign_test_pickle_files_list = [ line.strip() for line in rf.readlines() ]

   with open(malware_train_pickle_files_list_path, "r") as rf:
      malware_train_pickle_files_list = [ line.strip() for line in rf.readlines() ]

   with open(malware_test_pickle_files_list_path, "r") as rf:
      malware_test_pickle_files_list = [ line.strip() for line in rf.readlines() ]
   #-------------------------------

   benign_source_dirname = os.path.split(source_benign_dirpath)[1]   
   malware_source_dirname = os.path.split(source_malware_dirpath)[1]
 
   # Distribute (copy) pickle files from 'source_dir' to 'train_dest_dir' and 'test_dest_dir'. 

   # bengin-train ---------------------------------------------------------------------------------------------
   print("START copying Benign-Train"+"-"*50, flush= True)
   if os.path.exists( os.path.join(dest_train_dir, benign_source_dirname )):
      shutil.rmtree( os.path.join(dest_train_dir, benign_source_dirname ))
   os.makedirs( os.path.join(dest_train_dir, benign_source_dirname ) )

   for benign_train_pickle_file in benign_train_pickle_files_list:
      shutil.copyfile(src= os.path.join(source_benign_dirpath, benign_train_pickle_file),
                      dst= os.path.join(dest_train_dir, benign_source_dirname, benign_train_pickle_file))
      print(f"copied {os.path.join(source_benign_dirpath, benign_train_pickle_file)} to {os.path.join(dest_train_dir, benign_source_dirname, benign_train_pickle_file)}", flush= True)

   # bengin-test ---------------------------------------------------------------------------------------------
   print("START copying Benign-Test"+"-"*50, flush= True)
   if os.path.exists( os.path.join(dest_test_dir, benign_source_dirname )):
      shutil.rmtree( os.path.join(dest_test_dir, benign_source_dirname ))
   os.makedirs( os.path.join(dest_test_dir, benign_source_dirname ) )

   for benign_test_pickle_file in benign_test_pickle_files_list:
      shutil.copyfile(src= os.path.join(source_benign_dirpath, benign_test_pickle_file),
                      dst= os.path.join(dest_test_dir, benign_source_dirname, benign_test_pickle_file))
      print(f"copied {os.path.join(source_benign_dirpath, benign_test_pickle_file)} to {os.path.join(dest_test_dir, benign_source_dirname, benign_test_pickle_file)}", flush= True)

   # malware-train ---------------------------------------------------------------------------------------------
   print("START copying Malware-Train"+"-"*50, flush= True)
   if os.path.exists( os.path.join(dest_train_dir, malware_source_dirname )):
      shutil.rmtree( os.path.join(dest_train_dir, malware_source_dirname ))
   os.makedirs( os.path.join(dest_train_dir, malware_source_dirname ) )

   for malware_train_pickle_file in malware_train_pickle_files_list:
      shutil.copyfile(src= os.path.join(source_malware_dirpath, malware_train_pickle_file),
                      dst= os.path.join(dest_train_dir, malware_source_dirname, malware_train_pickle_file))
      print(f"copied {os.path.join(source_malware_dirpath, malware_train_pickle_file)} to {os.path.join(dest_train_dir, malware_source_dirname, malware_train_pickle_file)}", flush= True)

   # malware-test ---------------------------------------------------------------------------------------------
   print("START copying Malware-Test"+"-"*50, flush= True)
   if os.path.exists( os.path.join(dest_test_dir, malware_source_dirname )):
      shutil.rmtree( os.path.join(dest_test_dir, malware_source_dirname ))
   os.makedirs( os.path.join(dest_test_dir, malware_source_dirname ) )

   for malware_test_pickle_file in malware_test_pickle_files_list:
      shutil.copyfile(src= os.path.join(source_malware_dirpath, malware_test_pickle_file),
                      dst= os.path.join(dest_test_dir, malware_source_dirname, malware_test_pickle_file))
      print(f"copied {os.path.join(source_malware_dirpath, malware_test_pickle_file)} to {os.path.join(dest_test_dir, malware_source_dirname, malware_test_pickle_file)}", flush= True)

   print("DONE")

