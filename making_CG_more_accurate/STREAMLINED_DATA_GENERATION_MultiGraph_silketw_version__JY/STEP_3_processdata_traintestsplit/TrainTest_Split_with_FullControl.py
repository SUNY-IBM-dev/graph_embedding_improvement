import os
import shutil

if __name__ == "__main__":
    
   #source_dirpath = "/data/d1/jgwak1/tabby/NEW_Projection3_Datasets_Based_On_Modified_CG_Code__20230124/GENERAL_AND_NORMAL_LOG_COLLECTION_SUBGRAPHS_20230202/Processed_Benign_ONLY_TaskName_edgeattr"
   source_dirpath = "/data/d1/jgwak1/tabby/NEW_Projection3_Datasets_Based_On_Modified_CG_Code__20230124/GENERAL_LOG_COLLECTION_SUBGRAPHS_20230131/Processed_Benign_ONLY_TaskName_edgeattr"
   
   key_dirname = os.path.split(os.path.split(source_dirpath)[0])[1]

   train_dest_dirpath = "/data/d1/jgwak1/tabby/NEW_Projection3_Datasets_Based_On_Modified_CG_Code__20230124/OFFLINE_TRAINING__BenignTrain666_BenignUser57_MalTrain416__edgeattr_only_TaskName/Processed_Benign_ONLY_TaskName_edgeattr"
   test_dest_dirpath = "/data/d1/jgwak1/tabby/NEW_Projection3_Datasets_Based_On_Modified_CG_Code__20230124/NOT_USED_FOR_OFFLINE_TRAINING__BenignTest167_BenignUser15_MalTest104__edgeattr_only_TaskName/Processed_Benign_ONLY_TaskName_edgeattr"

   Train_ratio = 0.8
   Test_ratio = 1 - Train_ratio    
    
   source_pickle_filenames = [ pklfname for pklfname in os.listdir(source_dirpath) if pklfname.startswith("Processed")] 

   number_of_picklefiles_in_sourcedir = len(source_pickle_filenames)
   
   first_X_for_training = int(number_of_picklefiles_in_sourcedir * Train_ratio)

   train_pickles = source_pickle_filenames[:first_X_for_training]
   test_pickles = source_pickle_filenames[first_X_for_training:]
   
   read_save_dir = "/data/d1/jgwak1/tabby/NEW_Projection3_Datasets_Based_On_Modified_CG_Code__20230124"
   with open(f"{read_save_dir}/BENIGN_NEW_TRAIN_INDEX_LIST__{key_dirname}.txt", "w") as wf:
          [wf.write(f"{idx}\n") for idx in train_pickles]
   with open(f"{read_save_dir}/BENIGN_NEW_TEST_INDEX_LIST__{key_dirname}.txt", "w") as wf:
          [wf.write(f"{idx}\n") for idx in test_pickles]

   
   for train_pkl in train_pickles:
        shutil.copyfile(src= os.path.join(source_dirpath, train_pkl),
                        dst= os.path.join(train_dest_dirpath, train_pkl))
        print(f"copied {os.path.join(source_dirpath, train_pkl)} to {os.path.join(train_dest_dirpath, train_pkl)}", flush= True)
       
   for test_pkl in test_pickles:
        shutil.copyfile(src= os.path.join(source_dirpath, test_pkl),
                        dst= os.path.join(test_dest_dirpath, test_pkl))
        print(f"copied {os.path.join(source_dirpath, test_pkl)} to {os.path.join(test_dest_dirpath, test_pkl)}", flush= True)
              
   print("\n\n\n\n")
   print(f"copying done for {source_dirpath}")
   print(f"len(source_pickle_filenames):{len(source_pickle_filenames)}")
   print(f"len(train_pickles): {len(train_pickles)}")
   print(f"len(test_pickles): {len(test_pickles)}")