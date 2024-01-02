import os
import re

if __name__ == "__main__":

   # Set Parameters
   Train_ratio = 0.8
   
   read_save_dir = "/data/d1/jgwak1/tabby/NEW_Projection3_Datasets_Based_On_Modified_CG_Code__20230124"
   #-----------------------------
   Test_ratio = 1 - Train_ratio

   ##################################################################################################################################
   # MALWARE
   malware_prefix = "Processed_Malware_Sample_"


   Processed_Malware_dir = f"{read_save_dir}/Processed_Malware"
   Processed_Malware_pickle_fnames = os.listdir(Processed_Malware_dir)

   Stripped_Malware_All_Indices = [x.removeprefix( malware_prefix ).removesuffix(".pickle") for x in Processed_Malware_pickle_fnames]
   
   mal_indices = [x for x in Stripped_Malware_All_Indices if re.findall('mal\d+', x ) ]
   malware_sample_indices = [ x for x in Stripped_Malware_All_Indices if re.findall('malware_sample\d+', x ) ]
   malware_indices = [ x for x in Stripped_Malware_All_Indices if re.findall('malware\d+', x ) ]
   
   Sorted_mal_indices = sorted(mal_indices, key = lambda x: int( re.findall('\d+', x )[0] ) )
   Sorted_malware_sample_indices = sorted(malware_sample_indices, key = lambda x: int( re.findall('\d+', x )[0] ) )
   Sorted_malware_indices = sorted(malware_indices, key = lambda x: int( re.findall('\d+', x )[0] ) )

   #--------------------------------------------------------------------------------------------------------------------------------
   Malware_All_Indices = Sorted_mal_indices + Sorted_malware_sample_indices + Sorted_malware_indices

   number_of_ALL_Malware_subgraphs = len(Malware_All_Indices)
   number_of_Test_Malware_subgraphs = int(number_of_ALL_Malware_subgraphs * 0.2)

   Malware_Test_Indices = Sorted_malware_indices[ 0 : number_of_Test_Malware_subgraphs ]
   Malware_Train_Indices = Sorted_mal_indices + Sorted_malware_sample_indices + Sorted_malware_indices[ number_of_Test_Malware_subgraphs : ]

   ''' These INDEX_LIST files can be used for later use such as RF-Ngram.'''
   with open(f"{read_save_dir}/MALWARE_ALL_INDEX_LIST.txt", "w") as wf:
      [wf.write(f"{idx}\n") for idx in Malware_All_Indices]
      
   with open(f"{read_save_dir}/MALWARE_TEST_INDEX_LIST.txt", "w") as wf:
      [wf.write(f"{idx}\n") for idx in Malware_Test_Indices]

   with open(f"{read_save_dir}/MALWARE_TRAIN_INDEX_LIST.txt", "w") as wf:
      [wf.write(f"{idx}\n") for idx in Malware_Train_Indices]


   ''' Write out Malware Pickle file lists '''
   with open(f"{read_save_dir}/MALWARE_ALL_PICKLE_FILES_LIST.txt", "w") as wf:
      [wf.write(f"{malware_prefix}{idx}.pickle\n") for idx in Malware_All_Indices]
      
   with open(f"{read_save_dir}/MALWARE_TEST_PICKLE_FILES_LIST.txt", "w") as wf:
      [wf.write(f"{malware_prefix}{idx}.pickle\n") for idx in Malware_Test_Indices]

   with open(f"{read_save_dir}/MALWARE_TRAIN_PICKLE_FILES_LIST.txt", "w") as wf:
      [wf.write(f"{malware_prefix}{idx}.pickle\n") for idx in Malware_Train_Indices]

      
   ##################################################################################################################################
   # BENIGN
   Processed_Benign_dir = f"{read_save_dir}/Processed_Benign"
   Processed_Benign_pickle_fnames = os.listdir(Processed_Benign_dir) # Note that the order is sorted in index-range.

   number_of_ALL_benign_subgraphs = len(Processed_Benign_pickle_fnames)
   number_of_Train_benign_subgraphs = int(number_of_ALL_benign_subgraphs * Train_ratio)

   Benign_Train_pickle_files = Processed_Benign_pickle_fnames[ : number_of_Train_benign_subgraphs ]
   Benign_Test_pickle_files = Processed_Benign_pickle_fnames[ number_of_Train_benign_subgraphs : ]

   ''' Write out Benign Pickle file lists '''

   with open(f"{read_save_dir}/BENIGN_ALL_PICKLE_FILES_LIST.txt", "w") as wf:
      [wf.write(f"{idx}\n") for idx in Processed_Benign_pickle_fnames]
      
   with open(f"{read_save_dir}/BENIGN_TRAIN_PICKLE_FILES_LIST.txt", "w") as wf:
      [wf.write(f"{idx}\n") for idx in Benign_Train_pickle_files]

   with open(f"{read_save_dir}/BENIGN_TEST_PICKLE_FILES_LIST.txt", "w") as wf:
      [wf.write(f"{idx}\n") for idx in Benign_Test_pickle_files]   