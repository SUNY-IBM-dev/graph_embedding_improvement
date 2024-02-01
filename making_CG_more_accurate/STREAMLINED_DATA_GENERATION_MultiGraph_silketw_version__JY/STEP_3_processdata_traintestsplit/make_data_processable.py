import pickle
import os
import shutil

#if __name__ == "__main__":

# def organize_and_rename_subgraphs_into_Benign_dir(main_dirpath):

#    benign_dirnames = [ d for d in os.listdir(main_dirpath) if d.startswith("general_log_collection") or d.startswith("etw") ] 

#    Path_which_datapreprocessor_looks_for = os.path.join(main_dirpath, "Benign")
#    if os.path.exists( Path_which_datapreprocessor_looks_for ):
#       shutil.rmtree( Path_which_datapreprocessor_looks_for )
#    os.mkdir( Path_which_datapreprocessor_looks_for )

#    for benign_dirname in benign_dirnames:

#       Benign_Sample_P3_dirnames = [d for d in os.listdir(os.path.join(main_dirpath,benign_dirname)) if d.startswith("Benign_Sample_P3") or d.startswith("SUBGRAPH_P3")]

#       for Benign_Sample_P3_dirname in Benign_Sample_P3_dirnames:

#          shutil.copytree( src= os.path.join( main_dirpath , benign_dirname, Benign_Sample_P3_dirname ),
#                           dst= os.path.join( Path_which_datapreprocessor_looks_for, Benign_Sample_P3_dirname) )
#          print(f"copied {Benign_Sample_P3_dirname} from\n{os.path.join( main_dirpath , benign_dirname, Benign_Sample_P3_dirname)} to\n{os.path.join( Path_which_datapreprocessor_looks_for, Benign_Sample_P3_dirname)}")      




def organize_and_rename_subgraphs_into_LABEL_dir(main_dirpath, dir_start_pattern, label):

   if label.lower() == "malware":
      dirnames = [ d for d in os.listdir(main_dirpath) if d.startswith(dir_start_pattern)]
   elif label.lower() == "benign":
      dirnames = [ d for d in os.listdir(main_dirpath) if d.startswith(dir_start_pattern)]

   Path_which_datapreprocessor_looks_for = os.path.join(main_dirpath, label)
   if not os.path.exists( Path_which_datapreprocessor_looks_for ):
      # shutil.rmtree( Path_which_datapreprocessor_looks_for )
      os.makedirs( Path_which_datapreprocessor_looks_for )

   for malware_dirname in dirnames:

      Malware_Sample_dirnames = [d for d in os.listdir(os.path.join(main_dirpath,malware_dirname)) if d.startswith("SUBGRAPH_P3")]

      for Malware_Sample_dirname in Malware_Sample_dirnames:

         if label == "Benign":
            Final_Sample_dirname = Malware_Sample_dirname.replace("SUBGRAPH_P3", "SUBGRAPH_P3")
         else:
            Final_Sample_dirname = Malware_Sample_dirname


         shutil.copytree( src= os.path.join( main_dirpath , malware_dirname, Malware_Sample_dirname ),
                          dst= os.path.join( Path_which_datapreprocessor_looks_for, Final_Sample_dirname) )
         print(f"copied {Malware_Sample_dirname} from\n{os.path.join( main_dirpath , malware_dirname, Malware_Sample_dirname)} to\n{os.path.join( Path_which_datapreprocessor_looks_for, Final_Sample_dirname )}")      

