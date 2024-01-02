import os
import re
import shutil




if __name__ == "__main__":

   # Set Parameters -----------------------------------------------------------------------------------------------------------------------------------------


   #sourcedirs_useractivity_pickle_files= \
   #[
   #   '/data/d1/jgwak1/tabby/NEW_Projection3_Datasets_Based_On_Modified_CG_Code__20230124/NEW_BENIGN_USER_ACTIVITY_SUBGRPAHS_SET_1/Processed_Not_Normalized_Benign',
   #   '/data/d1/jgwak1/tabby/NEW_Projection3_Datasets_Based_On_Modified_CG_Code__20230124/NEW_BENIGN_USER_ACTIVITY_SUBGRPAHS_SET_2/Processed_Not_Normalized_Benign',
   #]
   #dest_train_dir = "/data/d1/jgwak1/tabby/NEW_Projection3_Datasets_Based_On_Modified_CG_Code__20230124/OFFLINE_TRAINING_NOT_NORMALIZED_BenignTrain666_BenignUser22_MalTrain416"
   #dest_test_dir = "/data/d1/jgwak1/tabby/NEW_Projection3_Datasets_Based_On_Modified_CG_Code__20230124/NOT_USED_FOR_OFFLINE_TRAINING_NOT_NORMALIZED__BenignTest167_BenignUser7_MalTest104"

   sourcedirs_useractivity_pickle_files= \
   [
      '/data/d1/jgwak1/tabby/NEW_Projection3_Datasets_Based_On_Modified_CG_Code__20230124/NEW_BENIGN_USER_ACTIVITY_SUBGRPAHS_SET_1/Processed_Benign_ONLY_TaskName_edgeattr',
      '/data/d1/jgwak1/tabby/NEW_Projection3_Datasets_Based_On_Modified_CG_Code__20230124/NEW_BENIGN_USER_ACTIVITY_SUBGRPAHS_SET_2/Processed_Benign_ONLY_TaskName_edgeattr',
      '/data/d1/jgwak1/tabby/NEW_Projection3_Datasets_Based_On_Modified_CG_Code__20230124/NEW_BENIGN_USER_ACTIVITY_SUBGRPAHS_SET_3/Processed_Benign_ONLY_TaskName_edgeattr',
      '/data/d1/jgwak1/tabby/NEW_Projection3_Datasets_Based_On_Modified_CG_Code__20230124/NEW_BENIGN_USER_ACTIVITY_SUBGRPAHS_SET_4/Processed_Benign_ONLY_TaskName_edgeattr'
   ]
   dest_train_dir = "/data/d1/jgwak1/tabby/NEW_Projection3_Datasets_Based_On_Modified_CG_Code__20230124/OFFLINE_TRAINING__BenignTrain666_BenignUser57_MalTrain416__edgeattr_only_TaskName"
   dest_test_dir = "/data/d1/jgwak1/tabby/NEW_Projection3_Datasets_Based_On_Modified_CG_Code__20230124/NOT_USED_FOR_OFFLINE_TRAINING__BenignTest167_BenignUser15_MalTest104__edgeattr_only_TaskName"

   TEST_PICKLE_FILENAMES = [
      'Processed_Benign_Sample_P3_benignuseractivity__googleearth_softwareinstallation.pickle',
      'Processed_Benign_Sample_P3_benignuseractivity__naverwhale_browser_softwareinstallation.pickle',
      'Processed_Benign_Sample_P3_benignuseractivity__rstudio_softwaredevelopment.pickle',
      'Processed_Benign_Sample_P3_benignuseractivity__firefox_interaction.pickle',
      'Processed_Benign_Sample_P3_benignuseractivity__chrome12__interactive--googlechrome(browser)--butbrowser-eventsnotprovided.pickle',
      'Processed_Benign_Sample_P3_benignuseractivity__chrome14__interactive--googlechrome(browser)--butbrowser-eventsnotprovided.pickle',
      'Processed_Benign_Sample_P3_benignuseractivity__vncviewer_interaction.pickle',
      'Processed_Benign_Sample_P3_benignuseractivity__youtubemusic10__interactive--youtubemusicdesktopapp(stream).pickle',
      'Processed_Benign_Sample_P3_benignuseractivity__vncviewer3__interactive--vncviewer(connecttopanther).pickle',
      'Processed_Benign_Sample_P3_benignuseractivity__paint10__interactive--mspaint(microsoftdefaultpaintapp).pickle',
      'Processed_Benign_Sample_P3_benignuseractivity__notepad10__interactive--notepad(microsoftdefaultpaintapp).pickle',
      'Processed_Benign_Sample_P3_benignuseractivity__notepad_interaction.pickle',
      'Processed_Benign_Sample_P3_benignuseractivity__hearthstone_game_play.pickle',
      'Processed_Benign_Sample_P3_benignuseractivity__plantsvszombies__interactive--playplantsvs.zombies(game).pickle',
      'Processed_Benign_Sample_P3_benignuseractivity__plantsvszombies10__interactive--playplantsvs.zombies(game).pickle'
   ]

   # TRAIN_PICKLE_FILENAMES = < REST are the training-files, no need to define >

   ###################################################################################################################################################
   
   # Get the mapping between 'useractivity_pickle_files_dirpath' and the 'pickle filenames' in it.
   useractivity_dir2files_mapping = dict()
   for sourcedir_useractivity_pickle_files in sourcedirs_useractivity_pickle_files:
      useractivity_dir2files_mapping[ sourcedir_useractivity_pickle_files ] = os.listdir(sourcedir_useractivity_pickle_files)


   # Distribute (copy) pickle files from 'source_dir' to 'train_dest_dir' and 'test_dest_dir'. 
   for sourcedir_useractivity_pickle_files in useractivity_dir2files_mapping:

      # If the dest-dir doesn't exist, make it.
      benign_source_dirname = os.path.split(sourcedir_useractivity_pickle_files)[1]   
      if not os.path.exists( os.path.join( dest_train_dir, benign_source_dirname )):
         os.makedirs( os.path.join( dest_train_dir, benign_source_dirname ) )
      if not os.path.exists( os.path.join( dest_test_dir, benign_source_dirname )):
         os.makedirs( os.path.join( dest_test_dir, benign_source_dirname ) )

      # iterate through each pickle-file from all available source-dirs
      for pickle_filename in useractivity_dir2files_mapping[sourcedir_useractivity_pickle_files]:

         if pickle_filename in TEST_PICKLE_FILENAMES:
            
            shutil.copyfile(src= os.path.join(sourcedir_useractivity_pickle_files, pickle_filename),
                            dst= os.path.join(dest_test_dir, benign_source_dirname, pickle_filename))
            print(f"copied {os.path.join(sourcedir_useractivity_pickle_files, pickle_filename)} to {os.path.join(dest_test_dir, benign_source_dirname, pickle_filename)}", flush= True)


         else: # pickle_file correspond to Train-data
            shutil.copyfile(src= os.path.join(sourcedir_useractivity_pickle_files, pickle_filename),
                            dst= os.path.join(dest_train_dir, benign_source_dirname, pickle_filename))
            print(f"copied {os.path.join(sourcedir_useractivity_pickle_files, pickle_filename)} to {os.path.join(dest_train_dir, benign_source_dirname, pickle_filename)}", flush= True)

