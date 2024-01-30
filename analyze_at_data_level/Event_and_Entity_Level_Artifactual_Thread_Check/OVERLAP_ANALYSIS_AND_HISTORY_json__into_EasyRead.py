import os
import json


if __name__ == "__main__":

   Save_EasyRead_TXTs_dirpath = "indices__intersecting_entities__to__pidtid__used_node_identifier_True___WITH_OVERLAP_ANALYSIS_AND_HISTORY___Easy"

   OVERLAP_ANALYSIS_AND_HISTORY__SPLIT__json_filepaths = \
      [  # benign-splits
         "/home/jgwak1/tabby/graph_embedding_improvement_JY_git/analyze_at_data_level/Event_and_Entity_Level_Artifactual_Thread_Check/RESULTS/Event_and_Entity_Level_Artifactual_Thread_Check__Dataset-Case-1__2024-01-24_134317/indices__intersecting_entities__to__pidtid__used_node_identifier_True___WITH_OVERLAP_ANALYSIS_AND_HISTORY___BENIGN__SPLIT_0.json",
         "/home/jgwak1/tabby/graph_embedding_improvement_JY_git/analyze_at_data_level/Event_and_Entity_Level_Artifactual_Thread_Check/RESULTS/Event_and_Entity_Level_Artifactual_Thread_Check__Dataset-Case-1__2024-01-24_134317/indices__intersecting_entities__to__pidtid__used_node_identifier_True___WITH_OVERLAP_ANALYSIS_AND_HISTORY___BENIGN__SPLIT_1.json",
         "/home/jgwak1/tabby/graph_embedding_improvement_JY_git/analyze_at_data_level/Event_and_Entity_Level_Artifactual_Thread_Check/RESULTS/Event_and_Entity_Level_Artifactual_Thread_Check__Dataset-Case-1__2024-01-24_134317/indices__intersecting_entities__to__pidtid__used_node_identifier_True___WITH_OVERLAP_ANALYSIS_AND_HISTORY___BENIGN__SPLIT_2.json",
         "/home/jgwak1/tabby/graph_embedding_improvement_JY_git/analyze_at_data_level/Event_and_Entity_Level_Artifactual_Thread_Check/RESULTS/Event_and_Entity_Level_Artifactual_Thread_Check__Dataset-Case-1__2024-01-24_134317/indices__intersecting_entities__to__pidtid__used_node_identifier_True___WITH_OVERLAP_ANALYSIS_AND_HISTORY___BENIGN__SPLIT_3.json",
         "/home/jgwak1/tabby/graph_embedding_improvement_JY_git/analyze_at_data_level/Event_and_Entity_Level_Artifactual_Thread_Check/RESULTS/Event_and_Entity_Level_Artifactual_Thread_Check__Dataset-Case-1__2024-01-24_134317/indices__intersecting_entities__to__pidtid__used_node_identifier_True___WITH_OVERLAP_ANALYSIS_AND_HISTORY___BENIGN__SPLIT_4.json",
         # malware splits
         "/home/jgwak1/tabby/graph_embedding_improvement_JY_git/analyze_at_data_level/Event_and_Entity_Level_Artifactual_Thread_Check/RESULTS/Event_and_Entity_Level_Artifactual_Thread_Check__Dataset-Case-1__2024-01-24_134317/indices__intersecting_entities__to__pidtid__used_node_identifier_True___WITH_OVERLAP_ANALYSIS_AND_HISTORY___MALWARE__SPLIT_0.json",
         "/home/jgwak1/tabby/graph_embedding_improvement_JY_git/analyze_at_data_level/Event_and_Entity_Level_Artifactual_Thread_Check/RESULTS/Event_and_Entity_Level_Artifactual_Thread_Check__Dataset-Case-1__2024-01-24_134317/indices__intersecting_entities__to__pidtid__used_node_identifier_True___WITH_OVERLAP_ANALYSIS_AND_HISTORY___MALWARE__SPLIT_1.json",
         "/home/jgwak1/tabby/graph_embedding_improvement_JY_git/analyze_at_data_level/Event_and_Entity_Level_Artifactual_Thread_Check/RESULTS/Event_and_Entity_Level_Artifactual_Thread_Check__Dataset-Case-1__2024-01-24_134317/indices__intersecting_entities__to__pidtid__used_node_identifier_True___WITH_OVERLAP_ANALYSIS_AND_HISTORY___MALWARE__SPLIT_2.json",
         "/home/jgwak1/tabby/graph_embedding_improvement_JY_git/analyze_at_data_level/Event_and_Entity_Level_Artifactual_Thread_Check/RESULTS/Event_and_Entity_Level_Artifactual_Thread_Check__Dataset-Case-1__2024-01-24_134317/indices__intersecting_entities__to__pidtid__used_node_identifier_True___WITH_OVERLAP_ANALYSIS_AND_HISTORY___MALWARE__SPLIT_3.json",
         "/home/jgwak1/tabby/graph_embedding_improvement_JY_git/analyze_at_data_level/Event_and_Entity_Level_Artifactual_Thread_Check/RESULTS/Event_and_Entity_Level_Artifactual_Thread_Check__Dataset-Case-1__2024-01-24_134317/indices__intersecting_entities__to__pidtid__used_node_identifier_True___WITH_OVERLAP_ANALYSIS_AND_HISTORY___MALWARE__SPLIT_4.json",
      ]

   

   for json_filepath in OVERLAP_ANALYSIS_AND_HISTORY__SPLIT__json_filepaths:

      with open( json_filepath, "r" ) as file:
         OVERLAP_ANALYSIS_AND_HISTORY__SPLIT__dict = json.load(file)
      
      TO_SAVE_Dirpath = os.path.split(json_filepath)[0]
      EasyRead_txt_filename = f"{os.path.split(json_filepath)[1].removesuffix('.json')}__EASY_READ.txt"

      file_ptr = open(os.path.join(TO_SAVE_Dirpath, EasyRead_txt_filename), "w")

      for index in OVERLAP_ANALYSIS_AND_HISTORY__SPLIT__dict:

         file_ptr.write(f"="*25)
         file_ptr.write(f"\n") 
         file_ptr.write(f"{index}\n\n")


         for shared_entity in OVERLAP_ANALYSIS_AND_HISTORY__SPLIT__dict[index]:

            file_ptr.write(f"   {shared_entity}    <-- in '{index}'\n")

            pid_tids = OVERLAP_ANALYSIS_AND_HISTORY__SPLIT__dict[index][shared_entity]['pid_tids']
            overlap_exists = OVERLAP_ANALYSIS_AND_HISTORY__SPLIT__dict[index][shared_entity]['overlap_exists']
            overlap_exists_between = OVERLAP_ANALYSIS_AND_HISTORY__SPLIT__dict[index][shared_entity]['overlap_exists_between']
            events_history = OVERLAP_ANALYSIS_AND_HISTORY__SPLIT__dict[index][shared_entity]['events_history']


            file_ptr.write(f"       pid_tids: {pid_tids}\n")
            file_ptr.write(f"       overlap_exists: {overlap_exists}\n")
            file_ptr.write(f"       overlap_exists_between: {overlap_exists_between}\n")
            file_ptr.write(f"       events_history:\n")
            

            key_thresholds_character = {
               'EventName': 15,
               'TimeStamp': 25,
               'pid_tid': 18,
               'PROVIDER_SPECIFIC_ENTITY': None  # None means print the entire value
            }            

            for event in events_history:
               formatted_line = ""
               for key, value in event.items():
                  threshold = key_thresholds_character.get(key, None)
                  if threshold is not None:
                        formatted_value = f"{value[:threshold]}".rjust(threshold, ' ')
                  else:
                        formatted_value = f"{value}"

                  if key == 'pid_tid':
                      key = 'Thread'
                  elif key == 'PROVIDER_SPECIFIC_ENTITY':
                      key = 'Entity'

                  formatted_line += f"'{key}': {formatted_value} | "

               # file_ptr.write(f"         {event}\n")
               formatted_line.removesuffix(' | ')

               file_ptr.write(f"         {formatted_line}\n")
            file_ptr.write("\n")