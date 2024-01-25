import os
import json



            # maybe to this for the specifically suspciious ones?
            # or do it wholistically?
            # complexity will be 

            #  for thread in all indices:
            # 
            #     for all threads in all other indices:
            #            

            #   N * M * N-1 * M-1

if __name__ == "__main__":

   processThread_grouped_events_jsons_dirpath = \
      "/home/jgwak1/tabby/graph_embedding_improvement_JY_git/analyze_at_data_level/Event_and_Entity_Level_Artifactual_Thread_Check/RESULTS/Event_and_Entity_Level_Artifactual_Thread_Check__Dataset-Case-1__2024-01-24_134317/processThread_grouped_events"
   
   index_json_files = os.listdir(processThread_grouped_events_jsons_dirpath)

   cnt = 0


   all__thread_event_entity_combination_sequences__with_id__across_indices = dict() # id(int) -- # {'sequence': sequence,
                                                                                    #                 'length' (int),
                                                                                    #                'index' }

   for index_json_file in index_json_files:
      
      cnt+=1
      index = index_json_file.removesuffix('.json')
      print(f"{cnt} / {len(index_json_files)} | {index}", flush= True)

      index__processThread_grouped_events_json_fpath = os.path.join(processThread_grouped_events_jsons_dirpath, index_json_file)

      with open( index__processThread_grouped_events_json_fpath, 'r') as file:
         index__processThread_grouped_events = json.load(file)


      # Prepare the 'all__thread_event_entity_combination_sequences__with_id__across_indices'
      cnt = 1
      for pid in index__processThread_grouped_events:

         for tid in index__processThread_grouped_events[pid]:

            thread_sorted_log_entries =  index__processThread_grouped_events[pid][tid]
            thread_sorted_EventName_Entity_pair_seuqence = [ f"{x['EventName']}__{x['PROVIDER_SPECIFIC_ENTITY']}" for x in thread_sorted_log_entries ]
            
            all__thread_event_entity_combination_sequences__with_id__across_indices[str(cnt)] = {
               "length" : len(thread_sorted_EventName_Entity_pair_seuqence),
               "sorted_sequence" : thread_sorted_EventName_Entity_pair_seuqence,
               "index": index  
            }
            cnt += 1

   # Now work on, based on the collected 'all__thread_event_entity_combination_sequences__with_id__across_indices', :
   #  Goal is to get a mapping between each thread event-entity combination sequence in all indices and all indices where the sequence appears 
   # 
   thread_event_entity_combination_sequence_TO__all_occurering_indices = dict()



