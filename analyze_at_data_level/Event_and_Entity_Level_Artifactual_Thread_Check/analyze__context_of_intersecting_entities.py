import os
import json 

'''
   Provides information of what kind of events the 'interesecting entity' is associated with, within the relevant threads

'''

if __name__ == "__main__":

   

   indices__intersecting_entities__to__pidtid__json_fpath = \
      "/data/d1/jgwak1/tabby/graph_embedding_improvement_JY_git/analyze_at_data_level/Event_and_Entity_Level_Artifactual_Thread_Check/RESULTS/Event_and_Entity_Level_Artifactual_Thread_Check__Dataset-Case-1__2024-01-24_134317/indices__intersecting_entities__to__pidtid__used_node_identifier_True.json"

   processThread_grouped_events_jsons_dirpath = "/data/d1/jgwak1/tabby/graph_embedding_improvement_JY_git/analyze_at_data_level/Event_and_Entity_Level_Artifactual_Thread_Check/RESULTS/Event_and_Entity_Level_Artifactual_Thread_Check__Dataset-Case-1__2024-01-24_134317/processThread_grouped_events"


   with open(indices__intersecting_entities__to__pidtid__json_fpath, 'r') as file:
      indices__intersecting_entities__to__pidtid = json.load(file)

   # provides information of what kind of events the 'interesecting entity' is associated with, within the relevant threads 


   indices = list( indices__intersecting_entities__to__pidtid.keys() )
   cnt = 0
   for index in indices:
         cnt+=1
         print(f"{cnt} / {len(indices)} | {index}", flush= True)
         index__processThread_grouped_events_json_fpath = os.path.join(processThread_grouped_events_jsons_dirpath, f"{index}.json")

         with open( index__processThread_grouped_events_json_fpath, 'r') as file:
            index__processThread_grouped_events = json.load(file)


         for intersecting_entity in indices__intersecting_entities__to__pidtid[index].keys():
             
            pid_tids__entity_interacted_with = indices__intersecting_entities__to__pidtid[index][intersecting_entity]

            for pid_tid in pid_tids__entity_interacted_with:
               parts = pid_tid.split('_') 
               pid = parts[0] + '_' + parts[1]
               tid = parts[2] + '_' + parts[3]

               events_associaed_with_intersecting_entity = [x for x in index__processThread_grouped_events[pid][tid] \
                                                            if x['PROVIDER_SPECIFIC_ENTITY'] == intersecting_entity]



               pid_tid_with_context = {pid_tid: events_associaed_with_intersecting_entity}

               idx_of_pid_tid_in_list = indices__intersecting_entities__to__pidtid[index][intersecting_entity].index(pid_tid)
               indices__intersecting_entities__to__pidtid[index][intersecting_entity][idx_of_pid_tid_in_list] = pid_tid_with_context


   print()

   TO_SAVE_dirpath = os.path.split(processThread_grouped_events_jsons_dirpath)[0]
   save_fpath = os.path.split(indices__intersecting_entities__to__pidtid__json_fpath)[1].removesuffix(".json") + "___WITH_ASSOCIATED_EVENTS_INFO.json"
   with open(os.path.join( TO_SAVE_dirpath, save_fpath ), 'w') as json_file:
      json.dump(indices__intersecting_entities__to__pidtid, json_file)
