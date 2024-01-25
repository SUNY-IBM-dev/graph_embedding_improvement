import os
import json 
from datetime import datetime
import sys
'''
   Provides information of what kind of events the 'interesecting entity' is associated with, within the relevant threads

'''

if __name__ == "__main__":

   

   indices__intersecting_entities__to__pidtid__json_fpath = \
      "/home/jgwak1/tabby/graph_embedding_improvement_JY_git/analyze_at_data_level/Event_and_Entity_Level_Artifactual_Thread_Check/RESULTS/Event_and_Entity_Level_Artifactual_Thread_Check__Dataset-Case-1__2024-01-24_134317/indices__intersecting_entities__to__pidtid__used_node_identifier_True.json"

   processThread_grouped_events_jsons_dirpath = "/home/jgwak1/tabby/graph_embedding_improvement_JY_git/analyze_at_data_level/Event_and_Entity_Level_Artifactual_Thread_Check/RESULTS/Event_and_Entity_Level_Artifactual_Thread_Check__Dataset-Case-1__2024-01-24_134317/processThread_grouped_events"


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

   # TO_SAVE_dirpath = os.path.split(processThread_grouped_events_jsons_dirpath)[0]
   # save_fpath = os.path.split(indices__intersecting_entities__to__pidtid__json_fpath)[1].removesuffix(".json") + "___WITH_ASSOCIATED_EVENTS_INFO.json"
   # with open(os.path.join( TO_SAVE_dirpath, save_fpath ), 'w') as json_file:
   #    json.dump(indices__intersecting_entities__to__pidtid, json_file)

   # ------------------------------------------------------------------------------------------------------------------------------
   split_by_how_many_by_class__N = 5
   SAVE_indices__intersecting_entities__to__pidtid__json_fpath_____WITH_ASSOCIATED_EVENTS_INFO___SPLIT = False
   SAVE_indices__intersecting_entities__to__pidtid__json_fpath_____WITH_OVERLAP_ANALYSIS_AND_HISTORY___SPLIT = True
   def split_dict_into_n(dictionary, n):
         # Calculate the number of elements in each part
         num_elements = len(dictionary)
         elements_per_part = num_elements // n
         split_dicts = [dict(list(dictionary.items())[i * elements_per_part: (i + 1) * elements_per_part]) for i in range(n)]
         return split_dicts

   benign_indices__intersecting_entities__to__pidtid = {key: val for key,val in indices__intersecting_entities__to__pidtid.items() \
                                                         if "benign" in key}
   malware_indices__intersecting_entities__to__pidtid = {key: val for key,val in indices__intersecting_entities__to__pidtid.items() \
                                                         if "malware" in key}

   N_splitted__benign_indices__intersecting_entities__to__pidtid = split_dict_into_n(benign_indices__intersecting_entities__to__pidtid, 
                                                                                       split_by_how_many_by_class__N)
   
   N_splitted__malware_indices__intersecting_entities__to__pidtid = split_dict_into_n(malware_indices__intersecting_entities__to__pidtid, 
                                                                                       split_by_how_many_by_class__N)      

   
   if SAVE_indices__intersecting_entities__to__pidtid__json_fpath_____WITH_ASSOCIATED_EVENTS_INFO___SPLIT:
         TO_SAVE_dirpath = os.path.split(processThread_grouped_events_jsons_dirpath)[0]

         for i in range(split_by_how_many_by_class__N):
            benign_split_save_fpath = os.path.split(indices__intersecting_entities__to__pidtid__json_fpath)[1].removesuffix(".json") + f"___WITH_ASSOCIATED_EVENTS_INFO___BENIGN__SPLIT_{i+1}.json"
            malware_split_save_fpath = os.path.split(indices__intersecting_entities__to__pidtid__json_fpath)[1].removesuffix(".json") + f"___WITH_ASSOCIATED_EVENTS_INFO___MALWARE__SPLIT_{i+1}.json"

            with open(os.path.join( TO_SAVE_dirpath, benign_split_save_fpath ), 'w') as json_file:
               json.dump ( N_splitted__benign_indices__intersecting_entities__to__pidtid[i], json_file)

            with open(os.path.join( TO_SAVE_dirpath, malware_split_save_fpath ), 'w') as json_file:
               json.dump ( N_splitted__malware_indices__intersecting_entities__to__pidtid[i], json_file)            

   # ------------------------------------------------------------------------------------------------------------------------------
   # Shared-event history tracker 
   # 
   # shared-entity : {
   #     "shared_across" : [ pid-tid-1, pid-tid-2, ..] ,      # Add a check if there is overlap in timestamp between events
   #                                                          # i.e., interleaving happens       
   #     "history": [                                         # should be sorted by timestamp in ascending order
   #          { "EventName": EventName, "TimeStamp": Timestamp, "Pid-Tid": pid_tid },
   #          { "EventName": EventName, "TimeStamp": Timestamp, "Pid-Tid": pid_tid },  
   #          { "EventName": EventName, "TimeStamp": Timestamp, "Pid-Tid": pid_tid },
   #          { "EventName": EventName, "TimeStamp": Timestamp, "Pid-Tid": pid_tid },  
   #           
   #                          ....
   # ]
   # }



   def check_overlap(list_of_lists):
         overlap_between = []
         for i in range(len(list_of_lists) - 1):
            for j in range(i + 1, len(list_of_lists)):
                  list1 = list_of_lists[i]
                  list2 = list_of_lists[j]

                  list1_pidtid = list1[0]['pid_tid']
                  list2_pidtid = list2[0]['pid_tid']

                  # Extract timestamps from dictionaries
                  timestamps1 = []
                  for d in list1:
                     try:
                        d_TimeStamp = datetime.strptime( d['TimeStamp'], '%Y-%m-%d %H:%M:%S.%f' )                          
                     except:
                        try:
                           d_TimeStamp = datetime.strptime( d['TimeStamp'], '%Y-%m-%d %H:%M:%S' )  # sometimes there's no floating   
                     
                        except Exception as e:
                              print(e)
                              sys.exit()
                  
                     timestamps1.append(d_TimeStamp)
                  
                  # Extract timestamps from dictionaries
                  timestamps2 = []
                  for d in list2:
                     try:
                        d_TimeStamp = datetime.strptime( d['TimeStamp'], '%Y-%m-%d %H:%M:%S.%f' )                          
                     except:
                        try:
                           d_TimeStamp = datetime.strptime( d['TimeStamp'], '%Y-%m-%d %H:%M:%S' )  # sometimes there's no floating   
                     
                        except Exception as e:
                              print(e)
                              sys.exit()
                  
                     timestamps2.append(d_TimeStamp)

                  

                  # Check for overlap

                  # cover all possible cases, including scenarios where the sizes of the lists vary.
                  if max(timestamps1) >= min(timestamps2) and min(timestamps1) <= max(timestamps2):
                     overlap_between.append( (list1_pidtid, list2_pidtid))

         if len(overlap_between) == 0:
            return False, overlap_between  # No overlap found
         else:
                                             
            return True, overlap_between  # Overlap found
   


   for i in range(split_by_how_many_by_class__N):

      # ------Benign---------------------------------------
      benign_cnt = 1
      for benign_index in N_splitted__benign_indices__intersecting_entities__to__pidtid[i]:
         print(f"split_by_how_many_by_class__N == {i+1} / {split_by_how_many_by_class__N} |  {benign_cnt} / {len(N_splitted__benign_indices__intersecting_entities__to__pidtid[i])}  |  {benign_index}", flush=True)
         for shared_entity in N_splitted__benign_indices__intersecting_entities__to__pidtid[i][benign_index]:
            
            all_log_entries_of__ALL_pid_tid__interact_with_shared_entity = []
            all_pid_tids = []

            for pid_tid in [ list(x.keys())[0] for x in N_splitted__benign_indices__intersecting_entities__to__pidtid[i][benign_index][shared_entity]]:

                  all_log_entries_of__THIS_pid_tid__interact_with_shared_entity = \
                     [ x for x in N_splitted__benign_indices__intersecting_entities__to__pidtid[i][benign_index][shared_entity] if list(x.keys())[0] == pid_tid][0][pid_tid]

                  for log_entry_dict in all_log_entries_of__THIS_pid_tid__interact_with_shared_entity:
                      log_entry_dict['pid_tid'] = pid_tid
                      Provider_specific_entity = log_entry_dict.pop("PROVIDER_SPECIFIC_ENTITY")
                      log_entry_dict['PROVIDER_SPECIFIC_ENTITY'] = Provider_specific_entity   # for correcting the order
                  all_log_entries_of__THIS_pid_tid__interact_with_shared_entity__SORTED = \
                                                                  sorted(all_log_entries_of__THIS_pid_tid__interact_with_shared_entity,
                                                                                         key = lambda log_entry: log_entry['TimeStamp'])


                  all_log_entries_of__ALL_pid_tid__interact_with_shared_entity.append\
                                                   (all_log_entries_of__THIS_pid_tid__interact_with_shared_entity__SORTED)

                  all_pid_tids.append(pid_tid)




            # Add a check if there is overlap in timestamp between events
            # i.e., interleaving happens  
            # CHECK-- using all_'log_entries_of__ALL_pid_tid__interact_with_shared_entity'
            

            overlap_exists, overlap_between = check_overlap( all_log_entries_of__ALL_pid_tid__interact_with_shared_entity )

            if overlap_exists == True:
                abv = 1


            # Now sort the entire one
            def flatten(list_of_list):
                return [item for sublist in list_of_list for item in sublist]

            all_log_entries_of__ALL_pid_tid__interact_with_shared_entity__SORTED = \
               sorted( flatten(all_log_entries_of__ALL_pid_tid__interact_with_shared_entity), 
                        key = lambda log_entry: log_entry['TimeStamp'])
            


            
            N_splitted__benign_indices__intersecting_entities__to__pidtid[i][benign_index][shared_entity] =\
               {  "pid_tids": all_pid_tids,
                  "overlap_exists": overlap_exists,
                  "overlap_exists_between": overlap_between,
                  "events_history": all_log_entries_of__ALL_pid_tid__interact_with_shared_entity__SORTED}
         benign_cnt += 1   

      # Now save the updated 'N_splitted__benign_indices__intersecting_entities__to__pidtid[i]'

      if SAVE_indices__intersecting_entities__to__pidtid__json_fpath_____WITH_OVERLAP_ANALYSIS_AND_HISTORY___SPLIT:
            TO_SAVE_dirpath = os.path.split(processThread_grouped_events_jsons_dirpath)[0]
            benign_split_save_fpath = os.path.split(indices__intersecting_entities__to__pidtid__json_fpath)[1].removesuffix(".json") + f"___WITH_OVERLAP_ANALYSIS_AND_HISTORY___BENIGN__SPLIT_{i}.json"

            with open(os.path.join( TO_SAVE_dirpath, benign_split_save_fpath ), 'w') as json_file:
                  json.dump ( N_splitted__benign_indices__intersecting_entities__to__pidtid[i], json_file)

      # ------Malware---------------------------------------
      malware_cnt = 1
      for malware_index in N_splitted__malware_indices__intersecting_entities__to__pidtid[i]:
         print(f"split_by_how_many_by_class__N == {i+1} / {split_by_how_many_by_class__N}  |  {malware_cnt} / {len(N_splitted__malware_indices__intersecting_entities__to__pidtid[i])}  |  {malware_index}", flush=True)
         for shared_entity in N_splitted__malware_indices__intersecting_entities__to__pidtid[i][malware_index]:
            
            all_log_entries_of__ALL_pid_tid__interact_with_shared_entity = []
            all_pid_tids = []

            for pid_tid in [ list(x.keys())[0] for x in N_splitted__malware_indices__intersecting_entities__to__pidtid[i][malware_index][shared_entity]]:

                  all_log_entries_of__THIS_pid_tid__interact_with_shared_entity = \
                     [ x for x in N_splitted__malware_indices__intersecting_entities__to__pidtid[i][malware_index][shared_entity] if list(x.keys())[0] == pid_tid][0][pid_tid]

                  for log_entry_dict in all_log_entries_of__THIS_pid_tid__interact_with_shared_entity:
                      log_entry_dict['pid_tid'] = pid_tid
                      Provider_specific_entity = log_entry_dict.pop("PROVIDER_SPECIFIC_ENTITY")
                      log_entry_dict['PROVIDER_SPECIFIC_ENTITY'] = Provider_specific_entity   # for correcting the order
                  all_log_entries_of__THIS_pid_tid__interact_with_shared_entity__SORTED = \
                                                                  sorted(all_log_entries_of__THIS_pid_tid__interact_with_shared_entity,
                                                                                         key = lambda log_entry: log_entry['TimeStamp'])


                  all_log_entries_of__ALL_pid_tid__interact_with_shared_entity.append\
                                                   (all_log_entries_of__THIS_pid_tid__interact_with_shared_entity__SORTED)

                  all_pid_tids.append(pid_tid)




            # Add a check if there is overlap in timestamp between events
            # i.e., interleaving happens  
            # CHECK-- using all_'log_entries_of__ALL_pid_tid__interact_with_shared_entity'
            

            overlap_exists, overlap_between = check_overlap( all_log_entries_of__ALL_pid_tid__interact_with_shared_entity )

            if overlap_exists == True:
                abv = 1


            # Now sort the entire one
            def flatten(list_of_list):
                return [item for sublist in list_of_list for item in sublist]

            all_log_entries_of__ALL_pid_tid__interact_with_shared_entity__SORTED = \
               sorted( flatten(all_log_entries_of__ALL_pid_tid__interact_with_shared_entity), 
                        key = lambda log_entry: log_entry['TimeStamp'])
            


            
            N_splitted__malware_indices__intersecting_entities__to__pidtid[i][malware_index][shared_entity] =\
               {  "pid_tids": all_pid_tids,
                  "overlap_exists": overlap_exists,
                  "overlap_exists_between": overlap_between,
                  "events_history": all_log_entries_of__ALL_pid_tid__interact_with_shared_entity__SORTED}
         malware_cnt += 1   

      # Now save the updated 'N_splitted__malware_indices__intersecting_entities__to__pidtid[i]'

      if SAVE_indices__intersecting_entities__to__pidtid__json_fpath_____WITH_OVERLAP_ANALYSIS_AND_HISTORY___SPLIT:
            TO_SAVE_dirpath = os.path.split(processThread_grouped_events_jsons_dirpath)[0]

            malware_split_save_fpath = os.path.split(indices__intersecting_entities__to__pidtid__json_fpath)[1].removesuffix(".json") + f"___WITH_OVERLAP_ANALYSIS_AND_HISTORY___MALWARE__SPLIT_{i}.json"

            with open(os.path.join( TO_SAVE_dirpath, malware_split_save_fpath ), 'w') as json_file:
                  json.dump ( N_splitted__malware_indices__intersecting_entities__to__pidtid[i], json_file)    



