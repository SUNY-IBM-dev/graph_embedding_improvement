import os
import json
import ast
import base64

            # maybe to this for the specifically suspciious ones?
            # or do it wholistically?
            # complexity will be 

            #  for thread in all indices:
            # 
            #     for all threads in all other indices:
            #            

            #   N * M * N-1 * M-1


'''
JY @ 2024-1-28

This check should be done based on 'use_node_indentifier == False' instead of True,
since this is inter-sample check, not an intra-sample check.

'''


if __name__ == "__main__":

   running_from_machine = "ocelot"

   if running_from_machine == "ocelot":
      abs_path_to_tabby = "/data/d1/jgwak1/tabby"
   elif running_from_machine == "panther":
      abs_path_to_tabby = "/home/jgwak1/tabby"
   else:
      ValueError()


   processThread_grouped_events_jsons_dirpath = \
      f"{abs_path_to_tabby}/graph_embedding_improvement_JY_git/analyze_at_data_level/Event_and_Entity_Level_Artifactual_Thread_Check/RESULTS/Event_and_Entity_Level_Artifactual_Thread_Check__Dataset-Case-1__2024-01-24_134317/processThread_grouped_events"
   
   index_json_files = os.listdir(processThread_grouped_events_jsons_dirpath)

   TO_SAVE_DIRPATH = os.path.split(processThread_grouped_events_jsons_dirpath)[0]


   all__thread_event_entity_combination_sequences__with_id__across_indices = dict() # id(int) -- # {'sequence': sequence,
                                                                                    #                 'length' (int),
                                                                                    #                'index' }
   thread_cnt = 1
   index_cnt = 0
   for index_json_file in index_json_files:
      index_cnt += 1
      index = index_json_file.removesuffix('.json')
      print(f"{index_cnt} / {len(index_json_files)} | {index}", flush= True)

      index__processThread_grouped_events_json_fpath = os.path.join(processThread_grouped_events_jsons_dirpath, index_json_file)

      with open( index__processThread_grouped_events_json_fpath, 'r') as file:
         index__processThread_grouped_events = json.load(file)


      # Prepare the 'all__thread_event_entity_combination_sequences__with_id__across_indices'
      for pid in index__processThread_grouped_events:

         for tid in index__processThread_grouped_events[pid]:

            thread_sorted_log_entries =  index__processThread_grouped_events[pid][tid]

            # ----------------------------------------------------------------------------------------------------------------
            # JY @ 2024-1-28
            #        This check should be done based on 'use_node_indentifier == False' instead of True,
            #        since this is inter-sample check, not an intra-sample check.
            #        
            #        Therefore, drop the object-identifer (meory address) 
            #
            # following referred to : /data/d1/jgwak1/tabby/graph_embedding_improvement_JY_git/analyze_at_data_level/Event_and_Entity_Level_Artifactual_Thread_Check/analyze__processThread_grouped_events.py

            for log_entry in thread_sorted_log_entries:
               if log_entry['PROVIDER_SPECIFIC_ENTITY'] == None:
                  continue
               if "__" in log_entry['PROVIDER_SPECIFIC_ENTITY']:
                  log_entry['PROVIDER_SPECIFIC_ENTITY'] = log_entry['PROVIDER_SPECIFIC_ENTITY'].split("__",1)[1]
               
               # if log_entry['PROVIDER_SPECIFIC_ENTITY'] == 'None':
               #    continue

               # if log_entry['PROVIDER_SPECIFIC_ENTITY'] == '':
               #    continue
            # ----------------------------------------------------------------------------------------------------------------


            thread_sorted_EventName_Entity_pair_seuqence = [ f"{x['EventName']}__{x['PROVIDER_SPECIFIC_ENTITY']}" for x in thread_sorted_log_entries ]
            
            all__thread_event_entity_combination_sequences__with_id__across_indices[str(thread_cnt)] = {
               "length" : len(thread_sorted_EventName_Entity_pair_seuqence),
               "sorted_event_entity_str_sequence_list" : thread_sorted_EventName_Entity_pair_seuqence,
               "index": index  
            }
            thread_cnt += 1

   # Now work on, based on the collected 'all__thread_event_entity_combination_sequences__with_id__across_indices', :
   #  Goal is to get a mapping between each thread event-entity combination sequence in all indices and all indices where the sequence appears 
   # 
   from collections import defaultdict
   thread_event_entity_combination_sequence_TO__all_occurering_indices = defaultdict(set) # <-- this can lead to memory issue



   # only consider threads more than N events
   N = 2
   all__thread_event_entity_combination_sequences__with_id__across_indices__MORE_THAN_N_events =\
      {k:v for k,v in all__thread_event_entity_combination_sequences__with_id__across_indices.items() if v['length'] > N}

   



   keys = list(all__thread_event_entity_combination_sequences__with_id__across_indices__MORE_THAN_N_events.keys())

   id_sets__with_same_thread_event_entity_combination_sequences = list()



   from math import factorial

   # def calculate_combinations_count(n, k):
   #    return factorial(n) // (factorial(k) * factorial(n - k))

   # total_combinations = 0

   # for r in range(1, len(keys) + 1):
   #    total_combinations += calculate_combinations_count(len(keys), r)
   outercnt = 1
   for i in range(len(keys)):
      for j in range(i + 1, len(keys)):


         key1 = keys[i]
         key2 = keys[j]

         key1__index = all__thread_event_entity_combination_sequences__with_id__across_indices__MORE_THAN_N_events[key1]['index']
         key2__index = all__thread_event_entity_combination_sequences__with_id__across_indices__MORE_THAN_N_events[key2]['index']


         if key1__index == key2__index :
             # continue since we want to identify artifactual-threads across indices
             continue


         key1__length = all__thread_event_entity_combination_sequences__with_id__across_indices__MORE_THAN_N_events[key1]['length']
         key2__length = all__thread_event_entity_combination_sequences__with_id__across_indices__MORE_THAN_N_events[key2]['length']


         #print(f"key1_length: {key1__length} | key2_legnth: {key2__length}", flush=True)
         if key1__length == key2__length: 

               if (key1__length == 30 and key2__length == 30):
                  stophere = "yes"



               key1__sorted_event_entity_str_sequence_list = all__thread_event_entity_combination_sequences__with_id__across_indices__MORE_THAN_N_events[key1]['sorted_event_entity_str_sequence_list']
               key2__sorted_event_entity_str_sequence_list = all__thread_event_entity_combination_sequences__with_id__across_indices__MORE_THAN_N_events[key2]['sorted_event_entity_str_sequence_list']

               if key1__sorted_event_entity_str_sequence_list == key2__sorted_event_entity_str_sequence_list:
                  
                  matched_key = key1


                  print(key1__sorted_event_entity_str_sequence_list)


                  found_idset__where_matched_key_belongs = False
                  for id_set in id_sets__with_same_thread_event_entity_combination_sequences:
                     
                     found_so_break_out = False

                     for key__in__id_set in id_set:


                        if all__thread_event_entity_combination_sequences__with_id__across_indices__MORE_THAN_N_events[key__in__id_set]['sorted_event_entity_str_sequence_list'] == \
                           all__thread_event_entity_combination_sequences__with_id__across_indices__MORE_THAN_N_events[matched_key]['sorted_event_entity_str_sequence_list']:                                    
                           found_idset__where_matched_key_belongs = True
                        
                        if found_idset__where_matched_key_belongs:
                           
                           found_so_break_out = True
                           break
                     
                     if found_so_break_out == True:
                        id_set.add(key1)
                        id_set.add(key2)
                        break


                  if found_idset__where_matched_key_belongs == False:
                     id_sets__with_same_thread_event_entity_combination_sequences.append({key1, key2})
                     

                  # folowing -- this can lead to memory issue
                  thread_event_entity_combination_sequence_TO__all_occurering_indices[str(key1__sorted_event_entity_str_sequence_list)].add(key1__index)
                  thread_event_entity_combination_sequence_TO__all_occurering_indices[str(key1__sorted_event_entity_str_sequence_list)].add(key2__index)


                  print()
      print(f"{outercnt} / {len(keys)}", flush =True)
      outercnt +=1               
      print()  
   print() 

   results_save_fpath = os.path.join(TO_SAVE_DIRPATH, "analyze__artifactual_threads_Results.txt")
   # Results based on 'thread_event_entity_combination_sequence_TO__all_occurering_indices'
   results_save_fp = open(results_save_fpath, "w")
   count = 0
   for key,val in thread_event_entity_combination_sequence_TO__all_occurering_indices.items():
      count += 1

      thread__event_entity__sorted_sequence = ast.literal_eval(key)
      appeared_in_indices = list(set(val))
      appeared_in_benign = [x for x in appeared_in_indices if "benign" in x ]
      appeared_in_malware = [x for x in appeared_in_indices if "malware" in x ]

      results_save_fp.write(f"Thread Event-Entity Sorted Sequence Pattern {count} (with length {len(thread__event_entity__sorted_sequence)}):\n\n")
      results_save_fp.write(f"   {thread__event_entity__sorted_sequence}\n\n")
      
      results_save_fp.write(f"Appeared in {len(appeared_in_benign)} benign-indices and {len(appeared_in_malware)} malware-indices:\n\n")
      results_save_fp.write(f"   {appeared_in_benign}\n\n")
      results_save_fp.write(f"   {appeared_in_malware}\n\n")
      results_save_fp.write(f"-"*50)
      results_save_fp.write(f"\n")
   results_save_fp.close()


   # Results based on 'id_sets__with_same_thread_event_entity_combination_sequences'

   # [ all__thread_event_entity_combination_sequences__with_id__across_indices__MORE_THAN_N_events[x] for x in id_sets__with_same_thread_event_entity_combination_sequences[0] ]
   results_save_fpath = os.path.join(TO_SAVE_DIRPATH, "analyze__artifactual_threads_Results__based_on_id_sets.txt")
   results_save_fp = open(results_save_fpath, "w")
   count = 0

   for thread__event_entity__sorted_sequence__id_set in id_sets__with_same_thread_event_entity_combination_sequences:
      count += 1

      corresponding_ids__info = [ all__thread_event_entity_combination_sequences__with_id__across_indices__MORE_THAN_N_events[_id] \
                                 for _id in thread__event_entity__sorted_sequence__id_set ]

      appeared_in_indices = list( set( [ x['index'] for x in corresponding_ids__info ] ) )
      appeared_in_benign = [x for x in appeared_in_indices if "benign" in x ]
      appeared_in_malware = [x for x in appeared_in_indices if "malware" in x ]


      sorted_event_entity_str_sequence_list = corresponding_ids__info[0]['sorted_event_entity_str_sequence_list']

      results_save_fp.write(f"Thread Event-Entity Sorted Sequence Pattern {count} (with length {len(sorted_event_entity_str_sequence_list)}):\n\n")
      results_save_fp.write(f"   {sorted_event_entity_str_sequence_list}\n\n")
      
      results_save_fp.write(f"Appeared in {len(appeared_in_benign)} benign-indices and {len(appeared_in_malware)} malware-indices:\n\n")
      results_save_fp.write(f"   {appeared_in_benign}\n\n")
      results_save_fp.write(f"   {appeared_in_malware}\n\n")
      results_save_fp.write(f"-"*50)
      results_save_fp.write(f"\n")
   results_save_fp.close()
