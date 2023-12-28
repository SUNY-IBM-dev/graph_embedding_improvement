''' 
******************************************************************************************************************
JY @ 2023-12-26 : Analyze the threads (e.g. if there are artifactual threads that we can be certain of)
                  so that we can perform weighted-sum aggregation for graph-embedding 
******************************************************************************************************************
'''
import sys
sys.path.append("/data/d1/jgwak1/tabby/graph_embedding_improvement_JY_git/graph_embedding_improvement_efforts/Trial_2__Weighted_Thread_Sum_Aggr")

from helper_funcs import *
                        

#from status_0__97_techniques import status_0__97_techniques__list


from elasticsearch import Elasticsearch, helpers

import os
import json
from datetime import datetime

if __name__ == "__main__":


   ''' (1) Don't need to identify the ProcessID of "splunkd.exe" using Elasticsearch (PID may not be needed during execution)'''

   index_list = [ # JY @ 2023-12-26: These are good indices to analyze threads, since we know key-entities
       "joonyoung_single_technique_profile_for_file_event_invoking_custom_technique",
      #  "joonyoung_single_technique_profile_for_registry_event_invoking_custom_technique",
   ]


   ''' (2) Get the 'index' to 'all corresponding log-entries' mapping '''

   indices = index_list
   skipped_indices = []

   indices_to_all_log_entries_dict = dict()
   indices__to__splunkd_and_descendent_pids__dict = dict()


   for index in indices:
      try:
         # Read in all log entries of current es-index.
         es = Elasticsearch(['http://ocelot.cs.binghamton.edu:9200'],timeout = 300)   
         es.indices.put_settings(index = index, body={'index':{'max_result_window':99999999}})
         result = es.search(index = index, size = 99999999)
         index__all_log_entries = result['hits']['hits']    # list of dicts
         indices_to_all_log_entries_dict[index] = index__all_log_entries

         # Also get and store index's splunkd and descentdent pids for 
         # "(3) Identify the ProcessID of the 'caldera-technique process (splunkd's last child process)"
         index__splunkd_and_descendent_pids = get_splunkd_and_descendent_pids( index__all_log_entries )
         indices__to__splunkd_and_descendent_pids__dict[index] = index__splunkd_and_descendent_pids

      except:
         skipped_indices.append(index)
         print(f"\n{len(skipped_indices)}:  {index}  is skipped as Elasticsearch doesn't contain it\n", flush = True)


   ''' (3) Identify the ProcessID of the 'caldera-technique process (splunkd's last child process)' '''

   indices__to__caldera_technique_process__dict = dict()
   for index in indices__to__splunkd_and_descendent_pids__dict:
         splunkd_pid = [ k for k,v in indices__to__splunkd_and_descendent_pids__dict[index].items() \
                        if v['ProcessName'] == 'splunkd' ][0]
         splunkd_children_pids__dict = { k:v for k,v in indices__to__splunkd_and_descendent_pids__dict[index].items() \
                                                                              if v['ParentProcessID'] == splunkd_pid }
         splunkd_children__timestamp_to_pid__dict = { nested_dict['Timestamp'] : pid \
                                                      for pid, nested_dict in splunkd_children_pids__dict.items() }
         last_timestamp_key = max(splunkd_children__timestamp_to_pid__dict)
         caldera_technique_process_pid = splunkd_children__timestamp_to_pid__dict[last_timestamp_key]
         indices__to__caldera_technique_process__dict[index] = caldera_technique_process_pid


   ''' (4) Identify the ProcessIDs of the DESCENDENTS (if any) of the 'caldera-technique process (splunkd's last child process)' '''

   indices__to__caldera_technique_process_and_its_descendents__dict = dict()

   for index in indices__to__splunkd_and_descendent_pids__dict:

         splunkd_and_descendent_pids__dict = indices__to__splunkd_and_descendent_pids__dict[index]
         caldera_technique_process_pid = indices__to__caldera_technique_process__dict[index]

         # get all the n-hop descendents of caldera-techniuqe-process
         caldera_technique_process_and_descendents_pids = [ caldera_technique_process_pid ] 

         for pid in splunkd_and_descendent_pids__dict:
            if splunkd_and_descendent_pids__dict[pid]['ParentProcessID'] in caldera_technique_process_and_descendents_pids:
               caldera_technique_process_and_descendents_pids.append( pid )

         indices__to__caldera_technique_process_and_its_descendents__dict[index] = caldera_technique_process_and_descendents_pids

   ''' (5) Get the log-entries of the caldera-technique-process and its descendents (if any) '''

   indices__to__log_entries_of_caldera_technique_process_and_its_descendents__dict = dict()


   for index, all_log_entries in indices_to_all_log_entries_dict.items():


         caldera_technique_process_and_descendents_pids = indices__to__caldera_technique_process_and_its_descendents__dict[index]

         caldera_technique_process_and_descendents_log_entries = \
            get_log_entries_of_process_of_interest_and_descendents( all_log_entries, 
                                                                    caldera_technique_process_and_descendents_pids )

         # Log-entries are NOT sorted by timestamps by default. So could just explicitly sort it here
         caldera_technique_process_and_descendents_log_entries = sorted( caldera_technique_process_and_descendents_log_entries, 
                                                                         key= lambda item: item['_source']['TimeStamp'] )


         # JY @ 2023-12-26 -- NOW convert @timestamp and TimeStamp from 'datetime' to 'str'
         for log_entry in caldera_technique_process_and_descendents_log_entries:
             log_entry['_source']['TimeStamp'] = str(log_entry['_source']['TimeStamp'])
             log_entry['_source']['@timestamp'] = str(log_entry['_source']['@timestamp'])


         indices__to__log_entries_of_caldera_technique_process_and_its_descendents__dict[index] = caldera_technique_process_and_descendents_log_entries



   ''' (6) Finally, group the caldera-technique-process and its descendent's (if any) log-entries by processThreads '''




   indices__to__processThread_to_logentries__dict = dict()

   for index in indices__to__log_entries_of_caldera_technique_process_and_its_descendents__dict:
         
         log_entries_of_caldera_technique_process_and_its_descendents = indices__to__log_entries_of_caldera_technique_process_and_its_descendents__dict[index]

         log_entries_of_caldera_technique_process_and_its_descendents__with_EntityInfo = \
                              get_log_entries_with_entity_info( log_entries_of_caldera_technique_process_and_its_descendents )

         processThread_to_logentries_dict = group_log_entries_by_processThreads( log_entries_of_caldera_technique_process_and_its_descendents__with_EntityInfo )

         indices__to__processThread_to_logentries__dict[index] = processThread_to_logentries_dict


   # ''' (7) Save the results '''

   results_dirpath = "/data/d1/jgwak1/tabby/graph_embedding_improvement_JY_git/graph_embedding_improvement_efforts/Trial_2__Weighted_Thread_Sum_Aggr/GROUPING_RESULTS"
   if not os.path.exists(results_dirpath):
      raise RuntimeError(f"{results_dirpath} does not exist!")


   for index in indices__to__processThread_to_logentries__dict:

      results_fpath = os.path.join(results_dirpath, f"processThread_to_logentries__of__caldera_technique_and_descendent_procs__{index}__{datetime.now().strftime('%Y-%m-%d_%H_%M_%S')}.json")
      
      # json dumps indent related:
      # https://stackoverflow.com/questions/13249415/how-to-implement-custom-indentation-when-pretty-printing-with-the-json-module

      # JY @ 2023-12-26: Got Object of type datetime is not json serializable b/c of @timestamp attribute -- fix it.
      
      json_string = json.dumps( indices__to__processThread_to_logentries__dict[index] , indent= 2, cls = NoIndentEncoder )
      with open(results_fpath, "w") as json_fp:
            json_fp.write(json_string)