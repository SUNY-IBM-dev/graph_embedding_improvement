# Refered to : 'identify_artifactual_events_by_trivial_technique_profile.py' and 'identify_behavioral_events.py' 

import sys
sys.path.append("/data/d1/jgwak1/tabby/SUNYIBM_ExplainableAI_2nd_Year_JY/Task_1__Behavior_identification_and_intention_learning/1_0__Identify_Behavioral_Events")

from helper_funcs import get_log_entries_with_entity_info,\
                         summarize_log_entires_by_entity_and_key_info,\
                         group_log_entries_by_entities,\
                         get_splunkd_and_descendent_pids,\
                         get_log_entries_of_process_of_interest_and_descendents,\
                         find_unsorted_elements_and_indices,\
                         check_whether_log_entries_sorted_within_same_ProcessThread,\
                         group_log_entries_by_processThreads

from status_0__97_techniques import status_0__97_techniques__list


from elasticsearch import Elasticsearch, helpers

import os
import json
from datetime import datetime

if __name__ == "__main__":


   ''' (1) Don't need to identify the ProcessID of "splunkd.exe" using Elasticsearch (PID may not be needed during execution)'''

   index_list = [
      "atomic__t1055_003__multiple__thread_execution_hijacking__6a64ea6e29cdb83d468a27d6f69960cb__trial_1", # : ["notepad.exe", "T1055.003\\bin\\InjectContext.exe", "InjectContext.exe"]}, // Found "notepad.exe" in log-entries

      "atomic__t1219__command-and-control__remote_access_software__aada5380e7d0a4c7b71f2a324d9d5327__trial_1", # :["AnyDesk.exe"]},  // Found in log-entries

      "atomic__t1556_002__multiple__modify_authentication_process-_password_filter_dll__cc7f0eb8b9115b271eaaa42c9b6f3dca__trial_1", # : ["reg.exe", "HKLM\\SYSTEM\\CurrentControlSet\\Control\\Lsa\\", "AtomicRedTeamPWFilter.dll", "C:\\Windows\\System32"]}, // Found "reg.exe" and "Lsa" in log-entries

      "atomic__t1547_014__multiple__active_setup__7ad5840a79f3259965fa28835dda93c4__trial_1", # : ["HKCU:\SOFTWARE\Microsoft\Active Setup\Installed Components\{C9E9A340-D1F1-11D0-821E-444553540600}", "{C9E9A340-D1F1-11D0-821E-444553540600}", "C9E9A340-D1F1-11D0-821E-444553540600", "system32\\runonce.exe"]}, // Found "runonce.exe" and "C9E9A340-D1F1-11D0-821E-444553540600" in log-entries

      "atomic__t1055_012__multiple__process_injection-_process_hollowing__557321faaf98c77b2b452cecd7b1de37__trial_1", # :["WINWORD.EXE", "winword.exe", "notepad.exe"]}, # only by description // Found "notepad.exe" in log-entries.

      "atomic__t1055_003__multiple__thread_execution_hijacking__6a64ea6e29cdb83d468a27d6f69960cb__trial_1", # ["notepad.exe", "InjectContext.exe"]}, # not sure how "InjectContext.exe" has run?  It seems $PathToAtomicsFolder is a network path? Need to check on VM but when caldera C2 control is established  // Found "notepad.exe" in log-entries.

      "atomic__t1548_002__multiple__abuse_elevation_control_mechanism-_bypass_user_account_control__64430e7597668877a832b9d1e379c9f2__trial_1", # :["HKCU:\\Software\\Classes\\AppX82a6gwre4fdg3bt635tn5ctqjf8msdd2\\Shell\\open\\command", "DelegateExecute", "WSReset.exe", "wsreset.exe"]}, // Found ALL in log-entries.

      "atomic__t1564_003__defense-evasion__hide_artifacts-_hidden_window__f1222384fe40cc71e7dea9d182014eaf__trial_1", # :["powershell.exe", "calc.exe"]}, // Found "calc.exe" in log-entries.

      "atomic__t1548_002__multiple__abuse_elevation_control_mechanism-_bypass_user_account_control__e7d20e7f0087f8a4234c1d1b7a228bb0__trial_1", # :["HKLM:\\SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\Policies\\System","ConsentPromptBehaviorAdmin"]}, // Found "ConsentPromptBehaviorAdmin" in log-entries. 

      "atomic__t1219__command-and-control__remote_access_software__f1b3fca18d7465cd10e5a7477a3bf97d__trial_1", # ["LogMeInIgnition.msi","LMIIgnition.exe"]}, // Found "LMIIgnition.exe" in log-entries.

      "atomic__t1070_004__defense-evasion__indicator_removal_on_host-_file_deletion__2413b013bc82d152765e2ac34601a327__trial_1", # ["\prefetch\", ".pf"]},  # Good example for file-extension matters   // Found both  "prefetch" and ".pf" in log-entries.

      "atomic__t1055_002__multiple__process_injection-_portable_executable_injection__ca6a3f579181ea47b7d95779e8d8a79b__trial_1", # :["971b85_RedInjection.exe", "notepad.exe"]}, # Note sure if "971b85_RedInjection.exe" was actually on the VM -- why got status 0 and success on caldera-gui?  // Found both "971b85_RedInjection.exe" and "notepad.exe" in log-entries
   
      "atomic__t1059_005__execution__command_and_scripting_interpreter-_visual_basic__f2131e45dbd95e3057bd3494b5aeed41__trial_1", # ["cscript.exe", "a771e6_sys_info.vbs", "T1059.005.out.txt"]}, // Found only "cscript.exe" in log entries 

      "atomic__t1070_003__defense-evasion__indicator_removal_on_host-_clear_command_history__adce11c81bb77ae74660c6c743a0442d__trial_1", # :["ConsoleHost_history.txt"]}, # good example  // Found in log-entries

      "atomic__t1219__command-and-control__remote_access_software__b6ebae300e5ff115e965cc9179d4f831__trial_1", # ["GoToAssist.exe"]}, # may not be able to get full activity of Invoke-WebRequest // Found in log-entries

      "atomic__t1036__defense-evasion__masquerading__7575d3d5ae97ee568d49afbd0f878fe2__trial_1", # :["T1036", "cmd", ".pdf"]}, # since the description says it change .pdf to .dll, those pdf and changed dll should have the same name before "." (e.g. abc.pdf --> abc.dll) // Found "T1036" in log entries

   ]


   ''' (2) Get the 'index' to 'all corresponding log-entries' mapping '''

   indices = status_0__97_techniques__list
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


         indices__to__log_entries_of_caldera_technique_process_and_its_descendents__dict[index] = caldera_technique_process_and_descendents_log_entries



   ''' (6) Finally, group the caldera-technique-process and its descendent's (if any) log-entries by entities '''




   indices__to__processThread_to_logentries__dict = dict()

   for index in indices__to__log_entries_of_caldera_technique_process_and_its_descendents__dict:
         
         log_entries_of_caldera_technique_process_and_its_descendents = indices__to__log_entries_of_caldera_technique_process_and_its_descendents__dict[index]

         log_entries_of_caldera_technique_process_and_its_descendents__with_EntityInfo = \
                              get_log_entries_with_entity_info( log_entries_of_caldera_technique_process_and_its_descendents )

         processThread_to_logentries_dict = group_log_entries_by_processThreads( log_entries_of_caldera_technique_process_and_its_descendents__with_EntityInfo )

         indices__to__processThread_to_logentries__dict[index] = processThread_to_logentries_dict


   # ''' (7) Save the results '''

   results_dirpath = "/data/d1/jgwak1/tabby/SUNYIBM_ExplainableAI_2nd_Year_JY/Task_1__Behavior_identification_and_intention_learning/1_0__Identify_Behavioral_Events/grouping_results"
   if not os.path.exists(results_dirpath):
      raise RuntimeError(f"{results_dirpath} does not exist!")


   for index in indices__to__processThread_to_logentries__dict:

      results_fpath = os.path.join(results_dirpath, f"processThread_to_logentries__of__caldera_technique_and_descendent_procs__{index}__{datetime.now().strftime('%Y-%m-%d_%H_%M_%S')}.json")

      with open(results_fpath, "w") as json_fp:
            json.dump( indices__to__processThread_to_logentries__dict[index], json_fp )