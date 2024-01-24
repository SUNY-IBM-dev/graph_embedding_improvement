''' 
JY @ 2024-1-23: Following functions are from/ or adjusted based on functions in 
                tabby/graph_embedding_improvement_JY_git/analyze_at_data_level/Event_and_Entity_Level_Artifactual_Thread_Check/1_0__Identify_Behavioral_Events/helper_funcs.py
'''

EventID_to_RegEventName_dict =\
{
"EventID(1)":"CreateKey", 
"EventID(2)":"OpenKey",
"EventID(3)":"DeleteKey", 
"EventID(4)":"QueryKey", 
"EventID(5)":"SetValueKey", 
"EventID(6)":"DeleteValueKey", 
"EventID(7)":"QueryValueKey",  
"EventID(8)":"EnumerateKey", 
"EventID(9)":"EnumerateValueKey", 
"EventID(10)":"QueryMultipleValueKey",
"EventID(11)":"SetinformationKey", 
"EventID(13)":"CloseKey", 
"EventID(14)":"QuerySecurityKey",
"EventID(15)":"SetSecurityKey", 
"Thisgroupofeventstrackstheperformanceofflushinghives": "RegPerfOpHiveFlushWroteLogFile",
}


from datetime import datetime


def timestamp_conversion(logentry_timestamp_raw_str : str):

      decimal_places = 6 # precision to allow -- datetime.strptime can only handle upto 6
      precision_dot_pos = logentry_timestamp_raw_str.find('.')
      if precision_dot_pos != -1: truncated_logentry_timestamp_raw_str = logentry_timestamp_raw_str[:precision_dot_pos + 1 + decimal_places]
      else: truncated_logentry_timestamp_raw_str = logentry_timestamp_raw_str  # No dot found           

      if truncated_logentry_timestamp_raw_str[-1].lower() == 'z': # drop the trailing Z if there is
         truncated_logentry_timestamp_raw_str = truncated_logentry_timestamp_raw_str[:-1]
      
      if truncated_logentry_timestamp_raw_str.rfind('-') > 10:
         # e.g. '2023-12-19T14:23:07.31751-'
         trailing_dash_pos = truncated_logentry_timestamp_raw_str.rfind('-')
         truncated_logentry_timestamp_raw_str = truncated_logentry_timestamp_raw_str[:trailing_dash_pos]

      if truncated_logentry_timestamp_raw_str.find('.') == -1:
         # e.g. '2023-12-19T14:23:52'
         logentry_timestamp = datetime.strptime(truncated_logentry_timestamp_raw_str, '%Y-%m-%dT%H:%M:%S%f')
      else:
         logentry_timestamp = datetime.strptime(truncated_logentry_timestamp_raw_str, '%Y-%m-%dT%H:%M:%S.%f')
      return logentry_timestamp



def get_subgraph_process_tree_pids( es_index__all_log_entries : list,
                                    subgraph_root_pid : int  ) -> dict:

         # JY @ 2023-10-25 : Get events that are descendent of "splunkd"
         #                   Need to write code that first figures out the
         #                   dependencies (root == splunkd process and process-tree)
         #     First write a loop for identifying that.


         if type(subgraph_root_pid) != int:
            subgraph_root_pid = int(subgraph_root_pid)

         subgraph_root_and_descendent_pids = dict()
         subgraph_root_process_found = False
        
         num_entries = len(es_index__all_log_entries)
         cnt = 0
         for i, log_entry in enumerate(es_index__all_log_entries):
            cnt += 1
            print(f"get_subgraph_process_tree_pids() -- {cnt}/{len(es_index__all_log_entries)}")
            logentry_TaskName = log_entry.get('_source', {}).get('EventName')
            logentry_ProcessID = log_entry.get('_source', {}).get('ProcessID')
            logentry_ThreadID = log_entry.get('_source', {}).get('ThreadID')
            logentry_ProcessName = log_entry.get('_source', {}).get('ProcessName')
            logentry_ProviderName = log_entry.get('_source', {}).get('ProviderName')
            logentry_ProviderGuid = log_entry.get('_source', {}).get('ProviderGuid')
            logentry_XmlEventData = log_entry.get('_source', {}).get('XmlEventData')

            logentry_event_processing_timestamp_raw_str = log_entry.get('_source', {}).get('@timestamp')
            logentry_TimeStamp_raw_str = log_entry.get('_source', {}).get('TimeStamp')
            # convert logentry_timestamp raw-string (e.g. '2023-11-08T17:14:37.327690900Z') into a datetime object



            logentry_processing_timestamp = timestamp_conversion(logentry_event_processing_timestamp_raw_str)
            logentry_timestamp = timestamp_conversion(logentry_TimeStamp_raw_str)

            # ==============================================================================================
            # 1. Get the PID of "splunkd.exe"
            #    Could utilize the 'json' file in "caldera/etw/tmp", 
            #    but there wre incidents where it is incorrect.
            #    So just capture the first entry with ProcessName of "splunkd"
            #    and get the PID
            # if ("splunkd" in logentry_ProcessName):
            #     print()
                # [x for x in es_index__all_log_entries if 'splunkd' in x['_source']['ProcessName'] ]

            if ( logentry_ProcessID == subgraph_root_pid ) and ( subgraph_root_process_found == False ):
               subgraph_root_and_descendent_pids[logentry_ProcessID] = {"ProcessName": logentry_ProcessName,
                                                                        "ParentProcessID" : "no-need-to-collect",
                                                                        "ProcessID" : logentry_ProcessID,
                                                                        "FormattedMessage": "no-need-to-collect",
                                                                        "Timestamp": logentry_timestamp,
                                                                        "@timestamp": logentry_processing_timestamp
                                                                        }
               subgraph_root_process_found = True
            # ==============================================================================================

            # 2. Record the descendent processes of subgraph-root process
            if ("ProcessStart" in logentry_TaskName) and ( subgraph_root_process_found == True ):
               
               ProcessStart_Event_ParentProcessID = int(logentry_XmlEventData['ParentProcessID'].replace( "," , "" ))
               ProcessStart_Event_ChildProcessID = int(logentry_XmlEventData['ProcessID'].replace( "," , "" ))
               ProcessStart_Event_FormattedMessage = logentry_XmlEventData['FormattedMessage']

               if ProcessStart_Event_ParentProcessID in subgraph_root_and_descendent_pids:
                  subgraph_root_and_descendent_pids[ ProcessStart_Event_ChildProcessID ] = {"ProcessName": "N/A",
                                                                                            "ParentProcessID" : ProcessStart_Event_ParentProcessID,
                                                                                            "ChildProcessID" : ProcessStart_Event_ChildProcessID,
                                                                                            "FormattedMessage": ProcessStart_Event_FormattedMessage,
                                                                                            "Timestamp": logentry_timestamp,
                                                                                            "@timestamp": logentry_processing_timestamp
                                                                                             }
            # ==============================================================================================
            # 3. Try to get the ProcessName of descendent-processes of subgraph root 
            #    (empirically, could get 'conhost' but others hard to get -- values are N/A )

            if (logentry_ProcessID in subgraph_root_and_descendent_pids) and (subgraph_root_and_descendent_pids[logentry_ProcessID]["ProcessName"] == "N/A"):
               subgraph_root_and_descendent_pids[logentry_ProcessID]["ProcessName"] = logentry_ProcessName
         

            # if cnt == 254277:
            #     print()
            # print(f"entry process .. {cnt} / {num_entries}", flush=True)

         return subgraph_root_and_descendent_pids


def get_log_entries_of_process_of_interest_and_descendents( es_index__all_log_entries : list,
                                                            process_of_interest_and_its_descendents ) -> list:


         process_of_interest_and_descentdents_log_entries = []
         cnt = 0
         for i, log_entry in enumerate(es_index__all_log_entries):
            cnt += 1
            print(f"get_log_entries_of_process_of_interest_and_descendents() -- {cnt}/{len(es_index__all_log_entries)}")

            logentry_ProcessID = log_entry.get('_source', {}).get('ProcessID')
            logentry_TaskName = log_entry.get('_source', {}).get('EventName')

            if logentry_ProcessID in process_of_interest_and_its_descendents:
                  # ==============================================================================================
                  # tasknames to skip
                  # -- based on : /data/d1/jgwak1/tabby/STREAMLINED_DATA_GENERATION_MultiGraph/STEP_2_Benign_NON_TARGETTED_SUBGRAPH_GENERATION_GeneralLogCollection_subgraphs/model_v3_PW/FirstStep.py
                  if logentry_TaskName.lower() in ["operationend", "namedelete"]: 
                     continue
                  # ==============================================================================================

                  # Also replace log-entry's timestamp attribute into datetime-object for later conveninces
                  logentry_timestamp_raw_str = log_entry.get('_source', {}).get('TimeStamp')

                  logentry_event_processing_timestamp_raw_str = log_entry.get('_source', {}).get('@timestamp')

                  # convert logentry_timestamp raw-string (e.g. '2023-11-08T17:14:37.327690900Z') into a datetime object

                  # decimal_places = 6 # precision to allow -- datetime.strptime can only handle upto 6
                  # precision_dot_pos = logentry_timestamp_raw_str.find('.')
                  # if precision_dot_pos != -1: truncated_logentry_timestamp_raw_str = logentry_timestamp_raw_str[:precision_dot_pos + 1 + decimal_places]
                  # else: truncated_logentry_timestamp_raw_str = logentry_timestamp_raw_str  # No dot found           

                  # if truncated_logentry_timestamp_raw_str[-1].lower() == 'z': # drop the trailing Z if there is
                  #    truncated_logentry_timestamp_raw_str = truncated_logentry_timestamp_raw_str[:-1]

                  # if truncated_logentry_timestamp_raw_str.rfind('-') > 10:
                  #    # e.g. '2023-12-19T14:23:07.31751-'
                  #    trailing_dash_pos = truncated_logentry_timestamp_raw_str.rfind('-')
                  #    truncated_logentry_timestamp_raw_str = truncated_logentry_timestamp_raw_str[:trailing_dash_pos]

                  # if truncated_logentry_timestamp_raw_str.find('.') == -1:
                  #    # e.g. '2023-12-19T14:23:52'
                  #    logentry_timestamp = datetime.strptime(truncated_logentry_timestamp_raw_str, '%Y-%m-%dT%H:%M:%S%f')
                  # else:
                  #    logentry_timestamp = datetime.strptime(truncated_logentry_timestamp_raw_str, '%Y-%m-%dT%H:%M:%S.%f')




                  # logentry_timestamp_datetime_object = datetime.strptime(truncated_logentry_timestamp_raw_str,
                  #                                                        '%Y-%m-%dT%H:%M:%S.%f')

                  logentry_timestamp = timestamp_conversion(logentry_timestamp_raw_str)
                  logentry_processing_timestamp = timestamp_conversion(logentry_event_processing_timestamp_raw_str)


                  log_entry['_source']['TimeStamp'] = logentry_timestamp
                  log_entry['_source']['@timestamp'] = logentry_processing_timestamp


                  # ==============================================================================================

                  process_of_interest_and_descentdents_log_entries.append(log_entry)


         return process_of_interest_and_descentdents_log_entries




def group_log_entries_by_processThreads(log_entries : list) -> dict:
    
    # JY @ 2023-11-09: Correcntess of this function seems OK
    #                  Cross-checked with Elastic-search Index 

    processThread_to_logentries_dict = dict()

    # first group log-entries by process
    # then group log-entries by thread

    ''' 
    JY @ 2023-12-27:
       Did not take into account cases where same process-id is re-used.
       Did not take into account the cases where same thread-id is re-used under same process-id

       Could later incorporate by processstart and threadstart
       Not the highest priority since the goal here is to see ordered events of a thread to see if there is a pattern (identify artifactual threads?)
       and it is more rare that a same thread-id is reused under the same re-used process-id
    '''


    cnt = 0
    for log_entry in log_entries:
        cnt += 1
        print(f"group_log_entries_by_processThreads() -- {cnt}/{len(log_entries)}", flush=True)        


        log_entry_pid = f"pid_{log_entry['_source']['ProcessID']}"
        log_entry_tid = f"tid_{log_entry['_source']['ThreadID']}"

        log_entry['_source']['TimeStamp'] = str(log_entry['_source']['TimeStamp']) # for json -- can convert a string objcet into datetime object later
        log_entry['_source']['@timestamp'] = str(log_entry['_source']['@timestamp']) # for json




        logentry_EventName = log_entry['_source']['EventName'] 

        if logentry_EventName in EventID_to_RegEventName_dict:
           logentry_EventName = EventID_to_RegEventName_dict[logentry_EventName]



        # Following can serve as key-info for log-entry:
        #  log_entry['PROVIDER_SPECIFIC_ENTITY']
        #  log_entry['_source']['EventName']
        #  log_entry['_source']['TimeStamp']
        #  log_entry['_source']['XmlEventData']

         # this order is more readable
        log_entry_key_info = {
            "EventName" : logentry_EventName,
            "TimeStamp": log_entry['_source']['TimeStamp'],
            "PROVIDER_SPECIFIC_ENTITY" : log_entry['PROVIDER_SPECIFIC_ENTITY'],
            # "XmlEventData": log_entry['_source']['XmlEventData']
        }
        log_entry_key_info_fstring = f"{log_entry_key_info}"
        delimiter_fstring = "-"*150 # for easier reading

        if log_entry_pid in processThread_to_logentries_dict: # if log-entry's pid exists as a key

           if log_entry_tid in processThread_to_logentries_dict[log_entry_pid]:
               # under this process, there exists a key for the thread,
               # so just append it 

               processThread_to_logentries_dict[log_entry_pid][log_entry_tid].append(log_entry_key_info)

               # processThread_to_logentries_dict[log_entry_pid][log_entry_tid].append(log_entry_key_info_fstring)
               # processThread_to_logentries_dict[log_entry_pid][log_entry_tid].append(delimiter_fstring)

           else:
               # under this process, first event for this process-thread
               # so create space for it, and append the first event
               processThread_to_logentries_dict[log_entry_pid][log_entry_tid] = list()
               # processThread_to_logentries_dict[log_entry_pid][log_entry_tid].append(log_entry_key_info_fstring)
               # processThread_to_logentries_dict[log_entry_pid][log_entry_tid].append(delimiter_fstring)
               processThread_to_logentries_dict[log_entry_pid][log_entry_tid].append(log_entry_key_info)


        else:
            # if log-entry's pid key is not populated yet,
            # obviously there is no corresponding space for the process-thread
            # so create the space and append the first log-entry for that process-thread             
            processThread_to_logentries_dict[log_entry_pid] = dict()
            processThread_to_logentries_dict[log_entry_pid][log_entry_tid] = list()
            # processThread_to_logentries_dict[log_entry_pid][log_entry_tid].append( log_entry_key_info_fstring )
            # processThread_to_logentries_dict[log_entry_pid][log_entry_tid].append(delimiter_fstring)
            processThread_to_logentries_dict[log_entry_pid][log_entry_tid].append(log_entry_key_info)



    # Added by JY @ 2023-12-27 Wrap "processThread_to_logentries_dict[log_entry_pid][log_entry_tid]" with NoIndent
    #           --> https://stackoverflow.com/questions/13249415/how-to-implement-custom-indentation-when-pretty-printing-with-the-json-module
               # obj = {
               #   "layer1": {
               #     "layer2": {
               #       "layer3_2": "string", 
               #       "layer3_1": NoIndent([{"y": 7, "x": 1}, {"y": 4, "x": 0}, {"y": 3, "x": 5}, {"y": 9, "x": 6}])
               #     }
               #   }
               # }
               # print json.dumps(obj, indent=2, cls=NoIndentEncoder)


   #  for pid, tid_to_logentrylist_dict in processThread_to_logentries_dict.items():
   #      for tid, logentrylist in tid_to_logentrylist_dict.items():
   #          processThread_to_logentries_dict[pid][tid] = NoIndent(logentrylist)

    # returns dict of dict 
    return processThread_to_logentries_dict



def get_log_entries_with_entity_info( log_entries : list ) -> list:

   ''' TODO / TOTHINK '''
   # JY @ 2023-11-09: 
   #     QUESTION -- WHICH ENTITY SHOULD I ASSOCIATE A PROCESS-PROVIDER EVENT TO?
   #     
   #     For PROCESS-PROVIDER events, I guess entites can be process/thread 
   #     as in CG, a PROCESS-PROVIDER event happens between the 
   #     log-entry thread-node and process-node/thread-node
   # 
   #     perhaps the description for the entity-(process/thread)-node can be the
   #     relation with the log-entry thread-node (e.g. parent/child process / sibling-thread / itself, etc)
   #     -- for details, might need to refer to the FirstStep.py


   FILE_PROVIDER = "EDD08927-9CC4-4E65-B970-C2560FB5C289"
   NETWORK_PROVIDER = "7DD42A49-5329-4832-8DFD-43D979153A88"
   PROCESS_PROVIDER = "22FB2CD6-0E7B-422B-A0C7-2FAD1FD0E716"
   REGISTRY_PROVIDER = "70EB4F03-C1DE-4F73-A051-33D13D5413BD"


   ''' JY @ 2023-11-1: 
            Should the entity be same as UID if we are to utilize CG for explanation?
   
   '''

   log_entries_with_entity_info = []

   # For following,
   # refer to: /data/d1/jgwak1/tabby/STREAMLINED_DATA_GENERATION_MultiGraph/STEP_2_Benign_NON_TARGETTED_SUBGRAPH_GENERATION_GeneralLogCollection_subgraphs/model_v3_PW/FirstStep.py
   fileobject_to_filename_mapping = dict()
   keyobject_to_relativename_mapping = dict()
   

   cnt = 0
   for log_entry in log_entries:
      cnt += 1
      print(f"get_log_entries_with_entity_info() -- {cnt}/{len(log_entries)}", flush=True)        

      logentry_ProviderGuid = log_entry.get('_source', {}).get('ProviderGuid')
      logentry_TaskName = log_entry.get('_source', {}).get('EventName')
      logentry_ProcessID = log_entry.get('_source', {}).get('ProcessID')
      logentry_ThreadID = log_entry.get('_source', {}).get('ThreadID')
      # logentry_XmlEventData = log_entry.get('_source', {}).get('XmlEventData')

      # log_entry['GENERAL_ENTITY'] = f"{logentry_ProcessID}__{logentry_ThreadID}" # this doesn't really make sense 
                                                                                    # to use within sequence

      if logentry_TaskName.lower() in ["operationend", "namedelete"]: 
         continue


      #=============================================================================================================
      if logentry_ProviderGuid == FILE_PROVIDER.lower():
         # Entity associated with a File-Event?
         # --> Probably the 'File'            
         # ----> 'FileName' or 'FileObject', etc. depending on Event (need to resolve 'mapping')
         # ----> also, keep track of process & thread that carried out the file-event
         



         logentry_FileName = log_entry.get('_source', {}).get('XmlEventData').get('FileName') 
         logentry_FileObject = log_entry.get('_source', {}).get('XmlEventData').get('FileObject') 


         if logentry_TaskName.lower() in {"create", "createnewfile"}:
            # JY: 'create' and 'createnewfile' provides both 'logentry_FileObject' and 'logentry_FileName'
            fileobject_to_filename_mapping[logentry_FileObject] = str(logentry_FileName)
            log_entry['PROVIDER_SPECIFIC_ENTITY'] = f"{logentry_FileObject}__{str(logentry_FileName)}"


         elif logentry_TaskName.lower() in {"close"}:
            # JY: "close" appear to provide only 'logentry_FileObject'
            logentry_FileName = fileobject_to_filename_mapping.get(logentry_FileObject, "None")
            log_entry['PROVIDER_SPECIFIC_ENTITY'] = f"{logentry_FileObject}__{logentry_FileName}"

         else: # all other tasknames
            # JY: ALL OTHER Opcodes appear to provide only 'logentry_FileObject'
            logentry_FileName = fileobject_to_filename_mapping.get(logentry_FileObject, "None")
            log_entry['PROVIDER_SPECIFIC_ENTITY'] = f"{logentry_FileObject}__{logentry_FileName}"



      #=============================================================================================================
      if logentry_ProviderGuid == REGISTRY_PROVIDER.lower():
         # Entity associated with a Registry-Event?
         # --> Probably the 'Registry-Key'            
         # ----> 'RelativeName' or 'KeyObject', etc. depending on Event (need to resolve 'mapping')
         # ----> also, keep track of process & thread that carried out the file-event


         logentry_OpcodeName = log_entry.get('_source', {}).get('OpcodeName')

         logentry_KeyObject = log_entry.get('_source', {}).get('XmlEventData').get('KeyObject') 
         logentry_RelativeName = log_entry.get('_source', {}).get('XmlEventData').get('RelativeName') 

         logentry_KeyName = log_entry.get('_source', {}).get('XmlEventData').get('KeyName') # ?

         #if logentry_TaskName in {'EventID(1)','EventID(2)'}:---> option 1
         if logentry_OpcodeName in {"CreatKey","OpenKey"}: #-----> option2 
            # JY: 'CreateKey' and 'OpenKey' provides both 'logentry_KeyObject' and 'logentry_RelativeName'

            keyobject_to_relativename_mapping[logentry_KeyObject] = str(logentry_RelativeName)

            log_entry['PROVIDER_SPECIFIC_ENTITY'] = f"{logentry_KeyObject}__{str(logentry_RelativeName)}"  # add to entity

         #elif logentry_TaskName == 'EventID(13)' --->option1 
         elif logentry_OpcodeName == "CloseKey": #--->option2

            # JY: "CloseKey" appear to provide only 'logentry_KeyObject'

            logentry_RelativeName = keyobject_to_relativename_mapping.get(logentry_KeyObject, "None")

            log_entry['PROVIDER_SPECIFIC_ENTITY'] = f"{logentry_KeyObject}__{logentry_RelativeName}"

         else: # ALL OTHER Opcodes  
            # JY: ALL OTHER Opcodes appear to provide only 'logentry_KeyObject'

            logentry_RelativeName = keyobject_to_relativename_mapping.get(logentry_KeyObject, "None")

            log_entry['PROVIDER_SPECIFIC_ENTITY'] = f"{logentry_KeyObject}__{logentry_RelativeName}"

      #=============================================================================================================

      if logentry_ProviderGuid == NETWORK_PROVIDER.lower():
         # Entity associated with a Network-Event?
         # --> Probably the 'IP-address' that 'this-machine' communicated with            
         # ----> 'daddr' depending on Event 
         # ----> also, keep track of process & thread that carried out the file-event

         logentry_destaddr = log_entry.get('_source', {}).get('XmlEventData').get('daddr') 

         log_entry['PROVIDER_SPECIFIC_ENTITY'] = f"{logentry_destaddr}"


      #=============================================================================================================
      # Added by JY @ 2023-12-14
      if logentry_ProviderGuid == PROCESS_PROVIDER.lower():
         # Entity associated with a Process-Event?
         # --> Process and Thread that took action.

         logentry_imagename = log_entry.get('_source', {}).get('XmlEventData').get('ImageName') 

         log_entry['PROVIDER_SPECIFIC_ENTITY'] = logentry_imagename

         # Perhaps imagename? -- I think this mostly happens when imageload-and-unload events -- so don't use it.
         # logentry_ImageName = log_entry.get('_source_XmlEventData_ImageName', 'None')

         # if logentry_ImageName != "None":
         #    print()
         pass

      #=============================================================================================================
      log_entries_with_entity_info.append(log_entry)


   

   # [ f"{x['_source_EventName']}__{x['PROVIDER_SPECIFIC_ENTITY']}" for x in log_entries_with_entity_info ] 

   # [ {"ProcessID": x['_source_ProcessID'],
   #    "ThreadID": x['_source_ThreadID'],
   #    "TaskName": x['_source_EventName'],
   #    "logentry_OpcodeName": x['_source_OpcodeName'],
   #    "PROVIDER_SPECIFIC_ENTITY": x['PROVIDER_SPECIFIC_ENTITY']} 
   # for x in log_entries_with_entity_info ] 
      
      # x['_source_EventName']}__{x['PROVIDER_SPECIFIC_ENTITY']}" for x in log_entries_with_entity_info ] 

   return log_entries_with_entity_info
