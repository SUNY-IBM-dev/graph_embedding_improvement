import shutil
import socket
import csv
import igraph as g
from igraph import *
from igraph.clustering import *
import hashlib
import uuid
import networkx as nx
#import graph_tools
import json
from matplotlib import pyplot as plt
from itertools import product
import numpy as np
import ast
from elasticsearch import Elasticsearch
import sys
import os
import pickle
from datetime import datetime
from multiprocessing import Process, process




def from_elastic(idx):
    if idx == 'small_log2':
        dtype = '_doc'
    else:
        dtype = 'some_type'
    es = Elasticsearch(['http://panther.cs.binghamton.edu:9200'],timeout = 300)
    es.indices.put_settings(index=idx,
                            body={'index':{
                            'max_result_window':999999999}})
    
    necessary_attributes = ["ProviderId", 
                            "Task Name", "Opcode",
                            "ProcessId", "ProcessID", "ThreadId", "ThreadID", 
                            "CreateTime", "TimeStamp",
                            # File-Specific
                            "FileName", "FileObject",
                            # Network-Specific
                            "daddr",
                            # Process-Specific
                            "ImageName",
                            # Registry-Specific
                            'KeyObject', 'RelativeName',
                            ]
    
    
    query_body = {
        "query": {  "match_all": {} },
        "_source": necessary_attributes, 
        "sort" : [ { "TimeStamp" : { "order" : "asc" } } ]
    }

    # USE SCROLL?
    scroll_size = 10000
    scroll_time = '1m'

    result = es.search(index = idx, doc_type = dtype, body = query_body, 
                      scroll='1m',
                      size = 99999999)
    all_hits = result['hits']['hits']
    return all_hits

def get_host():
    #hostname = socket.gethostname()
    ## getting the IP address using socket.gethostbyname() method
    #ip_address = socket.gethostbyname(hostname)
    ## printing the hostname and ip_address
    #print(f"Hostname: {hostname}")
    #print(f"IP Address: {ip_address}")
    hostname = "ocelot" # just set it always ocelot directly.
    return hostname

def get_uid(hash, hostname, firststep_hash_debugging_mode : bool):
    
    if firststep_hash_debugging_mode:
        return str(hash)
    
    else:
        hash1 = hashlib.md5(hash.encode()).hexdigest()
        hash2 = hashlib.md5(hostname.encode()).hexdigest()
        h1 = str(bin(int(hash1, 16)).zfill(8))
        h2 = str(bin(int(hash2, 16)).zfill(8))
        out = h1[:92] + h2[92:]
        decimal_representation = int(out, 2)
        hexadecimal_string = hex(decimal_representation)
        hexadecimal_string = hexadecimal_string[2:]
        out = str(uuid.UUID(hexadecimal_string))

        # Following added by JY @ 2023-1-18
        # For minimal identification
        if "<<PROC-NODE>>" in hash:
            out = "PROC-NODE_"+out 
        if "<<REG-NODE>>" in hash:
            out = "REG-NODE_"+out
        if "<<NET-NODE>>" in hash:
            out = "NET-NODE_"+out 
        if "<<FILE-NODE>>" in hash:
            out = "FILE-NODE_"+out

        if "<<PROC-EDGE>>" in hash:
            out = "PROC-EDGE_"+out 
        if "<<REG-EDGE>>" in hash:
            out = "REG-EDGE_"+out
        if "<<NET-EDGE>>" in hash:
            out = "NET-EDGE_"+out 
        if "<<FILE-EDGE>>" in hash:
            out = "FILE-EDGE_"+out

        if "<<THREAD>>" in hash:
            out = "THREAD_" + out

        return out




############################################################################################################################################################################
Process_StartTime_dict = {} # Global variable ; Mapping of Process and its starttime ; {pid: startime} ; e.g. {6600: 1324324324}
ProcessThread_StartTime_dict = {}  # Global variable ; Mapping of Thread(that belongs to some process) and its starttime. ; { pid : { tid : starttime } }; e.g. { 6600: {12097: 1212332}}


def set_Process_StartTime_dict(PID, process_starttime):            
    
    # if PROCESSSTOP , assign "stop"

    Process_StartTime_dict[PID] = process_starttime # New or Overwrite.
    ProcessThread_StartTime_dict[PID] = dict() # If PROCESSSTART, set an empty dict for new-process in ProcessThread_StartTime_dict in preparation for its Threads.
                                               # ProcessThread_StartTime_dict[PID] = dict()

    return



def set_ProcessThread_StartTime_dict(PID, TID, thread_starttime):
    # TODO: Think about/ Identify exceptions and handle those.

    if PID not in Process_StartTime_dict: # If Process started before log-collection, but Thread is started after log-collection

        # Since PID will not be in 'Process_StartTime_dict', PID will also not be in 'ProcessThread_StartTime_dict' which is correct.
        # However, while PID is not in 'Process_StartTime_dict' indicating that this Process is started before log-collection,
        # 'Process_StartTime_dict' can still have { PID : {TID: thread_starttime} } indicating that this Thread that belongs to PID started after log-collection.
        ProcessThread_StartTime_dict[PID] = {TID: thread_starttime}

    else: # If Process is created after log-collection, and Thread (naturally) started after log-collection
        ProcessThread_StartTime_dict[PID][TID] = thread_starttime

    return


#def get_ProcessStartTime_dict():
#    return Process_StartTime_dict
#def get_ProcessThread_StartTime_dict():
#    return ProcessThread_StartTime_dict


############################################################################################################################################################################


#####################################################################################################################################################################
## Node Process -------------------------------------------------------------
def get_proc_info(log_entry, host, 
                  proc_uid_list, proc_thread_uid_list, proc_edge_uid_list, 
                  proc_node_dict, proc_thread_dict, proc_edge_dict,
                  
                  firststep_hash_debugging_mode = False
                  ):

    ''' TODO: WHEN WRITING THIS FUNCTION, CHECK FOR ALL POSSIBLE CASES IN A ACTUALL LOG! 
              e.g. "chrome_browser_all_providers.txt"
    '''


    # QUESTION: HOW TO TAKE CARE OF INCOMING T (External T---> P)? HOW WILL THE CORRESPONDING LOG-ENTRY LOOK LIKE INTERMS OF TASKNAME AND FEATURE VALUES?
    #           IF EXISTS, WHAT SHOULD BE DONE IN THIS STAGE TO REPRESENT IT CORRECTLY IN THE CG?


    ##################################################################################################################################################
    logentry_TaskName = log_entry.get('_source', {}).get('Task Name')             # feature
    logentry_Opcode = log_entry.get('_source', {}).get('Opcode')    
    logentry_CreateTime = log_entry.get('_source', {}).get('CreateTime')
    logentry_ThreadId = str(log_entry.get('_source', {}).get('ThreadId'))
    logentry_ThreadID = str(log_entry.get('_source', {}).get('ThreadID'))


    logentry_ProcessId = str(log_entry.get('_source', {}).get('ProcessId'))
    logentry_TimeStamp = log_entry.get('_source', {}).get('TimeStamp')
    logentry_ProcessID = str(log_entry.get('_source', {}).get('ProcessID'))
    logentry_ImageName = log_entry.get('_source', {}).get('ImageName')


    # logentry_ImageSize = log_entry.get('_source', {}).get('ImageSize')                    #feature
    # logentry_HandleCount = log_entry.get('_source', {}).get('HandleCount')                    #feature
    # logentry_TimeDateStamp = log_entry.get('_source', {}).get('TimeDateStamp')    
    # logentry_StatusCode = log_entry.get('_source', {}).get('StatusCode')
    # logentry_ProcessTokenElevationType = log_entry.get('_source', {}).get('ProcessTokenElevationType') # feature
    # logentry_ProcessTokenIsElevated = log_entry.get('_source', {}).get('ProcessTokenIsElevated')
    # logentry_MandatoryLabel = log_entry.get('_source',{}).get('MandatoryLabel') #feature
    # logentry_PackageFullName = log_entry.get('_source',{}).get('PackageFullName') # feature
    # logentry_PackageRelativeAppId = log_entry.get('_source',{}).get('PackageRelativeAppId')
    # logentry_TokenElevationType = log_entry.get('_source',{}).get('TokenElevationType') # feature
    # logentry_ReadOperationCount = log_entry.get('_source',{}).get('ReadOperationCount') # feature
    # logentry_WriteOperationCount = log_entry.get('_source',{}).get('WriteOperationCount') # feature
    # logentry_ReadTransferKiloBytes = log_entry.get('_source',{}).get('ReadTransferKiloBytes') # feature
    # logentry_WriteTransferKiloBytes= log_entry.get('_source',{}).get('WriteTransferKiloBytes') # feature
    # logentry_SubProcessTag = log_entry.get('_source', {}).get('SubProcessTag')             #feature
    ##################################################################################################################################################

    #============================================================================================================================================================================================================
    if logentry_TaskName == "PROCESSSTART": 
        
        # JY @ 2023-1-9 :  Note that for a PROCESSSTART event,
        #                  the newly created process's PID is "ProcessID" not the "ProcessId"
        #                  'ProcessId' is the 'log-entry PID' and the 'parent-process-id'.
        set_Process_StartTime_dict( PID = logentry_ProcessID, process_starttime = logentry_CreateTime )
        #Process_StartTime_dict = get_ProcessStartTime_dict()
        #ProcessThread_StartTime_dict = get_ProcessThread_StartTime_dict()       


        # If PROCESSSTART event, 
        # this log-entry should generate the following:
        # (Child Process)---Edge---(Parent Process Thread)


        # (Child-Process node)
        # > Just before, Child-Process PROCESSSTART is set in 'Process_StartTime_dict' which can be used to compute hash of child-process 
        ChildProcess_CreationTime = Process_StartTime_dict[ logentry_ProcessID ] # For PROCESS-START log-entry, ProcessID corresponds to ChildProcess.
        #proc_hash = str( logentry_ProcessID ) + str( ChildProcess_CreationTime )
        proc_hash = f"<<PROC-NODE>>(PID):{logentry_ProcessID}_(CT):{ChildProcess_CreationTime}"
        proc_uid = get_uid(proc_hash, host, firststep_hash_debugging_mode)
        proc_uid_list.append(proc_uid)
        proc_node_dict[proc_uid] = {'ProcessId': logentry_ProcessID } # JY @ 2023-1-10: If 'ProcessId' is what indicates this process-node's uid, 
                                                                      #                 then here its value should be 'ProcessID' not 'ProcessId'   (Discuss this)



        # (Parent-Process Thread node)
        # For PROCESSSTART, parent-process is ProcessId == ParentProcessId
        thread_hash = None
        thread_uid = None
        if logentry_ProcessId in Process_StartTime_dict: # "If parent-process was process-started after log-collection." 
                                                         #
                                                         #    NOTE: Whether this parent-process reused a stopped process's PID should not affect correctness.
                                                         #          Reason is in Process_StartTime_dict, the old starttime value of the stopped process (which used the same PID) 
                                                         #          is overwritten by this parent-process's starttime during PROCESSSTART of this parent-process (which reuses the same PID).
           
            if logentry_ThreadId in ProcessThread_StartTime_dict[logentry_ProcessId]: # "And If Thread of parent-process was started after log-collection"  
                                                                                      #
                                                                                      #    NOTE: Whether this ThreadId is reused (stopped and started) within this parent-process should not affect correctness.
                                                                                      #          Reason is in ProcessThread_StartTime_dict,
                                                                                      #          the old startime value of the stopped thread (which used same ThreadId) within this parent-process
                                                                                      #          is overwritten by the starttime of newly started thread (which reuses the same ThreadId within this parent-process).
                                                                                      #          during its THREADSTART.
                                                                                      #          i.e., 'ProcessThread_StartTime_dict' will be up-to-date (also for 'Process_StartTime_dict')
                
                #thread_hash = ( str(logentry_ThreadId) + str(ProcessThread_StartTime_dict[logentry_ProcessId][logentry_ThreadId]) )  +  ( str(logentry_ProcessId)+str( Process_StartTime_dict[logentry_ProcessId] ) )                
                thread_hash = f"<<THREAD>>(TID):{logentry_ThreadId}_(TS):{ProcessThread_StartTime_dict[logentry_ProcessId][logentry_ThreadId]}__(PID):{logentry_ProcessId}_(CT):{Process_StartTime_dict[logentry_ProcessId]}"
                
                
            # else: "logentry_ThreadId NOT in ProcessThread_StartTime_dict[logentry_ProcessId]" indicates this parent-process(ProcessId) does not own this thread(ThreadId),
            #       which is contradictiory since this parent-process started after the log-collection-start and for this log-entry to exist (with this ProcessId and ThreadId), 
            #       this parent-process must have started this thread.
            else:
                # thread_hash = "PROC-DIDITREALLYCOMEHERE-PROCESSSTART"

                thread_hash = f"<<THREAD>>(TID):{logentry_ThreadId}_(TS):N/A__(PID):{logentry_ProcessId}_(CT):{Process_StartTime_dict[logentry_ProcessId]}"

        else: # "If parent-process was process-started before log-collection"

            if logentry_ThreadId in ProcessThread_StartTime_dict.get(logentry_ProcessId, {}): # "And If Thread of parent-process was started after log-collection"  
                                                                                      #
                                                                                      #    NOTE: Whether this ThreadId is reused (stopped and started) within this parent-process should not affect correctness.
                                                                                      #          Reason is in ProcessThread_StartTime_dict,
                                                                                      #          the old startime value of the stopped thread (which used same ThreadId) within this parent-process
                                                                                      #          is overwritten by the starttime of newly started thread (which reuses the same ThreadId within this parent-process).
                                                                                      #          during its THREADSTART.
                                                                                      #          i.e., 'ProcessThread_StartTime_dict' will be up-to-date (also for 'Process_StartTime_dict')
                #thread_hash = ( str(logentry_ThreadId) + str(ProcessThread_StartTime_dict[logentry_ProcessId][logentry_ThreadId]) )  +  str(logentry_ProcessId)
                thread_hash = f"<<THREAD>>(TID):{logentry_ThreadId}_(TS):{ProcessThread_StartTime_dict[logentry_ProcessId][logentry_ThreadId]}__(PID):{logentry_ProcessId}_(CT):N/A"
            
            else: # "And If Thread of parent-process started before log-collection"        
                #thread_hash = str(logentry_ThreadId) + str(logentry_ProcessId)
                thread_hash = f"<<THREAD>>(TID):{logentry_ThreadId}_(TS):N/A__(PID):{logentry_ProcessId}_(CT):N/A"


        thread_uid = get_uid(thread_hash,host, firststep_hash_debugging_mode)
        proc_thread_uid_list.append(thread_uid)
        proc_thread_dict[thread_uid]= {'ThreadID': logentry_ThreadID, 'ProcessID': logentry_ProcessID, 
                                       'ThreadId': logentry_ThreadId, 'ProcessId': logentry_ProcessId}


        # (Edge)
        #hashout = str(logentry_ProcessID) + str(logentry_ThreadId) + str(logentry_TimeStamp)   
        hashout = f"<<PROC-EDGE>>(PID):{logentry_ProcessId}_(TID):{logentry_ThreadId}_(TN):{logentry_TaskName}_(OP):{logentry_Opcode}_(TS):{logentry_TimeStamp}"
        edge_uid = get_uid(hashout,host, firststep_hash_debugging_mode)
        proc_edge_uid_list.append(edge_uid)
        proc_edge_dict[edge_uid] = {'ProcessId': logentry_ProcessId, 'ProcessID': logentry_ProcessID, 
                                    'ThreadId': logentry_ThreadId, 'ThreadID': logentry_ThreadID,
                                    'Task Name':logentry_TaskName, 'Opcode':logentry_Opcode, 'ImageName':logentry_ImageName, 
                                    'TimeStamp': logentry_TimeStamp, 'CreateTime': logentry_CreateTime,
                                    
                                    # 'ImageSize':logentry_ImageSize,'HandleCount':logentry_HandleCount,
                                    # 'TimeDateStamp': logentry_TimeDateStamp, 'StatusCode': logentry_StatusCode,
                                    # 'ReadOperationCount':logentry_ReadOperationCount,'WriteOperationCount':logentry_WriteOperationCount, 
                                    # 'ReadTransferKiloBytes': logentry_ReadTransferKiloBytes, 'WriteTransferKiloBytes':logentry_WriteTransferKiloBytes,
                                    # 'ProcessTokenElevationType': logentry_ProcessTokenElevationType, 'ProcessTokenIsElevated': logentry_ProcessTokenIsElevated, 
                                    # 'MandatoryLabel': logentry_MandatoryLabel, 'PackageRelativeAppId': logentry_PackageRelativeAppId,'SubProcessTag': logentry_SubProcessTag,

                                    'PROC-NODE': proc_uid, 
                                    'THREAD-NODE': thread_uid,
                                    }


    #============================================================================================================================================================================================================
    elif logentry_TaskName == "THREADSTART":

        # Observed 2 types of THREADSTARTs (TODO: Check if there are more)
        # > For specific examples check THREADSTART events in chrome-browser user-activity event-log, with ProcessId "6600" and "ThreadId" 3560.


        #-----------------------------------------------------------------------------------------------------------------------------------------
        # THREADSTART Type-1: Process and its Thread THREADSTART-ing its Process's New Thread
        # e.g. Process 6600 and its Thread 3560 THREADSTART-ing Process 6600's New Thread 3812
        #                      [ProcessId]    [ThreadId]    [TaskName]     [ProcessID]   [ThreadID]
        #          log-entry:     6600           3560       THREADSTART        6600         3812
        #
        #     > Then this log-entry should generate the following:
        #       (Process 6600)----Edge----(Thread 3812)

        logentry_ThreadID = log_entry.get('_source', {}).get('ThreadID')
        #Process_StartTime_dict = get_ProcessStartTime_dict()


        if logentry_ProcessId == logentry_ProcessID:    # THREADSTART Type-1: Process and its Thread THREADSTART-ing Process's New Thread

            # Set the THREADSTART information for this new Thread in 'ProcessThread_StartTime_dict'; Get the updated 'ProcessThread_StartTime_dict'
            set_ProcessThread_StartTime_dict( PID = logentry_ProcessId, TID = logentry_ThreadID, thread_starttime= logentry_TimeStamp )   # THREADSTART doesn't have CreateTime, logentry_TimeStamp can still serve as an identifier.
            #ProcessThread_StartTime_dict = get_ProcessThread_StartTime_dict()       
            
            # (Process-node)
            #proc_hash = None
            if logentry_ProcessId in Process_StartTime_dict:    # "If process was process-started after log-collection." 
                #proc_hash = str( logentry_ProcessId ) + str( Process_StartTime_dict[logentry_ProcessId] )
                proc_hash = f"<<PROC-NODE>>(PID):{logentry_ProcessId}_(CT):{Process_StartTime_dict[logentry_ProcessId]}"
            else: # "If process was process-started before log-collection."                       
                #proc_hash = str( logentry_ProcessId )
                proc_hash = f"<<PROC-NODE>>(PID):{logentry_ProcessId}_(CT):N/A"
            proc_uid = get_uid(proc_hash, host, firststep_hash_debugging_mode)
            proc_uid_list.append(proc_uid)
            proc_node_dict[proc_uid] = {'ProcessId': logentry_ProcessId } 


            # (Thread-node)
            thread_hash = None
            thread_uid = None
            if logentry_ProcessId in Process_StartTime_dict: # "If process was process-started after log-collection."
                # "Since just before this new Thread is set with its creation-time in 'ProcessThread_StartTime_dict', no need to check whether ThreadId in 'ProcessThread_StartTime_dict' "
                #thread_hash = ( str(logentry_ThreadID) + str(ProcessThread_StartTime_dict[logentry_ProcessId][logentry_ThreadID]) ) + ( str(logentry_ProcessId)+str( Process_StartTime_dict[logentry_ProcessId] ) )                
                thread_hash = f"<<THREAD>>(TID):{logentry_ThreadID}_(TS):{ProcessThread_StartTime_dict[logentry_ProcessId][logentry_ThreadID]}__(PID):{logentry_ProcessId}_(CT):{Process_StartTime_dict[logentry_ProcessId]}"
            else: # "If process was process-started before log-collection."
                  # "Since just before this new Thread is set with its creation-time in 'ProcessThread_StartTime_dict', no need to check whether ThreadId in 'ProcessThread_StartTime_dict' "
                  #     > In "set_ProcessThread_StartTime_dict()", 
                  #       Already the case of "Child-Process started before log-collection, but Thread started after log-collection, 
                  #                            Thus 'ProcessThread_StartTime_dict' initially not having 'logentry_ProcessId' key"    is handled.
                #thread_hash = ( str(logentry_ThreadID) + str(ProcessThread_StartTime_dict[logentry_ProcessId][logentry_ThreadID]) )  + str(logentry_ProcessId)
                thread_hash = f"<<THREAD>>(TID):{logentry_ThreadID}_(TS):{ProcessThread_StartTime_dict[logentry_ProcessId][logentry_ThreadID]}__(PID):{logentry_ProcessId}_(CT):N/A"
            thread_uid = get_uid(thread_hash,host, firststep_hash_debugging_mode)
            proc_thread_uid_list.append(thread_uid)            
            proc_thread_dict[thread_uid]= {'ThreadID': logentry_ThreadID, 'ProcessID': logentry_ProcessID, 
                                           'ThreadId': logentry_ThreadId, 'ProcessId': logentry_ProcessId}

            # (Edge)
            #hashout = str(logentry_ProcessId) + str(logentry_ThreadID) + str(logentry_TimeStamp)   # keeping this might be fine
            hashout = f"<<PROC-EDGE>>(PID):{logentry_ProcessId}_(TID):{logentry_ThreadId}_(TN):{logentry_TaskName}_(OP):{logentry_Opcode}_(TS):{logentry_TimeStamp}"
            edge_uid = get_uid(hashout,host, firststep_hash_debugging_mode)
            proc_edge_uid_list.append(edge_uid)
            proc_edge_dict[edge_uid] = {'ProcessId': logentry_ProcessId, 'ProcessID': logentry_ProcessID, 
                                        'ThreadId': logentry_ThreadId, 'ThreadID': logentry_ThreadID,
                                        'Task Name':logentry_TaskName, 'Opcode':logentry_Opcode, 'ImageName':logentry_ImageName, 
                                        'TimeStamp': logentry_TimeStamp, 'CreateTime': logentry_CreateTime,
                                        
                                        # 'ImageSize':logentry_ImageSize,'HandleCount':logentry_HandleCount,
                                        # 'TimeDateStamp': logentry_TimeDateStamp, 'StatusCode': logentry_StatusCode,
                                        # 'ReadOperationCount':logentry_ReadOperationCount,'WriteOperationCount':logentry_WriteOperationCount, 
                                        # 'ReadTransferKiloBytes': logentry_ReadTransferKiloBytes, 'WriteTransferKiloBytes':logentry_WriteTransferKiloBytes,
                                        # 'ProcessTokenElevationType': logentry_ProcessTokenElevationType, 'ProcessTokenIsElevated': logentry_ProcessTokenIsElevated, 
                                        # 'MandatoryLabel': logentry_MandatoryLabel, 'PackageRelativeAppId': logentry_PackageRelativeAppId,'SubProcessTag': logentry_SubProcessTag,
                                        
                                        'PROC-NODE': proc_uid, 
                                        'THREAD-NODE': thread_uid,
                                        }


        #-----------------------------------------------------------------------------------------------------------------------------------------
        # THREADSTART Type-2: Parent-Process and its Thread THREADSTART-ing its Child-Process's New Thread
        # e.g. Process 6600 and its Thread 3560 THREADSTART-ing Process 6284's (Child-Process of Process 6600) New Thread 12704.
        #                      [ProcessId]    [ThreadId]    [TaskName]     [ProcessID]   [ThreadID]
        #          log-entry:     6600           3560       THREADSTART        6284         12704
        #
        #      > Then this log-entry should generate the following:
        #       (Process 6284)----Edge----(Thread 12704)

        if logentry_ProcessId != logentry_ProcessID:    # THREADSTART Type-2: Parent-Process and its Thread THREADSTART-ing its Child-Process's New Thread

            set_ProcessThread_StartTime_dict( PID = logentry_ProcessID, TID = logentry_ThreadID, thread_starttime= logentry_TimeStamp)   # THREADSTART doesn't have CreateTime, logentry_TimeStamp can still serve as an identifier.
            #ProcessThread_StartTime_dict = get_ProcessThread_StartTime_dict()            

            # (Process-node)
            #proc_hash = None
            if logentry_ProcessID in Process_StartTime_dict:    # "If Child-Process was process-started after log-collection." 
                #proc_hash = str( logentry_ProcessID ) + str( Process_StartTime_dict[logentry_ProcessID] )
                proc_hash = f"<<PROC-NODE>>(PID):{logentry_ProcessID}_(CT):{Process_StartTime_dict[logentry_ProcessID]}"
            else:   # "If Child-Process was also process-started before log-collection. (This case might be unlikely, but is not impossible)"
                #proc_hash = str( logentry_ProcessID )
                proc_hash = f"<<PROC-NODE>>(PID):{logentry_ProcessID}_(CT):N/A"

            proc_uid = get_uid(proc_hash, host, firststep_hash_debugging_mode)
            proc_uid_list.append(proc_uid)
            proc_node_dict[proc_uid] = {'ProcessId': logentry_ProcessID } # JY @ 2023-1-10: If 'ProcessId' is what indicates this process-node's uid, 
                                                                          #                 then here its value should be 'ProcessID' not 'ProcessId'   (Discuss this)

            # (Thread-node)
            thread_hash = None
            thread_uid = None
            if logentry_ProcessID in Process_StartTime_dict: # "If Child-Process was process-started after log-collection."
                # "Since just before this new Thread is set with its creation-time in 'ProcessThread_StartTime_dict', no need to check whether ThreadId in 'ProcessThread_StartTime_dict' "
                #thread_hash = ( str(logentry_ThreadID) + str(ProcessThread_StartTime_dict[logentry_ProcessID][logentry_ThreadID]) ) + ( str(logentry_ProcessID)+str( Process_StartTime_dict[logentry_ProcessID] ) )
                thread_hash = f"<<THREAD>>(TID):{logentry_ThreadID}_(TS):{ProcessThread_StartTime_dict[logentry_ProcessID][logentry_ThreadID]}__(PID):{logentry_ProcessID}_(CT):{Process_StartTime_dict[logentry_ProcessID]}"
            else: # "If Child-Process was also process-started before log-collection. (This case might be unlikely, but is not impossible)"
                  # "Since just before this new Thread is set with its creation-time in 'ProcessThread_StartTime_dict', no need to check whether ThreadId in 'ProcessThread_StartTime_dict' "
                  #     > In "set_ProcessThread_StartTime_dict()", 
                  #       Already the case of "Process started before log-collection, but Thread started after log-collection, 
                  #                            Thus 'ProcessThread_StartTime_dict' initially not having 'logentry_ProcessId' key"    is handled.
                #thread_hash = ( str(logentry_ThreadID) + str(ProcessThread_StartTime_dict[logentry_ProcessID][logentry_ThreadID]) )  + str(logentry_ProcessID)
                thread_hash = f"<<THREAD>>(TID):{logentry_ThreadID}_(TS):{ProcessThread_StartTime_dict[logentry_ProcessID][logentry_ThreadID]}__(PID):{logentry_ProcessID}_(CT):N/A"
            thread_uid = get_uid(thread_hash,host, firststep_hash_debugging_mode)
            proc_thread_uid_list.append(thread_uid)            
            proc_thread_dict[thread_uid]= {'ThreadID': logentry_ThreadID, 'ProcessID': logentry_ProcessID, 
                                           'ThreadId': logentry_ThreadId, 'ProcessId': logentry_ProcessId}

            # (Edge)
            #hashout = str(logentry_ProcessID) + str(logentry_ThreadID) + str(logentry_TimeStamp)   # this should be fine
            hashout = f"<<PROC-EDGE>>(PID):{logentry_ProcessId}_(TID):{logentry_ThreadId}_(TN):{logentry_TaskName}_(OP):{logentry_Opcode}_(TS):{logentry_TimeStamp}"
            edge_uid = get_uid(hashout,host, firststep_hash_debugging_mode)
            proc_edge_uid_list.append(edge_uid)
            proc_edge_dict[edge_uid] = {'ProcessId': logentry_ProcessId, 'ProcessID': logentry_ProcessID, 
                                        'ThreadId': logentry_ThreadId, 'ThreadID': logentry_ThreadID,
                                        'Task Name':logentry_TaskName, 'Opcode':logentry_Opcode, 'ImageName':logentry_ImageName, 
                                        'TimeStamp': logentry_TimeStamp, 'CreateTime': logentry_CreateTime,
                                        
                                        # 'ImageSize':logentry_ImageSize,'HandleCount':logentry_HandleCount,
                                        # 'TimeDateStamp': logentry_TimeDateStamp, 'StatusCode': logentry_StatusCode,
                                        # 'ReadOperationCount':logentry_ReadOperationCount,'WriteOperationCount':logentry_WriteOperationCount, 
                                        # 'ReadTransferKiloBytes': logentry_ReadTransferKiloBytes, 'WriteTransferKiloBytes':logentry_WriteTransferKiloBytes,
                                        # 'ProcessTokenElevationType': logentry_ProcessTokenElevationType, 'ProcessTokenIsElevated': logentry_ProcessTokenIsElevated, 
                                        # 'MandatoryLabel': logentry_MandatoryLabel, 'PackageRelativeAppId': logentry_PackageRelativeAppId,'SubProcessTag': logentry_SubProcessTag,
                                        
                                        'PROC-NODE': proc_uid, 
                                        'THREAD-NODE': thread_uid,
                                        }


    #============================================================================================================================================================================================================
    # Added by JY @ 2023-03-13:
    # -> For "CG Correctness Enhancement" 
    #    Case-Handlings for ["CPUPRIORITYCHANGE", "CPUBASEPRIORITYCHANGE", "IOPRIORITYCHANGE", "PAGEPRIORITYCHANGE"] and ["IMAGELOAD", "IMAGEUNLOAD"]
    #  
    #    All of these are based on: /data/d1/jgwak1/tabby/EVENTTYPE_EDGECASES_IDENTIFICATION_FOR_CG_CORRECTNESS/
    
    
    elif logentry_TaskName.upper() in ["CPUPRIORITYCHANGE", "CPUBASEPRIORITYCHANGE", "IOPRIORITYCHANGE", "PAGEPRIORITYCHANGE"]:
        # 3 cases: Non-Edge-Case, Case-1, Case-2
        # Refer to : https://docs.google.com/presentation/d/1prPhAl_8P6VQ7NYXt2l1j3f7wsd3b84a9PNC2UngAXk/edit#slide=id.g21912b4d815_2_9 (slide 439)

        # (Non-Edge-Case)
        #
        # > Log-entry Example
        #   
        #   ProcessId | ThreadId | ProcessID | ThreadID |  Task-Name  
        #   ----------------------------------------------------------
        #        4    |  68      |    4      |   68     |  CPUBASEPRIORITYCHNAGE 
        # 
        #   --->      ( Thread 68 (of Process 4) ) ---CPUBASEPRIORITYCHNAGE---> ( Process 4 )   
        #   i.e. Process 4 using its Thread 68, is invoking Process 4's Thread 68 to change its Process's (Process 4) CPUBASE/CPU/IO/PAGE-PRIORITY. 
        if logentry_ProcessId == logentry_ProcessID and logentry_ThreadId == logentry_ThreadID:
            
            # for proc-node -----------
            if logentry_ProcessId in Process_StartTime_dict:    
                proc_hash = f"<<PROC-NODE>>(PID):{logentry_ProcessId}_(CT):{Process_StartTime_dict[logentry_ProcessId]}"
            else: # "If process was process-started before log-collection.
                proc_hash = f"<<PROC-NODE>>(PID):{logentry_ProcessId}_(CT):N/A"
            proc_uid = get_uid(proc_hash, host, firststep_hash_debugging_mode)
            proc_uid_list.append(proc_uid)
            proc_node_dict[proc_uid] = {'ProcessId': logentry_ProcessId }             
            
            # for-threadnode -----------
            thread_hash = None
            thread_uid = None
            if logentry_ProcessId in Process_StartTime_dict: # "If process was process-started after log-collection." 
                if logentry_ThreadId in ProcessThread_StartTime_dict[logentry_ProcessId]: # "And If Thread of process was started after log-collection"   
                    thread_hash = f"<<THREAD>>(TID):{logentry_ThreadId}_(TS):{ProcessThread_StartTime_dict[logentry_ProcessId][logentry_ThreadId]}__(PID):{logentry_ProcessId}_(CT):{Process_StartTime_dict[logentry_ProcessId]}"
                else:
                    thread_hash = "<<THREAD>>PROC-DIDITREALLYCOMEHERE-ALLOTHERTASKS"
            else: # "If process was process-started before log-collection"
                if logentry_ThreadId in ProcessThread_StartTime_dict.get(logentry_ProcessId, {}): # "And If Thread of process was started after log-collection"   
                    thread_hash = f"<<THREAD>>(TID):{logentry_ThreadId}_(TS):{ProcessThread_StartTime_dict[logentry_ProcessId][logentry_ThreadId]}__(PID):{logentry_ProcessId}_(CT):N/A"
                else: # "And If Thread of process started before log-collection"        
                    thread_hash = f"<<THREAD>>(TID):{logentry_ThreadId}_(TS):N/A__(PID):{logentry_ProcessId}_(CT):N/A"
            thread_uid = get_uid(thread_hash,host, firststep_hash_debugging_mode)
            proc_thread_uid_list.append(thread_uid)
            proc_thread_dict[thread_uid]= {'ThreadID': logentry_ThreadID, 'ProcessID': logentry_ProcessID, 
                                           'ThreadId': logentry_ThreadId, 'ProcessId': logentry_ProcessId}

            # for edge -----------
            hashout = f"<<PROC-EDGE>>(PID):{logentry_ProcessId}_(TID):{logentry_ThreadId}_(TN):{logentry_TaskName}_(OP):{logentry_Opcode}_(TS):{logentry_TimeStamp}"
            edge_uid = get_uid(hashout,host, firststep_hash_debugging_mode)
            proc_edge_uid_list.append(edge_uid)
            proc_edge_dict[edge_uid] = {'ProcessId': logentry_ProcessId, 'ProcessID': logentry_ProcessID, 
                                        'ThreadId': logentry_ThreadId, 'ThreadID': logentry_ThreadID,
                                        'Task Name':logentry_TaskName, 'Opcode':logentry_Opcode, 'ImageName':logentry_ImageName, 
                                        'TimeStamp': logentry_TimeStamp, 'CreateTime': logentry_CreateTime,
                                        
                                        # 'ImageSize':logentry_ImageSize,'HandleCount':logentry_HandleCount,
                                        # 'TimeDateStamp': logentry_TimeDateStamp, 'StatusCode': logentry_StatusCode,
                                        # 'ReadOperationCount':logentry_ReadOperationCount,'WriteOperationCount':logentry_WriteOperationCount, 
                                        # 'ReadTransferKiloBytes': logentry_ReadTransferKiloBytes, 'WriteTransferKiloBytes':logentry_WriteTransferKiloBytes,
                                        # 'ProcessTokenElevationType': logentry_ProcessTokenElevationType, 'ProcessTokenIsElevated': logentry_ProcessTokenIsElevated, 
                                        # 'MandatoryLabel': logentry_MandatoryLabel, 'PackageRelativeAppId': logentry_PackageRelativeAppId,'SubProcessTag': logentry_SubProcessTag,
                                        
                                        'PROC-NODE': proc_uid,
                                        'THREAD-NODE': thread_uid
                                        }        


        # (Case-1)
        #   ProcessId | ThreadId | ProcessID | ThreadID |  Task-Name  
        #   ----------------------------------------------------------
        #     12628   |  2052    |   12628   |   6608   |  CPUBASEPRIORITYCHNAGE 
        # 
        #   --->      ( Thread 6608 (of Process 12628) ) ---CPUBASEPRIORITYCHNAGE---> ( Process 12628 ) 
        #
        #   i.e. Process 12628 using its Thread 2052, is invoking Process 12628's Thread 6608 to change its Process's (Process 12628) CPUBASE/CPU/IO/PAGE Priority. 
        # 
        # (Case-2)
        #   ProcessId | ThreadId | ProcessID | ThreadID |  Task-Name  
        #   ----------------------------------------------------------
        #     4136    |  2552     |   1476    |   3032   |  CPUBASEPRIORITYCHNAGE 
        # 
        #   --->      ( Thread 3032 (of Process 1476) ) ---CPUBASEPRIORITYCHNAGE---> ( Process 1476 ) 
        #   i.e. Process 4136 using its Thread 2552, is invoking Process 12628's Thread 3032 to change its Process's (Process 12628) CPUBASE/CPU/IO/PAGE Priority. 
        #
        # * Both 'Case-1' and 'Case-2' are handled by identical handling.
        #   > As 'Case-1' and 'Case-2' can both use "ProcessID" and "ThreadID" in determining hash for Process-Node and Thread-Node,
        
        # If 'Case-1' or 'Case-2'
        if (logentry_ProcessId == logentry_ProcessID and logentry_ThreadId != logentry_ThreadID) or\
           (logentry_ProcessId != logentry_ProcessID and logentry_ThreadId != logentry_ThreadID):
            
            # for proc-node -----------
            if logentry_ProcessID in Process_StartTime_dict:    
                proc_hash = f"<<PROC-NODE>>(PID):{logentry_ProcessID}_(CT):{Process_StartTime_dict[logentry_ProcessID]}"
            else: # "If process was process-started before log-collection.
                proc_hash = f"<<PROC-NODE>>(PID):{logentry_ProcessID}_(CT):N/A"
            proc_uid = get_uid(proc_hash, host, firststep_hash_debugging_mode)
            proc_uid_list.append(proc_uid)
            proc_node_dict[proc_uid] = {'ProcessId': logentry_ProcessID }             
            
            # for-threadnode -----------
            thread_hash = None
            thread_uid = None
            if logentry_ProcessID in Process_StartTime_dict: # "If process was process-started after log-collection." 
                if logentry_ThreadID in ProcessThread_StartTime_dict[logentry_ProcessID]: # "And If Thread of process was started after log-collection"   
                    thread_hash = f"<<THREAD>>(TID):{logentry_ThreadID}_(TS):{ProcessThread_StartTime_dict[logentry_ProcessID][logentry_ThreadID]}__(PID):{logentry_ProcessID}_(CT):{Process_StartTime_dict[logentry_ProcessID]}"
                else:
                    thread_hash = "<<THREAD>>PROC-DIDITREALLYCOMEHERE-ALLOTHERTASKS"
            else: # "If process was process-started before log-collection"
                if logentry_ThreadID in ProcessThread_StartTime_dict.get(logentry_ProcessID, {}): # "And If Thread of process was started after log-collection"   
                    thread_hash = f"<<THREAD>>(TID):{logentry_ThreadID}_(TS):{ProcessThread_StartTime_dict[logentry_ProcessID][logentry_ThreadID]}__(PID):{logentry_ProcessID}_(CT):N/A"
                else: # "And If Thread of process started before log-collection"        
                    thread_hash = f"<<THREAD>>(TID):{logentry_ThreadID}_(TS):N/A__(PID):{logentry_ProcessID}_(CT):N/A"
            thread_uid = get_uid(thread_hash,host, firststep_hash_debugging_mode)
            proc_thread_uid_list.append(thread_uid)
            proc_thread_dict[thread_uid]= {'ThreadID': logentry_ThreadID, 'ProcessID': logentry_ProcessID, 
                                           'ThreadId': logentry_ThreadId, 'ProcessId': logentry_ProcessId}

            # for edge -----------
            hashout = f"<<PROC-EDGE>>(PID):{logentry_ProcessID}_(TID):{logentry_ThreadID}_(TN):{logentry_TaskName}_(OP):{logentry_Opcode}_(TS):{logentry_TimeStamp}"
            edge_uid = get_uid(hashout,host, firststep_hash_debugging_mode)
            proc_edge_uid_list.append(edge_uid)
            proc_edge_dict[edge_uid] = {'ProcessId': logentry_ProcessId, 'ProcessID': logentry_ProcessID, 
                                        'ThreadId': logentry_ThreadId, 'ThreadID': logentry_ThreadID,
                                        'Task Name':logentry_TaskName, 'Opcode':logentry_Opcode, 'ImageName':logentry_ImageName, 
                                        'TimeStamp': logentry_TimeStamp, 'CreateTime': logentry_CreateTime,
                                        
                                        # 'ImageSize':logentry_ImageSize,'HandleCount':logentry_HandleCount,
                                        # 'TimeDateStamp': logentry_TimeDateStamp, 'StatusCode': logentry_StatusCode,
                                        # 'ReadOperationCount':logentry_ReadOperationCount,'WriteOperationCount':logentry_WriteOperationCount, 
                                        # 'ReadTransferKiloBytes': logentry_ReadTransferKiloBytes, 'WriteTransferKiloBytes':logentry_WriteTransferKiloBytes,
                                        # 'ProcessTokenElevationType': logentry_ProcessTokenElevationType, 'ProcessTokenIsElevated': logentry_ProcessTokenIsElevated, 
                                        # 'MandatoryLabel': logentry_MandatoryLabel, 'PackageRelativeAppId': logentry_PackageRelativeAppId,'SubProcessTag': logentry_SubProcessTag,
                                        
                                        'PROC-NODE': proc_uid,
                                        'THREAD-NODE': thread_uid
                                        }        



    elif logentry_TaskName in ["IMAGELOAD", "IMAGEUNLOAD"]: 
        # 2 cases: Case-3 , Case -4 
        # Refer to : https://docs.google.com/presentation/d/1prPhAl_8P6VQ7NYXt2l1j3f7wsd3b84a9PNC2UngAXk/edit#slide=id.g21912b4d815_2_9 (slide 439)

        # (Case-3)
        #   ProcessId | ThreadId | ProcessID | ThreadID |  Task-Name  
        #   ----------------------------------------------------------
        #     9156    |  7672    |   9156    |   -      |  IMAGEUNLOAD/IMAGELOAD 
        # 
        #   --->      ( Thread 7672 (of Process 9156) ) ---IMAGELOAD/IMAGEUNLOAD---> ( Process 9156 ) 
        #   i.e. Process 9156 using its Thread 7672, is IMAGELOAD/IMAGEUNLOAD-ing for Process 9156
        #
        # (Case-4)
        #   ProcessId | ThreadId | ProcessID | ThreadID |  Task-Name  
        #   ----------------------------------------------------------
        #     3880    |  7060    |   8040    |   -      |  IMAGEUNLOAD/IMAGELOAD 
        # 
        #   --->      ( Thread 7060 (of Process 3880) ) ---IMAGEUNLOAD/IMAGELOAD ---> ( Process 8040 ) 
        #   i.e. Process 3880 using its Thread 7060, is IMAGELOAD/IMAGEUNLOAD-ing for Process 8040        
        #
        # * Both 'Case-3' and 'Case-4' are handled by identical handling.
        #   > As 'Case-3' and 'Case-4' both uses ("ProcessID") for Process-Node
        #     and ("ThreadId" of "ProcessId") for Thread-Node.

        # If 'Case-3' or 'Case-4'        
        if logentry_ProcessId == logentry_ProcessID or\
           logentry_ProcessId != logentry_ProcessID:

            # for proc-node -----------
            if logentry_ProcessID in Process_StartTime_dict:    
                proc_hash = f"<<PROC-NODE>>(PID):{logentry_ProcessID}_(CT):{Process_StartTime_dict[logentry_ProcessID]}"
            else: # "If process was process-started before log-collection.
                proc_hash = f"<<PROC-NODE>>(PID):{logentry_ProcessID}_(CT):N/A"
            proc_uid = get_uid(proc_hash, host, firststep_hash_debugging_mode)
            proc_uid_list.append(proc_uid)
            proc_node_dict[proc_uid] = {'ProcessId': logentry_ProcessID }             
            
            # for-threadnode -----------
            thread_hash = None
            thread_uid = None
            if logentry_ProcessId in Process_StartTime_dict: # "If process was process-started after log-collection." 
                if logentry_ThreadId in ProcessThread_StartTime_dict[logentry_ProcessId]: # "And If Thread of process was started after log-collection"   
                    thread_hash = f"<<THREAD>>(TID):{logentry_ThreadId}_(TS):{ProcessThread_StartTime_dict[logentry_ProcessId][logentry_ThreadId]}__(PID):{logentry_ProcessId}_(CT):{Process_StartTime_dict[logentry_ProcessId]}"
                else:
                    thread_hash = "<<THREAD>>PROC-DIDITREALLYCOMEHERE-ALLOTHERTASKS"
            else: # "If process was process-started before log-collection"
                if logentry_ThreadId in ProcessThread_StartTime_dict.get(logentry_ProcessId, {}): # "And If Thread of process was started after log-collection"   
                    thread_hash = f"<<THREAD>>(TID):{logentry_ThreadId}_(TS):{ProcessThread_StartTime_dict[logentry_ProcessId][logentry_ThreadId]}__(PID):{logentry_ProcessId}_(CT):N/A"
                else: # "And If Thread of process started before log-collection"        
                    thread_hash = f"<<THREAD>>(TID):{logentry_ThreadId}_(TS):N/A__(PID):{logentry_ProcessId}_(CT):N/A"
            thread_uid = get_uid(thread_hash,host, firststep_hash_debugging_mode)
            proc_thread_uid_list.append(thread_uid)
            proc_thread_dict[thread_uid]= {'ThreadID': logentry_ThreadID, 'ProcessID': logentry_ProcessID, 
                                           'ThreadId': logentry_ThreadId, 'ProcessId': logentry_ProcessId}

            # for edge -----------
            hashout = f"<<PROC-EDGE>>(PID):{logentry_ProcessID}_(TID):{logentry_ThreadId}_(TN):{logentry_TaskName}_(OP):{logentry_Opcode}_(TS):{logentry_TimeStamp}"
            edge_uid = get_uid(hashout,host, firststep_hash_debugging_mode)
            proc_edge_uid_list.append(edge_uid)
            proc_edge_dict[edge_uid] = {'ProcessId': logentry_ProcessId, 'ProcessID': logentry_ProcessID, 
                                        'ThreadId': logentry_ThreadId, 'ThreadID': logentry_ThreadID,
                                        'Task Name':logentry_TaskName, 'Opcode':logentry_Opcode, 'ImageName':logentry_ImageName, 
                                        'TimeStamp': logentry_TimeStamp, 'CreateTime': logentry_CreateTime,
                                        
                                        # 'ImageSize':logentry_ImageSize,'HandleCount':logentry_HandleCount,
                                        # 'TimeDateStamp': logentry_TimeDateStamp, 'StatusCode': logentry_StatusCode,
                                        # 'ReadOperationCount':logentry_ReadOperationCount,'WriteOperationCount':logentry_WriteOperationCount, 
                                        # 'ReadTransferKiloBytes': logentry_ReadTransferKiloBytes, 'WriteTransferKiloBytes':logentry_WriteTransferKiloBytes,
                                        # 'ProcessTokenElevationType': logentry_ProcessTokenElevationType, 'ProcessTokenIsElevated': logentry_ProcessTokenIsElevated, 
                                        # 'MandatoryLabel': logentry_MandatoryLabel, 'PackageRelativeAppId': logentry_PackageRelativeAppId,'SubProcessTag': logentry_SubProcessTag,
                                        
                                        'PROC-NODE': proc_uid,
                                        'THREAD-NODE': thread_uid
                                        }



        


    #============================================================================================================================================================================================================

    # JY: Might not have to explicitly handle for "PROCESSSTOP" given following example of Process stopping itself with its Thread.
    #     Also, handling of reusing PIDs is handled by overwriting values in Process_StartTime_dict (and for a Process-id to be re-used, there must be a correpsonidng PROCESSSART inbetween stop and re-use), 
    #     thus no need to explicitly manipulate values in Process_StartTime_dict when PROCESSSTOP.

    #elif logentry_TaskName == "PROCESSSTOP":
        # < Example of PROCESSSTOP event from "chrome_browser_all_providers.txt"> :
        #
        # {'EventHeader': {'Size': 175, 'HeaderType': 0, 'Flags': 576, 'EventProperty': 0, 'ThreadId': 8396, 'ProcessId': 6356, 'TimeStamp': 133173733608716024, 'ProviderId': '{22FB2CD6-0E7B-422B-A0C7-2FAD1FD0E716}', 
        #  'EventDescriptor': {'Id': 2, 'Version': 2, 'Channel': 16, 'Level': 4, 'Opcode': 2, 'Task': 2, 'Keyword': 9223372036854775824}, 
        #  'KernelTime': 0, 'UserTime': 0, 'ActivityId': '{00000000-0000-0000-0000-000000000000}'}, 'Task Name': 'PROCESSSTOP', 'ProcessID': '6356', 
        #  'ProcessSequenceNumber': '802', 'CreateTime': '\u200e2023\u200e-\u200e01\u200e-\u200e05T06:22:38.876035500Z', 'ExitTime': '\u200e2023\u200e-\u200e01\u200e-\u200e05T06:22:40.870870100Z', 'ExitCode': '0', 
        # 'TokenElevationType': '2', 'HandleCount': '221', 'CommitCharge': '14356480', 'CommitPeak': '14372864', 'CPUCycleCount': '232512840', 'ReadOperationCount': '83', 'WriteOperationCount': '128', 
        # 'ReadTransferKiloBytes': '60', 'WriteTransferKiloBytes': '19', 'HardFaultCount': '0', 'ImageName': 'chrome.exe', 
        # 'Description': 'Process %1 (which started at time %3) stopped at time %4 with exit code %5. '}) 
        #
        #     > This log-entry should generate the following:
        #       (Process 6356)----Edge: PROCESSSTOP ----(Thread 8396)


    #============================================================================================================================================================================================================
    
    # JY: Might not have to explicitly handle for "THREADSTOP" given following example of Process+Thread stopping itself (Process+Thread).
    #     Also, handling of reusing TIDs is handled by overwriting values in ProcessThread_StartTime_dict (and for a ThreadId to be re-used, there must be a correpsonidng THREADSTART inbetween stop and re-use), 
    #     thus no need to explicitly manipulate values in ProcessThread_StartTime_dict when THREADSTOP.

    
    #elif logentry_TaskName == "THREADSTOP":

        # < Example of THREADSTOP event from "chrome_browser_all_providers.txt"> :
        # * Unlike THREADSTART, THREADSTOP appears to have only 1 type.

        #{'EventHeader': {'Size': 156, 'HeaderType': 0, 'Flags': 576, 'EventProperty': 0, 'ThreadId': 4300, 'ProcessId': 9852, 'TimeStamp': 133173733636421112, 'ProviderId': '{22FB2CD6-0E7B-422B-A0C7-2FAD1FD0E716}', 
        # 'EventDescriptor': {'Id': 4, 'Version': 1, 'Channel': 16, 'Level': 4, 'Opcode': 2, 'Task': 4, 'Keyword': 9223372036854775840}, 
        # 'KernelTime': 0, 'UserTime': 0, 'ActivityId': '{00000000-0000-0000-0000-000000000000}'}, 'Task Name': 'THREADSTOP', 'ProcessID': '9852', 'ThreadID': '4300', 
        # 'StackBase': '0xFFFFA186DFA30000', 'StackLimit': '0xFFFFA186DFA29000', 'UserStackBase': '0xA590200000', 'UserStackLimit': '0xA5901FC000', 'StartAddr': '0x7FF8FFBD8AE0', 
        # 'Win32StartAddr': '0x7FF8FFBD8AE0', 'TebBase': '0xA58C0E5000', 'SubProcessTag': '0', 'CycleTime': '0x30F6B2', 'Description': 'Thread %2 (in Process %1) stopped. '}) 
        # 
        #     > This log-entry should generate the following:
        #       (Process 9852)----Edge: THREADSTOP----(Thread 8396)

    #============================================================================================================================================================================================================    
    
    else:   # FOR ALL OTHER TASKNAMES FROM THE PROCESS-PROVIDER 
            #  e.g. PROCESSSTOP, THREADSTOP, JOBSTART, JOBSTOP, IMAGELOAD, IMAGEUNLOAD, CPUBASEPRIORITYCHANGE, CPUPRIORITYCHANGE, IOPRIORITYCHANGE, PAGEPRIORITYCHANGE, THREADWORKONBEHALFUPDATE..
        
        #Process_StartTime_dict = get_ProcessStartTime_dict()
        #ProcessThread_StartTime_dict = get_ProcessThread_StartTime_dict()
        
        # (Process-node)
        #proc_hash = None
        if logentry_ProcessId in Process_StartTime_dict:    # "If process was process-started after log-collection." 
                                                            #
                                                            #    NOTE: Whether the process reused a stopped process's PID should not complicate things.
                                                            #          Reason is in Process_StartTime_dict, the old starttime value of the stopped process (which used the same PID)  
                                                            #          is overwritten by this process's starttime during PROCESSSTART of this process that reuses the same PID.
                                                            #          i.e., 'Process_StartTime_dict' will be up-to-date.
            #proc_hash = str( logentry_ProcessId ) + str( Process_StartTime_dict[logentry_ProcessId] )
            proc_hash = f"<<PROC-NODE>>(PID):{logentry_ProcessId}_(CT):{Process_StartTime_dict[logentry_ProcessId]}"
        else: # "If process was process-started before log-collection.
            #proc_hash = str( logentry_ProcessId )
            proc_hash = f"<<PROC-NODE>>(PID):{logentry_ProcessId}_(CT):N/A"
        proc_uid = get_uid(proc_hash, host, firststep_hash_debugging_mode)
        proc_uid_list.append(proc_uid)
        proc_node_dict[proc_uid] = {'ProcessId': logentry_ProcessId } 



        # (Thread-node)

        # JUST GETTING THE RIGHT PROCESS AND RIGHT PROCESS-THREAD
        thread_hash = None
        thread_uid = None
        if logentry_ProcessId in Process_StartTime_dict: # "If process was process-started after log-collection." 
            
            if logentry_ThreadId in ProcessThread_StartTime_dict[logentry_ProcessId]: # "And If Thread of process was started after log-collection"   
                                                                                      #
                                                                                      #          Reason is in ProcessThread_StartTime_dict,
                                                                                      #          the old startime value of the stopped thread (which used same ThreadId) within this process
                                                                                      #          is overwritten by the starttime of newly started thread (which reuses the same ThreadId within this process).
                                                                                      #          during its THREADSTART.
                                                                                      #          i.e., 'ProcessThread_StartTime_dict' will be up-to-date (also for 'Process_StartTime_dict')
                #thread_hash = ( str(logentry_ThreadId) + str(ProcessThread_StartTime_dict[logentry_ProcessId][logentry_ThreadId]) )  +  ( str(logentry_ProcessId)+str( Process_StartTime_dict[logentry_ProcessId] ) )
                thread_hash = f"<<THREAD>>(TID):{logentry_ThreadId}_(TS):{ProcessThread_StartTime_dict[logentry_ProcessId][logentry_ThreadId]}__(PID):{logentry_ProcessId}_(CT):{Process_StartTime_dict[logentry_ProcessId]}"
            # else: "logentry_ThreadId NOT in ProcessThread_StartTime_dict[logentry_ProcessId]" indicates this process(ProcessId) does not own this thread(ThreadId),
            #       which is contradictiory since this process started after the log-collection-start and for this log-entry to exist (with this ProcessId and ThreadId), 
            #       this process must have started this thread. 
            else:
                thread_hash = "<<THREAD>>PROC-DIDITREALLYCOMEHERE-ALLOTHERTASKS"

                # "THREADSTOP" and "THREADBEHALFONUPDATE" events reached here.
                # TODO: DEAL WITH THIS CASES AFTER OBSERVING FOR MULTIPLE DIFFERENT SAMPLES

        else: # "If process was process-started before log-collection"

            if logentry_ThreadId in ProcessThread_StartTime_dict.get(logentry_ProcessId, {}): # "And If Thread of process was started after log-collection"   
                                                                                      #
                                                                                      #    NOTE: Whether this ThreadId is reused (stopped and started) within this process should not affect correctness.
                                                                                      #          Reason is in ProcessThread_StartTime_dict,
                                                                                      #          the old startime value of the stopped thread (which used same ThreadId) within this process
                                                                                      #          is overwritten by the starttime of newly started thread (which reuses the same ThreadId within this process).
                                                                                      #          during its THREADSTART.
                                                                                      #          i.e., 'ProcessThread_StartTime_dict' will be up-to-date (also for 'Process_StartTime_dict')
                #thread_hash = ( str(logentry_ThreadId) + str(ProcessThread_StartTime_dict[logentry_ProcessId][logentry_ThreadId]) )  +  str(logentry_ProcessId)
                thread_hash = f"<<THREAD>>(TID):{logentry_ThreadId}_(TS):{ProcessThread_StartTime_dict[logentry_ProcessId][logentry_ThreadId]}__(PID):{logentry_ProcessId}_(CT):N/A"
            else: # "And If Thread of process started before log-collection"        
                #thread_hash = str(logentry_ThreadId) + str(logentry_ProcessId)
                thread_hash = f"<<THREAD>>(TID):{logentry_ThreadId}_(TS):N/A__(PID):{logentry_ProcessId}_(CT):N/A"
        thread_uid = get_uid(thread_hash,host, firststep_hash_debugging_mode)
        proc_thread_uid_list.append(thread_uid)
        proc_thread_dict[thread_uid]= {'ThreadID': logentry_ThreadID, 'ProcessID': logentry_ProcessID, 
                                       'ThreadId': logentry_ThreadId, 'ProcessId': logentry_ProcessId}

        # (Edge)
        #hashout = str(logentry_ProcessId) + str(logentry_ThreadId) + str(logentry_TimeStamp) 
        hashout = f"<<PROC-EDGE>>(PID):{logentry_ProcessId}_(TID):{logentry_ThreadId}_(TN):{logentry_TaskName}_(OP):{logentry_Opcode}_(TS):{logentry_TimeStamp}"
        edge_uid = get_uid(hashout,host, firststep_hash_debugging_mode)
        proc_edge_uid_list.append(edge_uid)
        proc_edge_dict[edge_uid] = {'ProcessId': logentry_ProcessId, 'ProcessID': logentry_ProcessID, 
                                    'ThreadId': logentry_ThreadId, 'ThreadID': logentry_ThreadID,
                                    'Task Name':logentry_TaskName, 'Opcode':logentry_Opcode, 'ImageName':logentry_ImageName, 
                                    'TimeStamp': logentry_TimeStamp, 'CreateTime': logentry_CreateTime,
                                    
                                    # 'ImageSize':logentry_ImageSize,'HandleCount':logentry_HandleCount,
                                    # 'TimeDateStamp': logentry_TimeDateStamp, 'StatusCode': logentry_StatusCode,
                                    # 'ReadOperationCount':logentry_ReadOperationCount,'WriteOperationCount':logentry_WriteOperationCount, 
                                    # 'ReadTransferKiloBytes': logentry_ReadTransferKiloBytes, 'WriteTransferKiloBytes':logentry_WriteTransferKiloBytes,
                                    # 'ProcessTokenElevationType': logentry_ProcessTokenElevationType, 'ProcessTokenIsElevated': logentry_ProcessTokenIsElevated, 
                                    # 'MandatoryLabel': logentry_MandatoryLabel, 'PackageRelativeAppId': logentry_PackageRelativeAppId,'SubProcessTag': logentry_SubProcessTag,
                                    
                                    'PROC-NODE': proc_uid,
                                    'THREAD-NODE': thread_uid
                                    }

#####################################################################################################################################################################
## Node Network -------------------------------------------------------------     
def get_net_info(log_entry, host, 
                 net_uid_list, net_thread_uid_list, net_edge_uid_list, 
                 net_node_dict, net_thread_dict, net_edge_dict,
                 
                 firststep_hash_debugging_mode = False
                 ):   

    logentry_destaddr = log_entry.get('_source', {}).get('daddr') # feature
    logentry_TaskName = log_entry.get('_source', {}).get('Task Name') # feature
    logentry_Opcode = log_entry.get('_source', {}).get('Opcode') # feature
    logentry_ThreadId = str(log_entry.get('_source', {}).get('ThreadId'))
    logentry_ThreadID = str(log_entry.get('_source', {}).get('ThreadID'))
    logentry_ProcessId = str(log_entry.get('_source', {}).get('ProcessId'))
    logentry_ProcessID = str(log_entry.get('_source', {}).get('ProcessID'))
    logentry_TimeStamp = log_entry.get('_source', {}).get('TimeStamp')

    # logentry_size = log_entry.get('_source', {}).get('size')             # feature
    # logentry_SubProcessTag = log_entry.get('_source', {}).get('SubProcessTag')         #feature

        
    # This Network log-entry should generate the following:
    # (Network node)---Edge---(Thread node)

    # (Network-node)
    #net_hash = str(logentry_destaddr)
    net_hash = f"<<NET-NODE>>(daddr):{logentry_destaddr}"
    net_uid = get_uid(net_hash, host, firststep_hash_debugging_mode)
    net_uid_list.append(net_uid)
    net_node_dict[net_uid] = {'daddr': logentry_destaddr}

    # (Thread-node)

    #Process_StartTime_dict = get_ProcessStartTime_dict() # To get the process's starttime (if exists) for the process component of the hash-input in computing the Thread-UID
    #ProcessThread_StartTime_dict = get_ProcessThread_StartTime_dict() # To get the thread's starttime (if exists) for the thread component of the hash-input in computing the Thread-UID
    
    # JUST GETTING THE RIGHT PROCESS AND RIGHT PROCESS-THREAD
    thread_hash = None
    thread_uid = None
    if logentry_ProcessId in Process_StartTime_dict: # "If process was process-started after log-collection." 
        if logentry_ThreadId in ProcessThread_StartTime_dict[logentry_ProcessId]:   # "And If Thread of process was started after log-collection"   
                                                                                    #
                                                                                    #    NOTE: Whether this ThreadId is reused (stopped and started) within this process should not affect correctness.
                                                                                    #          Reason is in ProcessThread_StartTime_dict,
                                                                                    #          the old startime value of the stopped thread (which used same ThreadId) within this process
                                                                                    #          is overwritten by the starttime of newly started thread (which reuses the same ThreadId within this process).
                                                                                    #          during its THREADSTART.
                                                                                    #          i.e., 'ProcessThread_StartTime_dict' will be up-to-date (also for 'Process_StartTime_dict')
            #thread_hash = ( str(logentry_ThreadId) + str(ProcessThread_StartTime_dict[logentry_ProcessId][logentry_ThreadId]) )  +  ( str(logentry_ProcessId)+str( Process_StartTime_dict[logentry_ProcessId] ) )
            thread_hash = f"<<THREAD>>(TID):{logentry_ThreadId}_(TS):{ProcessThread_StartTime_dict[logentry_ProcessId][logentry_ThreadId]}__(PID):{logentry_ProcessId}_(CT):{Process_StartTime_dict[logentry_ProcessId]}"
        # else: "logentry_ThreadId NOT in ProcessThread_StartTime_dict[logentry_ProcessId]" indicates this process(ProcessId) does not own this thread(ThreadId),
        #       which is contradictiory since this process started after the log-collection-start and for this log-entry to exist (with this ProcessId and ThreadId), 
        #       this process must have started this thread. 
        else:
            thread_hash = "<<THREAD>>NET-DIDITREALLYCOMEHERE"

    else: # "If process was process-started before log-collection"
        if logentry_ThreadId in ProcessThread_StartTime_dict.get(logentry_ProcessId, {}):   # "And If Thread of process was started after log-collection"   
                                                                                    #
                                                                                    #    NOTE: Whether this ThreadId is reused (stopped and started) within this process should not affect correctness.
                                                                                    #          Reason is in ProcessThread_StartTime_dict,
                                                                                    #          the old startime value of the stopped thread (which used same ThreadId) within this process
                                                                                    #          is overwritten by the starttime of newly started thread (which reuses the same ThreadId within this process).
                                                                                    #          during its THREADSTART.
                                                                                    #          i.e., 'ProcessThread_StartTime_dict' will be up-to-date (also for 'Process_StartTime_dict')
            #thread_hash = ( str(logentry_ThreadId) + str(ProcessThread_StartTime_dict[logentry_ProcessId][logentry_ThreadId]) )  +  str(logentry_ProcessId)
            thread_hash = f"<<THREAD>>(TID):{logentry_ThreadId}_(TS):{ProcessThread_StartTime_dict[logentry_ProcessId][logentry_ThreadId]}__(PID):{logentry_ProcessId}_(CT):N/A"
        else:   # "And If Thread of process started before log-collection" 
            #thread_hash = str(logentry_ThreadId) + str(logentry_ProcessId)
            thread_hash = f"<<THREAD>>(TID):{logentry_ThreadId}_(TS):N/A__(PID):{logentry_ProcessId}_(CT):N/A"
    thread_uid = get_uid(thread_hash, host, firststep_hash_debugging_mode)
    net_thread_uid_list.append(thread_uid)
    net_thread_dict[thread_uid] = {'ThreadID': logentry_ThreadID, 'ProcessID': logentry_ProcessID, 
                                    'ThreadId': logentry_ThreadId, 'ProcessId': logentry_ProcessId}

    # (Edge)
    #hashout = str(logentry_ProcessId) + str(logentry_ThreadId) + str(logentry_TimeStamp)
    hashout = f"<<NET-EDGE>>(PID):{logentry_ProcessId}_(TID):{logentry_ThreadId}_(TN):{logentry_TaskName}_(OP):{logentry_Opcode}_(TS):{logentry_TimeStamp}"
    edge_uid  = get_uid(hashout,host, firststep_hash_debugging_mode)
    net_edge_uid_list.append(edge_uid)
    net_edge_dict[edge_uid] = {'ProcessId': logentry_ProcessId, 'ProcessID': logentry_ProcessID,
                               'ThreadId': logentry_ThreadId, 'ThreadID': logentry_ThreadID,
                               'Task Name': logentry_TaskName, 'Opcode': logentry_Opcode,
                               'TimeStamp': logentry_TimeStamp,
                               
                            #    'size': logentry_size, 'SubProcessTag': logentry_SubProcessTag,
                               
                               "NET-NODE": net_uid,
                               "THREAD-NODE": thread_uid
                               }


#######################################################################################################################
# Node File----------------------------------------------------------------
def get_file_info(log_entry, host,
                  file_uid_list, file_thread_uid_list, file_edge_uid_list,
                  file_node_dict,file_thread_dict,file_edge_dict,
                  mapping,
                  
                  firststep_hash_debugging_mode = False
                  ):
           
    logentry_TaskName = log_entry.get('_source', {}).get('Task Name')
            
    if logentry_TaskName not in {'OPERATIONEND', "NAMEDELETE"}:


        logentry_Opcode = log_entry.get('_source', {}).get('Opcode')                         #feature
        logentry_ThreadId = str(log_entry.get('_source', {}).get('ThreadId'))
        logentry_ProcessId = str(log_entry.get('_source', {}).get('ProcessId'))
        logentry_ProcessID = str(log_entry.get('_source', {}).get('ProcessID'))
        logentry_TimeStamp = log_entry.get('_source', {}).get('TimeStamp')
        logentry_ThreadID = str(log_entry.get('_source', {}).get('ThreadID'))


        logentry_FileName = log_entry.get('_source', {}).get('FileName')                 # feature
        logentry_FileObject = log_entry.get('_source', {}).get('FileObject')
        # logentry_Irp = log_entry.get('_source', {}).get('Irp') #pool
        # logentry_SubProcessTag = log_entry.get('_source', {}).get('SubProcessTag')                         # feature
                


        # QUESTION: Ask Priti about this part in representing file-nodes.

        # (File-node) 
        if logentry_TaskName in {"CREATE", "CREATENEWFILE"}:

            # QUESTION: Why should the "file_hash" have a 'ProcessId' and 'ThreadId' component?
            #           Should not a file-node be able to be shared by (accessible by) different processes and threads?
            #           Incoming-edge cases for F/R/N; How can our T read from a file created by another process if file is represented in a way also dependent to PID and TID?
            #            
            #            e.g. 
            #                ProcessA (ProcessId 1) with its ThreadX (ThreadId 2) CREATEs a file with filename "hello.txt" and fileobject "abc", 
            #                then its file-uid will be "abcHELLO.TXT12", creating a file-node of uid == hash("abchello.txt12") in the CG.
            #                
            #                If ProcessB (ProcessId 9) with its ThreadY (ThreadId 8) READs that same file ("hello.txt" and fileobject "abc"),
            #                then in the CG, ProcessB's ThreadY will not be connected to the previously created 'file-node with uid==hash("abchello.txt12")'
            #                but instead will create another file-node with uid == hash( str(fileobj) + str(file_pid) + str(file_tid) ) ---> "abcHELLO.TXT98"
            #                and read from it in the CG.  ** Refer to the "else:" for ALL OTHER TASKNAMES
            #                |
            #                ㄴ> This may not necessarily be a problem for RESTRICTED-PROJECTION (Projection 2,3), but for PROJECTION-1 it might be a problem.
            #                

            #file_hash = str(logentry_FileObject)+ str(logentry_FileName.upper()) + str(logentry_ProcessId) + str(logentry_ThreadId) # > FileUID (Filepath)
            file_hash = f"<<FILE-NODE>>(FileObject):{logentry_FileObject}_(FileName):{logentry_FileName.upper()}_(PID):{logentry_ProcessId}_(TID):{logentry_ThreadId}"
            file_uid = get_uid(file_hash,host, firststep_hash_debugging_mode)
            file_uid_list.append(file_uid)
            file_node_dict[file_uid]= {'FileName': logentry_FileName, 'FileObject': logentry_FileObject}


            mapping[(logentry_FileObject, logentry_ProcessId, logentry_ThreadId)] = file_uid    # To keep track of mapping between (FileObject,ProcessId,ThreadId) tuple and corresponding file-uid
            
            
        elif logentry_TaskName == 'CLOSE':
            
            if (logentry_FileObject, logentry_ProcessId, logentry_ThreadId) in mapping:    # If this file which is being closed was created by this process and thread after log-collection start.
                
                file_uid = mapping[ (logentry_FileObject, logentry_ProcessId, logentry_ThreadId) ]  # retrieve the stored file-uid
                file_uid_list.append(file_uid)
                                              # Here, no need to do "file_node_dict[file_uid]= {'FileName': logentry_FileName, 'FileObject': logentry_FileObject}" as it was already done during creation
                
                mapping.pop( (logentry_FileObject, logentry_ProcessId, logentry_ThreadId) )    # As file closed by the process and thread created it, no long need to keep this mapping.
            

                # creation_file_dict = { hello.txt + ABC :  logentry_ProcessId, logentry_ThreadId }


            else:   # If this file which is being closed was either (1) created by this process and thread before log-collection start
                    #                                               (2) created by another process and thread before or after log-collection start -- not sure if this case is realistic, but if so handled here

                # QUESTION: What is the purpose of explicitly differentiating between files that were created after log-collection and those which are not by hash-inputs?
                #           "str(logentry_FileObject)+ str(logentry_FileName.upper()) + str(logentry_ProcessId) + str(logentry_ThreadId)"
                #                                                                 VS
                #           "str(logentry_FileObject) + str(logentry_ProcessId) + str(logentry_ThreadId)""

                #file_hash = str(logentry_FileObject) + str(logentry_ProcessId) + str(logentry_ThreadId)
                file_hash = f"<<FILE-NODE>>(FileObject):{logentry_FileObject}_(PID):{logentry_ProcessId}_(TID):{logentry_ThreadId}"
                file_uid=get_uid(file_hash,host, firststep_hash_debugging_mode)
                file_uid_list.append(file_uid)
                file_node_dict[file_uid]= {'FileName': logentry_FileName, 'FileObject': logentry_FileObject}
        
        else:   # ALL OTHER TaskNames  

                # QUESTION: What is the purpose of differentiating between files that were created after log-collection and those which are not by hash-inputs?
                #           "str(logentry_FileObject)+ str(logentry_FileName.upper()) + str(logentry_ProcessId) + str(logentry_ThreadId)"
                #                                                                 VS
                #           "str(logentry_FileObject) + str(logentry_ProcessId) + str(logentry_ThreadId)""

            #file_hash = str(logentry_FileObject) + str(logentry_ProcessId) + str(logentry_ThreadId)


            # log-entry 10 : READ 

            #'creation_file_dict[ ( logentry_FileObject , logentry_FileName ) ] = logentry_ProcessId, logentry_ThreadId'
            # >> 'FileName' is not present in READ event

            # Added by JY @ 2023-05-20 ############################################################################################
            # to handle normal-case
            if (logentry_FileObject, logentry_ProcessId, logentry_ThreadId) in mapping:
                file_uid = mapping[ (logentry_FileObject, logentry_ProcessId, logentry_ThreadId) ]
                file_uid_list.append(file_uid)
            #########################################################################################################################

            else:
                file_hash = f"<<FILE-NODE>>(FileObject):{logentry_FileObject}_(PID):{logentry_ProcessId}_(TID):{logentry_ThreadId}"
                file_uid=get_uid(file_hash,host, firststep_hash_debugging_mode)
                file_uid_list.append(file_uid)
                file_node_dict[file_uid]= {'FileName': logentry_FileName, 'FileObject': logentry_FileObject}


        
        # (Thread-node)

        #Process_StartTime_dict = get_ProcessStartTime_dict() # To get the process's starttime (if exists) for the process component of the hash-input in computing the Thread-UID
        #ProcessThread_StartTime_dict = get_ProcessThread_StartTime_dict() # To get the thread's starttime (if exists) for the thread component of the hash-input in computing the Thread-UID

        # JUST GETTING THE RIGHT PROCESS AND RIGHT PROCESS-THREAD
        thread_hash = None
        thread_uid = None
        if logentry_ProcessId in Process_StartTime_dict: # "If process was process-started after log-collection." 
            if logentry_ThreadId in ProcessThread_StartTime_dict[str(logentry_ProcessId)]:   # "And If Thread of process was started after log-collection"   
                                                                                        #
                                                                                        #    NOTE: Whether this ThreadId is reused (stopped and started) within this process should not affect correctness.
                                                                                        #          Reason is in ProcessThread_StartTime_dict,
                                                                                        #          the old startime value of the stopped thread (which used same ThreadId) within this process
                                                                                        #          is overwritten by the starttime of newly started thread (which reuses the same ThreadId within this process).
                                                                                        #          during its THREADSTART.
                                                                                        #          i.e., 'ProcessThread_StartTime_dict' will be up-to-date (also for 'Process_StartTime_dict')
                #thread_hash = ( str(logentry_ThreadId) + str(ProcessThread_StartTime_dict[logentry_ProcessId][logentry_ThreadId]) )  +  ( str(logentry_ProcessId)+str( Process_StartTime_dict[logentry_ProcessId] ) )
                thread_hash = f"<<THREAD>>(TID):{logentry_ThreadId}_(TS):{ProcessThread_StartTime_dict[logentry_ProcessId][logentry_ThreadId]}__(PID):{logentry_ProcessId}_(CT):{Process_StartTime_dict[logentry_ProcessId]}"
            # else: "logentry_ThreadId NOT in ProcessThread_StartTime_dict[logentry_ProcessId]" indicates this process(ProcessId) does not own this thread(ThreadId),
            #       which is contradictiory since this process started after the log-collection-start and for this log-entry to exist (with this ProcessId and ThreadId), 
            #       this process must have started this thread. 
            else:
                thread_hash = "<<THREAD>>FILE-DIDITREALLYCOMEHERE"

        else: # "If process was process-started before log-collection"
            if logentry_ThreadId in ProcessThread_StartTime_dict.get(logentry_ProcessId, {}):   # "And If Thread of process was started after log-collection"   
                                                                                        #
                                                                                        #    NOTE: Whether this ThreadId is reused (stopped and started) within this process should not affect correctness.
                                                                                        #          Reason is in ProcessThread_StartTime_dict,
                                                                                        #          the old startime value of the stopped thread (which used same ThreadId) within this process
                                                                                        #          is overwritten by the starttime of newly started thread (which reuses the same ThreadId within this process).
                                                                                        #          during its THREADSTART.
                                                                                        #          i.e., 'ProcessThread_StartTime_dict' will be up-to-date (also for 'Process_StartTime_dict')
                #thread_hash = ( str(logentry_ThreadId) + str(ProcessThread_StartTime_dict[logentry_ProcessId][logentry_ThreadId]) )  +  str(logentry_ProcessId)
                thread_hash = f"<<THREAD>>(TID):{logentry_ThreadId}_(TS):{ProcessThread_StartTime_dict[logentry_ProcessId][logentry_ThreadId]}__(PID):{logentry_ProcessId}_(CT):N/A"

            else:   # "And If Thread of process started before log-collection" 
                #thread_hash = str(logentry_ThreadId) + str(logentry_ProcessId)
                thread_hash = f"<<THREAD>>(TID):{logentry_ThreadId}_(TS):N/A__(PID):{logentry_ProcessId}_(CT):N/A"

        thread_uid = get_uid(thread_hash, host, firststep_hash_debugging_mode)
        file_thread_uid_list.append(thread_uid)
        file_thread_dict[thread_uid] = {'ThreadID': logentry_ThreadID, 'ProcessID': logentry_ProcessID, 
                                        'ThreadId': logentry_ThreadId, 'ProcessId': logentry_ProcessId}

        # (Edge)
        #hashout = str(logentry_ProcessId) + str(logentry_ThreadId) + str(logentry_TimeStamp)
        hashout = f"<<FILE-EDGE>>(PID):{logentry_ProcessId}_(TID):{logentry_ThreadId}_(TN):{logentry_TaskName}_(OP):{logentry_Opcode}_(TS):{logentry_TimeStamp}"       
        edge_uid = get_uid(hashout,host, firststep_hash_debugging_mode)
        file_edge_uid_list.append(edge_uid)
        file_edge_dict[edge_uid]= {'ProcessId': logentry_ProcessId, 'ProcessID': logentry_ProcessID,
                                   'ThreadId': logentry_ThreadId, 'ThreadID': logentry_ThreadID,
                                   'Task Name': logentry_TaskName, 'Opcode': logentry_Opcode,
                                   'TimeStamp': logentry_TimeStamp,
                                   
                                #    'Irp': logentry_Irp, 'SubProcessTag': logentry_SubProcessTag,
                                   
                                   'FILE-NODE': file_uid,
                                   'THREAD-NODE': thread_uid
                                   }


#####################################################################################################################################################################
# Registry File----------------------------------------------------------------
def get_reg_info(log_entry, host, 
                 reg_uid_list, reg_thread_uid_list, reg_edge_uid_list, 
                 reg_node_dict, reg_thread_dict, reg_edge_dict,
                 mapping,
                 
                 firststep_hash_debugging_mode = False
                 ):
        
    #valname = d.get('_source', {}).get('ValueName')
    logentry_TimeStamp = log_entry.get('_source', {}).get('TimeStamp')     
    logentry_TaskName = log_entry.get('_source', {}).get('Task Name')
    logentry_Opcode = log_entry.get('_source', {}).get('Opcode')
    logentry_ProcessId = str(log_entry.get('_source', {}).get('ProcessId'))
    logentry_ProcessID = str(log_entry.get('_source', {}).get('ProcessID'))
    logentry_ThreadId = str(log_entry.get('_source', {}).get('ThreadId'))
    logentry_ThreadID = str(log_entry.get('_source', {}).get('ThreadID'))

    logentry_KeyObject = log_entry.get('_source', {}).get('KeyObject')
    logentry_RelativeName = log_entry.get('_source', {}).get('RelativeName') # feature


    # logentry_Status = log_entry.get('_source', {}).get('Status') 
    # logentry_Disposition = log_entry.get('_source', {}).get('Disposition')
    # logentry_SubProcessTag = log_entry.get('_source', {}).get('SubProcessTag') #feature



    
    # (Registry-node)

    # Exact same logic as (File-node) in "get_file_info()".
    # Thus, I have the exact same questions.

    if logentry_Opcode in {32,33}:  # For Registry events, Opcode 32 and 33 should correspond to "CREATE" and "CREATENEWFILE" in File events.
                                    # Reason for using Opcode instead of TaskName should be that Registry events all appear to have one taskname of "Microsofy-Kernel.. Registry"

            # QUESTION: Why should the "reg_hash" have a 'ProcessId' and 'ThreadId' component?
            #           Should not a reg-node be able to be shared by (accessible by) different processes and threads?
            #           Incoming-edge cases for F/R/N; How can our T read from a file created by another process if registry is represented in a way also dependent to PID and TID?
            #            
            #           (For Example, Refer to the example in get_file_info() counterpart.) 
            #             


        #reg_hash = str(logentry_KeyObject) + str(logentry_RelativeName.upper()) + str(logentry_ProcessId) + str(logentry_ThreadId)
        reg_hash = f"<<REG-NODE>>(KeyObject):{logentry_KeyObject}_(RelativeName):{logentry_RelativeName.upper()}_(PID):{logentry_ProcessId}_(TID):{logentry_ThreadId}"
        reg_uid = get_uid(reg_hash,host, firststep_hash_debugging_mode)
        reg_uid_list.append(reg_uid)
        
        reg_node_dict[reg_uid] = {'KeyObject': logentry_KeyObject, 'RelativeName': logentry_RelativeName} # > Regisrty UID (KeyName + KeyObject)

        mapping[(logentry_KeyObject, logentry_ProcessId, logentry_ThreadId)] = reg_uid  # To keep track of mapping between (KeyObject,ProcessId,ThreadId) tuple and corresponding reg_uid

    elif logentry_Opcode == 44: # For Registry events, Opcode 44 should correspond to "CLOSE" in File events.

        if (logentry_KeyObject, logentry_ProcessId, logentry_ThreadId) in mapping:  # If this registry which is being closed was created by this process and thread after log-collection start.
            reg_uid = mapping[(logentry_KeyObject, logentry_ProcessId, logentry_ThreadId)] # retrieve the stored reg-uid
            reg_uid_list.append(reg_uid)
                                              # Here, no need to do "reg_node_dict[reg_uid]= {'KeyObject':logentry_KeyObject, 'RelativeName':logentry_RelativeName}" as it was already done during creation
            mapping.pop((logentry_KeyObject, logentry_ProcessId, logentry_ThreadId)) # As registry closed by the process and thread created it, no long need to keep this mapping.
        
        else:       # If this registry which is being closed was either (1) created by this process and thread before log-collection start
                    #                                                   (2) created by another process and thread before or after log-collection start -- not sure if this case is realistic, but if so handled here
            #reg_hash = str(logentry_KeyObject) + str(logentry_ProcessId) + str(logentry_ThreadId)
            reg_hash = f"<<REG-NODE>>(KeyObject):{logentry_KeyObject}_(PID):{logentry_ProcessId}_(TID):{logentry_ThreadId}"
            reg_uid = get_uid(reg_hash,host, firststep_hash_debugging_mode)
            reg_uid_list.append(reg_uid)
            reg_node_dict[reg_uid] = {'KeyObject': logentry_KeyObject, 'RelativeName': logentry_RelativeName}
    
    else: # ALL OTHER Opcodes  

          # QUESTION: What is the purpose of explicitly differentiating between registries that were created after log-collection-start and those which are not, by hash-inputs?
          #           "str(logentry_KeyObject) + str(logentry_RelativeName.upper()) + str(logentry_ProcessId) + str(logentry_ThreadId)"
          #                                                                 VS
          #           "str(logentry_KeyObject) + str(logentry_ProcessId) + str(logentry_ThreadId)""


        # Added by JY @ 2023-05-20 ############################################################################################
        # to handle normal-case
        if (logentry_KeyObject, logentry_ProcessId, logentry_ThreadId) in mapping:
            reg_uid = mapping[(logentry_KeyObject, logentry_ProcessId, logentry_ThreadId)] # retrieve the stored reg-uid
            reg_uid_list.append(reg_uid)
        #########################################################################################################################
        else:
            #reg_hash = str(logentry_KeyObject) + str(logentry_ProcessId) + str(logentry_ThreadId)
            reg_hash = f"<<REG-NODE>>(KeyObject):{logentry_KeyObject}_(PID):{logentry_ProcessId}_(TID):{logentry_ThreadId}"
            reg_uid = get_uid(reg_hash,host, firststep_hash_debugging_mode)
            reg_uid_list.append(reg_uid)
            reg_node_dict[reg_uid] = {'KeyObject': logentry_KeyObject, 'RelativeName': logentry_RelativeName}
        
        

    # (Thread-node)

    #Process_StartTime_dict = get_ProcessStartTime_dict() # To get the process's starttime (if exists) for the process component of the hash-input in computing the Thread-UID
    #ProcessThread_StartTime_dict = get_ProcessThread_StartTime_dict() # To get the thread's starttime (if exists) for the thread component of the hash-input in computing the Thread-UID

    # JUST GETTING THE RIGHT PROCESS AND RIGHT PROCESS-THREAD
    thread_hash = None
    thread_uid = None
    if logentry_ProcessId in Process_StartTime_dict: # "If process was process-started after log-collection." 
        if logentry_ThreadId in ProcessThread_StartTime_dict[logentry_ProcessId]:   # "And If Thread of process was started after log-collection"   
                                                                                    #
                                                                                    #    NOTE: Whether this ThreadId is reused (stopped and started) within this process should not affect correctness.
                                                                                    #          Reason is in ProcessThread_StartTime_dict,
                                                                                    #          the old startime value of the stopped thread (which used same ThreadId) within this process
                                                                                    #          is overwritten by the starttime of newly started thread (which reuses the same ThreadId within this process).
                                                                                    #          during its THREADSTART.
                                                                                    #          i.e., 'ProcessThread_StartTime_dict' will be up-to-date (also for 'Process_StartTime_dict')
            #thread_hash = ( str(logentry_ThreadId) + str(ProcessThread_StartTime_dict[logentry_ProcessId][logentry_ThreadId]) )  +  ( str(logentry_ProcessId)+str( Process_StartTime_dict[logentry_ProcessId] ) )
            thread_hash = f"<<THREAD>>(TID):{logentry_ThreadId}_(TS):{ProcessThread_StartTime_dict[logentry_ProcessId][logentry_ThreadId]}__(PID):{logentry_ProcessId}_(CT):{Process_StartTime_dict[logentry_ProcessId]}"

        # else: "logentry_ThreadId NOT in ProcessThread_StartTime_dict[logentry_ProcessId]" indicates this process(ProcessId) does not own this thread(ThreadId),
        #       which is contradictiory since this process started after the log-collection-start and for this log-entry to exist (with this ProcessId and ThreadId), 
        #       this process must have started this thread. 
        else:
            thread_hash = "<<THREAD>>REG-DIDITREALLYCOMEHERE"

    else: # "If process was process-started before log-collection"
        if logentry_ThreadId in ProcessThread_StartTime_dict.get(logentry_ProcessId, {}):   # "And If Thread of process was started after log-collection"   
                                                                                    #
                                                                                    #    NOTE: Whether this ThreadId is reused (stopped and started) within this process should not affect correctness.
                                                                                    #          Reason is in ProcessThread_StartTime_dict,
                                                                                    #          the old startime value of the stopped thread (which used same ThreadId) within this process
                                                                                    #          is overwritten by the starttime of newly started thread (which reuses the same ThreadId within this process).
                                                                                    #          during its THREADSTART.
                                                                                    #          i.e., 'ProcessThread_StartTime_dict' will be up-to-date (also for 'Process_StartTime_dict')
            #thread_hash = ( str(logentry_ThreadId) + str(ProcessThread_StartTime_dict[logentry_ProcessId][logentry_ThreadId]) )  +  str(logentry_ProcessId)
            thread_hash = f"<<THREAD>>(TID):{logentry_ThreadId}_(TS):{ProcessThread_StartTime_dict[logentry_ProcessId][logentry_ThreadId]}__(PID):{logentry_ProcessId}_(CT):N/A"

        else:   # "And If Thread of process started before log-collection" 
            #thread_hash = str(logentry_ThreadId) + str(logentry_ProcessId)
            thread_hash = f"<<THREAD>>(TID):{logentry_ThreadId}_(TS):N/A__(PID):{logentry_ProcessId}_(CT):N/A"

    thread_uid = get_uid(thread_hash, host, firststep_hash_debugging_mode)
    reg_thread_uid_list.append(thread_uid)
    reg_thread_dict[thread_uid] = {'ThreadID': logentry_ThreadID, 'ProcessID': logentry_ProcessID, 
                                    'ThreadId': logentry_ThreadId, 'ProcessId': logentry_ProcessId}


    # (Edge)
    #hashout = str(logentry_ProcessId) + str(logentry_ThreadId) + str(logentry_TimeStamp)
    hashout = f"<<REG-EDGE>>(PID):{logentry_ProcessId}_(TID):{logentry_ThreadId}_(TN):{logentry_TaskName}_(OP):{logentry_Opcode}_(TS):{logentry_TimeStamp}"

    edge_uid = get_uid(hashout,host, firststep_hash_debugging_mode)
    reg_edge_uid_list.append(edge_uid)
    reg_edge_dict[edge_uid]= {'ProcessId': logentry_ProcessId, 'ProcessID': logentry_ProcessID,
                              'ThreadId': logentry_ThreadId, 'ThreadID': logentry_ThreadID,
                              'Task Name': logentry_TaskName, 'Opcode': logentry_Opcode,
                              'TimeStamp': logentry_TimeStamp,
                              
                            #   'Status': logentry_Status, 'SubProcessTag': logentry_SubProcessTag, 'Disposition': logentry_Disposition,
                              
                              'REG-NODE': reg_uid,
                              'THREAD-NODE': thread_uid
                              }



#######################################################################################################################
#######################################################################################################################
#######################################################################################################################
#######################################################################################################################
#######################################################################################################################


def csv_file(csv_root,
             
             file_thread_uid_list, net_thread_uid_list, proc_thread_uid_list, reg_thread_uid_list,
             file_uid_list, net_uid_list, proc_uid_list, reg_uid_list,
             file_edge_uid_list, net_edge_uid_list, proc_edge_uid_list, reg_edge_uid_list):

    '''
    import pandas as pd
    proc_data = pd.DataFrame({'Thread': proc_thread_uid_list, 'Node': proc_uid_list ,'Edge': proc_edge_uid_list})
    net_data = pd.DataFrame({'Thread': net_thread_uid_list, 'Node': net_uid_list ,'Edge': net_edge_uid_list})
    file_data = pd.DataFrame({'Thread': file_thread_uid_list, 'Node': file_uid_list ,'Edge': file_edge_uid_list})
    reg_data = pd.DataFrame({'Thread': reg_thread_uid_list, 'Node': reg_uid_list ,'Edge': reg_edge_uid_list})
    proc_data.to_csv(os.path.join(csv_root,'proc.csv'))
    net_data.to_csv(os.path.join(csv_root,'net.csv'))
    file_data.to_csv(os.path.join(csv_root,'file.csv'))
    reg_data.to_csv(os.path.join(csv_root,'reg.csv'))
    '''
    
    f1= open(os.path.join(csv_root,'proc.csv'), 'w', encoding='UTF8')   # Technically we are not producing csv (comma-separated-files)
    f2= open(os.path.join(csv_root,'net.csv'), 'w', encoding='UTF8')
    f3= open(os.path.join(csv_root,'file.csv'), 'w', encoding='UTF8')
    f4= open(os.path.join(csv_root,'reg.csv'), 'w', encoding='UTF8')
    #writer1 = csv.writer(f1, delimiter=' ', lineterminator='\n', )
    #writer2= csv.writer(f2, delimiter=' ', lineterminator='\n', )
    #writer3 = csv.writer(f3, delimiter=' ', lineterminator='\n',)
    #writer4= csv.writer(f4, delimiter=' ', lineterminator='\n', )
    print(len(proc_edge_uid_list), len(net_edge_uid_list), len(file_edge_uid_list), len(reg_edge_uid_list))
    print(len(proc_thread_uid_list), len(net_thread_uid_list), len(file_thread_uid_list), len(reg_thread_uid_list))
    print(len(proc_uid_list), len(net_uid_list), len(file_uid_list), len(reg_uid_list))
    

    # ORDER: (THREAD , NODE , EDGE)


    proc_uid_list = [x.replace(' ','') for x in proc_uid_list] # GET RID OF POTENTIAL ' ' in the RegistryName as it confuses Graph.Read_Ncol, 
                                                                # since ' ' should be only used to separate [thread,node,edge]
    proc_edge_uid_list = [x.replace(' ','') for x in proc_edge_uid_list] # Because of TaskName "THIS GROUP OF EVENTS TRACKS THE PERFORMANCE OF LOADING HIVES"
    proc_thread_uid_list = [x.replace(' ','') for x in proc_thread_uid_list]    # JUST IN CASE
    for i in range(len(proc_edge_uid_list)):
        #data1 = [proc_thread_uid_list[i], proc_uid_list[i], proc_edge_uid_list[i]]
        #writer1.writerow(data1)
        row = f"{proc_thread_uid_list[i]} {proc_uid_list[i]} {proc_edge_uid_list[i]}\n"
        f1.write(row)

    net_uid_list = [x.replace(' ','') for x in net_uid_list] # GET RID OF POTENTIAL ' ' in the RegistryName as it confuses Graph.Read_Ncol, 
                                                                # since ' ' should be only used to separate [thread,node,edge]
    net_edge_uid_list = [x.replace(' ','') for x in net_edge_uid_list] # Because of TaskName "THIS GROUP OF EVENTS TRACKS THE PERFORMANCE OF LOADING HIVES"
    net_thread_uid_list = [x.replace(' ','') for x in net_thread_uid_list]    # JUST IN CASE
    for i in range(len(net_edge_uid_list)):
        #data2 = [net_thread_uid_list[i], net_uid_list[i], net_edge_uid_list[i]]        
        #writer2.writerow(data2)
        row = f"{net_thread_uid_list[i]} {net_uid_list[i]} {net_edge_uid_list[i]}\n"
        f2.write(row)


    file_uid_list = [x.replace(' ','') for x in file_uid_list] # GET RID OF POTENTIAL ' ' in the RegistryName as it confuses Graph.Read_Ncol, 
                                                                # since ' ' should be only used to separate [thread,node,edge]
    file_edge_uid_list = [x.replace(' ','') for x in file_edge_uid_list] # Because of TaskName "THIS GROUP OF EVENTS TRACKS THE PERFORMANCE OF LOADING HIVES"
    file_thread_uid_list = [x.replace(' ','') for x in file_thread_uid_list]    # JUST IN CASE
    for i in range(len(file_edge_uid_list)):
        #data3 = [file_thread_uid_list[i], file_uid_list[i], file_edge_uid_list[i]]                
        #writer3.writerow(data3)
        row = f"{file_thread_uid_list[i]} {file_uid_list[i]} {file_edge_uid_list[i]}\n"        
        f3.write(row)



    reg_uid_list = [x.replace(' ','') for x in reg_uid_list]    # GET RID OF POTENTIAL ' ' in the RegistryName as it confuses Graph.Read_Ncol, 
                                                                # since ' ' should be only used to separate [thread,node,edge]
    reg_edge_uid_list = [x.replace(' ','') for x in reg_edge_uid_list] # Because of TaskName "THIS GROUP OF EVENTS TRACKS THE PERFORMANCE OF LOADING HIVES"
    reg_thread_uid_list = [x.replace(' ','') for x in reg_thread_uid_list]  # JUST IN CASE
    for i in range(len(reg_edge_uid_list)):
        #data4 = [reg_thread_uid_list[i], reg_uid_list[i], reg_edge_uid_list[i]]                
        #writer4.writerow(data4)
        row = f"{reg_thread_uid_list[i]} {reg_uid_list[i]} {reg_edge_uid_list[i]}\n"
        f4.write(row)

    f1.close()
    f2.close()
    f3.close()
    f4.close()

    ##################
    # COULD GET RID OF ALL double-quotes(") IF IT APPEARS AS A PROBLEM
    #f1= open(os.path.join(csv_root,'proc.csv'), 'r', encoding='UTF8')   # Technically we are not producing csv (comma-separated-files)
    #f2= open(os.path.join(csv_root,'net.csv'), 'r', encoding='UTF8')
    #f3= open(os.path.join(csv_root,'file.csv'), 'r', encoding='UTF8')
    #f4= open(os.path.join(csv_root,'reg.csv'), 'r', encoding='UTF8')
    #for rowstring in f1.readlines():
    #    rowstring = rowstring.replace("\"",'')
    #for rowstring in f2.readlines():
    #    rowstring = rowstring.replace("\"",'')        
    #for rowstring in f3.readlines():
    #    if "\"" in rowstring:
    #        rowstring = rowstring.replace("\"",'')        
    #for rowstring in f4.readlines():
    #    rowstring = rowstring.replace("\"",'')
    #f1.close()
    #f2.close()
    #f3.close()
    #f4.close()
    


def get_graph(csv_root, graph_root, file_edge_uid_list, net_edge_uid_list, proc_edge_uid_list, reg_edge_uid_list):
    # csv root to store csv graph root to store graph 
    import igraph 

    # If this error "Error at src/io/ncol.c:151: Parse error in NCOL file, line 1 (syntax error, unexpected ALNUM, expecting NEWLINE). -- Parse error",
    # Its because of spaces in hash.
    g1 = Graph.Read_Ncol(os.path.join(csv_root,"proc.csv"), directed=True, names = True, weights =False)
    g1.es["name"] = proc_edge_uid_list
   # g1.es["Task"] = proc_event
    #Netwrok
    g2 = Graph.Read_Ncol(os.path.join(csv_root,"net.csv"), directed=True, names = True, weights =False)
    g2.es["name"] = net_edge_uid_list
    #g2.es["Task"] = net_event
    

    #File
    g3 = Graph.Read_Ncol(os.path.join(csv_root,"file.csv"), directed=True, names = True, weights =False)
    g3.es["name"] = file_edge_uid_list
    #g3.es["Task"] = file_event
    

    #Registry
    g4 = Graph.Read_Ncol(os.path.join(csv_root,"reg.csv"), directed=True, names = True, weights =False)
    g4.es["name"] = reg_edge_uid_list
    #g4.es["Task"] = reg_event

    # Union
    #final_union = Graph.union(g1,[g2,g3,g4], byname = 'auto')
    
    final_union = Graph.union(g1,[g2,g3,g4], byname = 'auto')
    
    #final_union.vs["color"] = g1.vs["color"]+g2.vs["color"]+g3.vs["color"]+g4.vs["color"]
    #final_union.vs["label"] = g1.vs["label"]+g2.vs["label"]+g3.vs["label"]+g4.vs["label"]
    try:
        final = final_union.simplify( multiple = True, combine_edges = ','.join)
    except:
        # Exception has occurred: TypeError
        # sequence item 0: expected str instance, NoneType found
        
        # Debugger [e for e in final_union.es] returns following
        # igraph.Edge(<igraph.Graph object at 0x7fb2401dfc70>, 235, {'name_1': None, 'name_2': None, 'name_3': '28bb59a8-2b3f-f98b-b046-a778a47faa78', 'name_4': None})
        # igraph.Edge(<igraph.Graph object at 0x7fb2401dfc70>, 235, {'name_1': None, 'name_2': None, 'name_3': '28bb59a8-2b3f-f98b-b046-a778a47faa78', 'name_4': None})
        # > We want to get rid of the Nones, and just have "'name': 32234324-afaf..." pair

        edgeindex_correctval_pair = dict()
        for e in final_union.es:
            # e.g. {'name_1': '291d4494-7ab2-1cdf-3bd6-6858a47faa78', 'name_2': None, 'name_3': None, 'name_4': None}
            edge_attrs = e.attributes() 
            # Try to get the correct value for all edges (e.g. '291d4494-7ab2-1cdf-3bd6-6858a47faa78') * there will be only 1 or None
            not_None_vals = [ v for k,v in edge_attrs.items() if v != None ]
            if len(not_None_vals) == 0: # UNLIKELY BUT JUST IN CASE
                # will later need to just drop this (before integraed) edge as carries no value.
                edgeindex_correctval_pair[e.index] = None
            else:             
                edgeindex_correctval_pair[e.index] = not_None_vals[0]   # if exists will be only one.

        # Now delete all "name_<int>"s in the igraph-level (internally, it seems they are contiguous, so generally need caution; but following appears safely deallocates)
        # > https://stackoverflow.com/questions/55291777/how-to-remove-an-edge-attribute-in-python-igraph
        for name_X__attr in edge_attrs:
            del( final_union.es[name_X__attr] )

        # Now
        for e in final_union.es: 
            
            if edgeindex_correctval_pair[e.index] == None:
                # this edge carries no value just get rid of it
                final_union.delete_edges([e.index])
            else:
                # Now use python-igraph "update_attribute" functionality for the edge to only have "name: <correct_value>" 
                e.update_attributes({"name": edgeindex_correctval_pair[e.index]})
                
        # Since we handled the source-of-error, do final_union.simplify
        final = final_union.simplify( multiple = True, combine_edges = ','.join)

    #final.write(os.path.join(graph_root,"Union.dot"))
    final.write_graphml(os.path.join(graph_root,"Union.GraphML"))
    #return final




def get_multi_graph(csv_root, graph_root, file_edge_uid_list, net_edge_uid_list, proc_edge_uid_list, reg_edge_uid_list):
    
    ''' JY @ 2023-05-20:
    
        NOT GOING THROUGH "final = final_union.simplify( multiple = True, combine_edges = ','.join)"
    
    '''
    
    # csv root to store csv graph root to store graph 
    import igraph 

    # If this error "Error at src/io/ncol.c:151: Parse error in NCOL file, line 1 (syntax error, unexpected ALNUM, expecting NEWLINE). -- Parse error",
    # Its because of spaces in hash.
    g1 = Graph.Read_Ncol(os.path.join(csv_root,"proc.csv"), directed=True, names = True, weights =False)
    g1.es["name"] = proc_edge_uid_list
   # g1.es["Task"] = proc_event
    #Netwrok
    g2 = Graph.Read_Ncol(os.path.join(csv_root,"net.csv"), directed=True, names = True, weights =False)
    g2.es["name"] = net_edge_uid_list
    #g2.es["Task"] = net_event
    

    #File
    g3 = Graph.Read_Ncol(os.path.join(csv_root,"file.csv"), directed=True, names = True, weights =False)
    g3.es["name"] = file_edge_uid_list
    #g3.es["Task"] = file_event
    

    #Registry
    g4 = Graph.Read_Ncol(os.path.join(csv_root,"reg.csv"), directed=True, names = True, weights =False)
    g4.es["name"] = reg_edge_uid_list
    #g4.es["Task"] = reg_event

    # Union
    #final_union = Graph.union(g1,[g2,g3,g4], byname = 'auto')
    
    final_union = Graph.union(g1,[g2,g3,g4], byname = 'auto')
    
    final_union.write_graphml(os.path.join(graph_root,"Union.GraphML"))

    #


    #final_union.vs["color"] = g1.vs["color"]+g2.vs["color"]+g3.vs["color"]+g4.vs["color"]
    #final_union.vs["label"] = g1.vs["label"]+g2.vs["label"]+g3.vs["label"]+g4.vs["label"]
    
    
    # try:
    #     final = final_union.simplify( multiple = True, combine_edges = ','.join)
    # except:
    #     # Exception has occurred: TypeError
    #     # sequence item 0: expected str instance, NoneType found
        
    #     # Debugger [e for e in final_union.es] returns following
    #     # igraph.Edge(<igraph.Graph object at 0x7fb2401dfc70>, 235, {'name_1': None, 'name_2': None, 'name_3': '28bb59a8-2b3f-f98b-b046-a778a47faa78', 'name_4': None})
    #     # igraph.Edge(<igraph.Graph object at 0x7fb2401dfc70>, 235, {'name_1': None, 'name_2': None, 'name_3': '28bb59a8-2b3f-f98b-b046-a778a47faa78', 'name_4': None})
    #     # > We want to get rid of the Nones, and just have "'name': 32234324-afaf..." pair

    #     edgeindex_correctval_pair = dict()
    #     for e in final_union.es:
    #         # e.g. {'name_1': '291d4494-7ab2-1cdf-3bd6-6858a47faa78', 'name_2': None, 'name_3': None, 'name_4': None}
    #         edge_attrs = e.attributes() 
    #         # Try to get the correct value for all edges (e.g. '291d4494-7ab2-1cdf-3bd6-6858a47faa78') * there will be only 1 or None
    #         not_None_vals = [ v for k,v in edge_attrs.items() if v != None ]
    #         if len(not_None_vals) == 0: # UNLIKELY BUT JUST IN CASE
    #             # will later need to just drop this (before integraed) edge as carries no value.
    #             edgeindex_correctval_pair[e.index] = None
    #         else:             
    #             edgeindex_correctval_pair[e.index] = not_None_vals[0]   # if exists will be only one.

    #     # Now delete all "name_<int>"s in the igraph-level (internally, it seems they are contiguous, so generally need caution; but following appears safely deallocates)
    #     # > https://stackoverflow.com/questions/55291777/how-to-remove-an-edge-attribute-in-python-igraph
    #     for name_X__attr in edge_attrs:
    #         del( final_union.es[name_X__attr] )

    #     # Now
    #     for e in final_union.es: 
            
    #         if edgeindex_correctval_pair[e.index] == None:
    #             # this edge carries no value just get rid of it
    #             final_union.delete_edges([e.index])
    #         else:
    #             # Now use python-igraph "update_attribute" functionality for the edge to only have "name: <correct_value>" 
    #             e.update_attributes({"name": edgeindex_correctval_pair[e.index]})
    #     # Since we handled the source-of-error, do final_union.simplify
    #     final = final_union.simplify( multiple = True, combine_edges = ','.join)




    #final.write(os.path.join(graph_root,"Union.dot"))
    #final.write_graphml(os.path.join(graph_root,"Union.GraphML"))
    #return final







def get_dicts(dicts_root,file_edge_dict,file_node_dict,file_thread_dict, proc_edge_dict, proc_node_dict, proc_thread_dict,
            reg_edge_dict, reg_node_dict, reg_thread_dict, net_edge_dict, net_node_dict, net_thread_dict):
    # Dictinary to hold 'name' attribute with its source and target pair
    # zipiter = zip(final_g.es['name'],final_g.get_edgelist() )
    # my_dict = dict(zipiter)


    # Saving dictionary to file
    # src_tar = open(os.path.join(dicts_root,"src_tar.json"), "w")
    # json.dump(my_dict,src_tar)
    # src_tar.close()

    # THIS IS A TREATMENT TO ENSURE THE UIDS(keys) IN DICT MATCHES UIDS IN CG; 
    # OTHERWISE, THERE CAN BE PROBLEMS IN earliest_time() of Projection 
    proc_node_dict = { k.replace(' ','') : v for k,v in proc_node_dict.items() }
    proc_edge_dict = { k.replace(' ','') : v for k,v in proc_edge_dict.items() }
    proc_thread_dict = { k.replace(' ','') : v for k,v in proc_thread_dict.items() }

    net_node_dict = { k.replace(' ','') : v for k,v in net_node_dict.items() }
    net_edge_dict = { k.replace(' ','') : v for k,v in net_edge_dict.items() }
    net_thread_dict = { k.replace(' ','') : v for k,v in net_thread_dict.items() }

    file_node_dict = { k.replace(' ','') : v for k,v in file_node_dict.items() }
    file_edge_dict = { k.replace(' ','') : v for k,v in file_edge_dict.items() }
    file_thread_dict = { k.replace(' ','') : v for k,v in file_thread_dict.items() }

    reg_node_dict = { k.replace(' ','') : v for k,v in reg_node_dict.items() }
    reg_edge_dict = { k.replace(' ','') : v for k,v in reg_edge_dict.items() }
    reg_thread_dict = { k.replace(' ','') : v for k,v in reg_thread_dict.items() }


    file_node = open(os.path.join(dicts_root,"file_node.json"), "w")
    json.dump(file_node_dict,file_node)
    file_node.close()

    file_edge = open(os.path.join(dicts_root,"file_edge.json"), "w")
    json.dump(file_edge_dict,file_edge)
    file_edge.close()

    file_thread = open(os.path.join(dicts_root,"file_thread.json"), "w")
    json.dump(file_thread_dict,file_thread)
    file_thread.close()


    net_node = open(os.path.join(dicts_root,"net_node.json"), "w")
    json.dump(net_node_dict,net_node)
    net_node.close()

    net_edge = open(os.path.join(dicts_root,"net_edge.json"), "w")
    json.dump(net_edge_dict,net_edge)
    net_edge.close()

    net_thread = open(os.path.join(dicts_root,"net_thread.json"), "w")
    json.dump(net_thread_dict,net_thread)
    net_thread.close()


    proc_node = open(os.path.join(dicts_root,"proc_node.json"), "w")
    json.dump(proc_node_dict,proc_node)
    proc_node.close()

    proc_edge = open(os.path.join(dicts_root,"proc_edge.json"), "w")
    json.dump(proc_edge_dict,proc_edge)
    proc_edge.close()

    proc_thread = open(os.path.join(dicts_root,"proc_thread.json"), "w")
    json.dump(proc_thread_dict,proc_thread)
    proc_thread.close()


    reg_node = open(os.path.join(dicts_root,"reg_node.json"), "w")
    json.dump(reg_node_dict,reg_node)
    reg_node.close()

    reg_edge = open(os.path.join(dicts_root,"reg_edge.json"), "w")
    json.dump(reg_edge_dict,reg_edge)
    reg_edge.close()

    reg_thread = open(os.path.join(dicts_root,"reg_thread.json"), "w")
    json.dump(reg_thread_dict,reg_thread)
    reg_thread.close()

    
def first_step(idx, root_path, EventTypes_to_Exclude_set : set, firststep_hash_debugging_mode : bool = False):
    
    # Added by JY @ 2023-03-02: Added 3rd argument "EventTypes_to_Exclude_set"    
    #  Elements in 'EventTypes_to_Exclude_set' will be in format of 
    #  f"{Task Name in original-format as in raw event-log}{Opcode (optional; only for Registry/Network events)}" 
    #  ^ Notice that no space between TaskName and Opcode.

    # Added by JY @ 2023-03-02: Lower-case all event-types to Exclude
    EventTypes_to_Exclude_set = { x.lower() for x in EventTypes_to_Exclude_set }



    #################################################################################################################################
    # Registry attributes
    reg_node_dict ={}
    reg_thread_dict = {}
    reg_edge_dict = {}
    
    reg_uid_list = []
    reg_thread_uid_list = []
    reg_edge_uid_list = []

    # File attributes
    file_node_dict = {}
    file_thread_dict ={}
    file_edge_dict = {}

    file_uid_list = []
    file_thread_uid_list = []
    file_edge_uid_list = []

    # Network attributes
    net_node_dict = {}
    net_thread_dict ={}
    net_edge_dict = {}

    net_uid_list = []
    net_thread_uid_list = []
    net_edge_uid_list = []

    # Process attributes
    proc_node_dict = {}
    proc_thread_dict = {}
    proc_edge_dict = {}

    proc_uid_list = []
    procthread_uid_list = []
    proc_edge_uid_list = []
    #################################################################################################################################

    created_filenode_to_fileuid_mapping = {}
    created_regnode_to_reguid_mapping = {}
    
    #################################################################################################################################
    

    FILE_provider = "{EDD08927-9CC4-4E65-B970-C2560FB5C289}"
    NETWORK_provider = "{7DD42A49-5329-4832-8DFD-43D979153A88}"
    PROCESS_provider = "{22FB2CD6-0E7B-422B-A0C7-2FAD1FD0E716}"
    REGISTRY_provider = "{70EB4F03-C1DE-4F73-A051-33D13D5413BD}"

    all_log_entries = from_elastic(idx)
    host_name = get_host() # part of the hash-function

    # Get providers data
    for i, log_entry in enumerate(all_log_entries):

        #  Added by JY @ 2023-03-02: Lower-case all event-types to Exclude----------
        logentry_TaskName = log_entry.get('_source', {}).get('Task Name')
        logentry_Opcode = log_entry.get('_source', {}).get('Opcode')
            # Now using this logentry's TaskName and Opcode, 
            # Get the TaskNameOpcode that is in format compatible with which was in 'EventTypes_to_Exclude_set'.
            #
            #  Elements in 'EventTypes_to_Exclude_set' will be in format of 
            #  f"{Task Name in original-format as in raw event-log}{Opcode (optional; only for Registry/Network events)}" 
            #  ^ Notice that no space between TaskName and Opcode.
            #  > To check "Task Name in original-format as in raw event-log" ---> Go to ElasticSearch
            # >  To check "Opcode (optional; only for Registry/Network events)" ---> Go to ~/tabby/BASELINE_COMPARISONS/Sequential/RF+Ngrams/RFSVM_ngram_flattened_subgraph.py
        logentry_TaskNameOpcode_in_matching_format = f"{logentry_TaskName.lower()}{logentry_Opcode}"
        if logentry_TaskNameOpcode_in_matching_format in EventTypes_to_Exclude_set:
            continue
        # --------------------------------------------------------------------------



        provider = log_entry["_source"].get('ProviderId')
         
        if provider == PROCESS_provider:
            get_proc_info(log_entry, host_name, 
                          proc_uid_list, procthread_uid_list, proc_edge_uid_list, 
                          proc_node_dict, proc_thread_dict, proc_edge_dict,

                          firststep_hash_debugging_mode
                          )

        if provider == NETWORK_provider:
            get_net_info(log_entry, host_name, 
                         net_uid_list, net_thread_uid_list, net_edge_uid_list, 
                         net_node_dict, net_thread_dict, net_edge_dict,
                         
                         firststep_hash_debugging_mode
                         )

        if provider == FILE_provider:    
            get_file_info(log_entry, host_name, 
                          file_uid_list, file_thread_uid_list, file_edge_uid_list, 
                          file_node_dict, file_thread_dict, file_edge_dict,
                          created_filenode_to_fileuid_mapping,
                          
                          firststep_hash_debugging_mode
                          )

        if provider == REGISTRY_provider:
            get_reg_info(log_entry, host_name, 
                         reg_uid_list, reg_thread_uid_list, reg_edge_uid_list,  
                         reg_node_dict, reg_thread_dict, reg_edge_dict,
                         created_regnode_to_reguid_mapping,

                         firststep_hash_debugging_mode
                         )

    #################################################################################################################################

    # get csv files for graph
    csv_file(root_path, file_thread_uid_list, net_thread_uid_list, procthread_uid_list, reg_thread_uid_list, 
                        file_uid_list, net_uid_list, proc_uid_list, reg_uid_list,
                        file_edge_uid_list, net_edge_uid_list, proc_edge_uid_list, reg_edge_uid_list) 


    #################################################################################################################################

    # # Generating computation graphs
    # get_graph(root_path, root_path, 
    #           file_edge_uid_list, net_edge_uid_list, proc_edge_uid_list, reg_edge_uid_list)


    get_multi_graph(root_path, root_path, file_edge_uid_list, net_edge_uid_list, proc_edge_uid_list, reg_edge_uid_list)

    #################################################################################################################################

    # Get all dictionaries
    get_dicts(root_path,
              file_edge_dict,file_node_dict,file_thread_dict, 
              proc_edge_dict, proc_node_dict, proc_thread_dict,
              reg_edge_dict, reg_node_dict, reg_thread_dict, 
              net_edge_dict, net_node_dict, net_thread_dict)
    
    
#if __name__ == "__main__":
#    main()

##TODO:
# 1. Generate all realtime malware data
# 2. Generate all offline malware data
# 3. to cross check the offline graphs with real time SG's