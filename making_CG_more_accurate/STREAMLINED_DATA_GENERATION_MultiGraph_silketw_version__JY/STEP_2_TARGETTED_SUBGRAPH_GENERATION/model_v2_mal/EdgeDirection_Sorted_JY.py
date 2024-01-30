import os
import profile
#from memory_profiler import profile
#from model_v1 import Realtime_FirstStep as fs
import networkx as nx
import math

# fp=open('memory_profiler.log','a+')
# @profile(stream=fp)

def set_edge_dir(graph_root, file_data, net_data, reg_data, proc_data): # 
    
    G = nx.read_graphml(os.path.join(graph_root,"Union.GraphML"))
    edgelist = list(G.edges(data = True))

    for src , tar , edge in edgelist: 
        
        # "SRC" - THREAD node 
        # "TAR" - Other nodes (F/R/N/P)
        
        edge_event_list= edge["name"].split(',')
        
        proc_SRC_TAR_sortdict={}    # Dictionary for Events with "edge-direction: SRC(PROC-NODE) ----> TAR(TRHEAD)" that will be sorted by TimeStamp.
        proc_TAR_SRC_sortdict={}    # Dictionary for Events with "edge-direction: TAR(TRHEAD) ----> SRC(PROC-NODE)" that will be sorted by TimeStamp.

        net_SRC_TAR_sortdict={}     # Dictionary for Events with "edge-direction: SRC(NET-NODE) ----> TAR(TRHEAD)" that will be sorted by TimeStamp.
        net_TAR_SRC_sortdict={}     # Dictionary for Events with "edge-direction: TAR(THREAD) ----> SRC(NET-NODE)" that will be sorted by TimeStamp.

        file_SRC_TAR_sortdict={}    # Dictionary for Events with "edge-direction: SRC(FILE-NODE) ----> TAR(TRHEAD)" that will be sorted by TimeStamp.
        file_TAR_SRC_sortdict={}    # Dictionary for Events with "edge-direction: TAR(THREAD) ----> SRC(FILE-NODE)" that will be sorted by TimeStamp.
        
        reg_SRC_TAR_sortdict={}     # Dictionary for Events with "edge-direction: SRC(REG-NODE) ----> TAR(TRHEAD)" that will be sorted by TimeStamp.
        reg_TAR_SRC_sortdict={}     # Dictionary for Events with "edge-direction: TAR(THREAD) ----> SRC(REG-NODE)" that will be sorted by TimeStamp.

        # THERE ARE TO VERIFY THE LENGTH OF LIST LATER
        new_arr1 =[]   
        new_arr2=[]
        new_arr3=[]
        new_arr4=[]


        for event in edge_event_list:
            
            proc_key = proc_data.get( event, {}).get("Task Name")
            proc_ts = proc_data.get( event, {}).get("TimeStamp")

            net_key = net_data.get( event, {}).get("Opcode")
            net_ts = net_data.get( event, {}).get("TimeStamp")

            file_key = file_data.get( event, {}).get("Task Name")
            file_ts = file_data.get( event, {}).get("TimeStamp")

            reg_key = reg_data.get( event, {}).get("Opcode")
            reg_ts = reg_data.get( event, {}).get("TimeStamp")            
            

            # IF PROCESS-EVENT -------------------------------------------------------------------------------------------------------
            if proc_key != None: 
                if G.has_edge(src,tar):
                    if (len(edge_event_list) > 1):


                        ## Original
                        #if proc_key in {'PROCESSSTART', 'PROCESSSTOP', 'JOBSTART', 'JOBSTOP', 
                        #                'CPUBASEPRIORITYCHANGE', 'CPUPRIORITYCHANGE', 'IOPRIORITYCHANGE', 'PAGEPRIORITYCHANGE'}:
                        #   # edge-direction: SRC(THREAD) ----> TAR(PROC-NODE)
                        #    proc_SRC_TAR_sortdict.update( {event:proc_ts})

                        # 'PROCESSSTART' should be 'T--->ChildP' NOT 'T---->ParentP'
                        # > For PROCESSSTOP, it appears 'PROCESS & THREAD' STOPS 'PROCESS', i.e. IT IS NOT THAT PARENT-PROCESS STOPS CHILD-PROCESS.  
                        #                    Thus, I think PROCESS-STOP can be just expressed as "T--->ParentP"
                
                        ## New

                    # Updated @ 2023-03-14, considering the edge-cases also found in 
                        #                        ["CPUPRIORITYCHANGE", "CPUBASEPRIORITYCHANGE", "IOPRIORITYCHANGE", "PAGEPRIORITYCHANGE"]
                        #                        +  ["IMAGELOAD", "IMAGEUNLOAD"]
                        # 
                            # elif logentry_TaskName.upper() in ["CPUPRIORITYCHANGE", "CPUBASEPRIORITYCHANGE", "IOPRIORITYCHANGE", "PAGEPRIORITYCHANGE"]:
                            #     # 3 cases: Non-Edge-Case, Case-1, Case-2
                            #     # Refer to : https://docs.google.com/presentation/d/1prPhAl_8P6VQ7NYXt2l1j3f7wsd3b84a9PNC2UngAXk/edit#slide=id.g21912b4d815_2_9 (slide 439)

                            #     # (Non-Edge-Case)
                            #     #
                            #     # > Log-entry Example
                            #     #   
                            #     #   ProcessId | ThreadId | ProcessID | ThreadID |  Task-Name  
                            #     #   ----------------------------------------------------------
                            #     #        4    |  68      |    4      |   68     |  CPUBASEPRIORITYCHNAGE 
                            #     # 
                            #     #   --->      ( Thread 68 (of Process 4) ) ---CPUBASEPRIORITYCHNAGE---> ( Process 4 )   
                            #     #   i.e. Process 4 using its Thread 68, is invoking Process 4's Thread 68 to change its Process's (Process 4) CPUBASE/CPU/IO/PAGE-PRIORITY. 
                        
                                  # (Case-1)
                                  #   ProcessId | ThreadId | ProcessID | ThreadID |  Task-Name  
                                  #   ----------------------------------------------------------
                                  #     12628   |  2052    |   12628   |   6608   |  CPUBASEPRIORITYCHNAGE 
                                  #   --->      ( Thread 6608 (of Process 12628) ) ---CPUBASEPRIORITYCHNAGE---> ( Process 12628 ) 
                                  #   i.e. Process 12628 using its Thread 2052, is invoking Process 12628's Thread 6608 to change its Process's (Process 12628) CPUBASE/CPU/IO/PAGE Priority. 
                                  # (Case-2)
                                  #   ProcessId | ThreadId | ProcessID | ThreadID |  Task-Name  
                                  #   ----------------------------------------------------------
                                  #     4136    |  2552     |   1476    |   3032   |  CPUBASEPRIORITYCHNAGE 
                                  #   --->      ( Thread 3032 (of Process 1476) ) ---CPUBASEPRIORITYCHNAGE---> ( Process 1476 ) 
                                  #   i.e. Process 4136 using its Thread 2552, is invoking Process 12628's Thread 3032 to change its Process's (Process 12628) CPUBASE/CPU/IO/PAGE Priority. 

                            # elif logentry_TaskName in ["IMAGELOAD", "IMAGEUNLOAD"]: 
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

                            # [ PROCESSSTOP ]
                            # > Edge Case 3
                            # --> PROCESSSTOP only has "Edge Case 3" (above example:  { 'ThreadId': 15312, 'ProcessId': 14900,  'ProcessID': '14900',  'ThreadID': None } )
                            #         #
                            #         #   ProcessId | ThreadId | ProcessID | ThreadID |  Task-Name  
                            #         #   ----------------------------------------------------------
                            #         #     14900   |  15312   |   14900   |   -      |  PROCESSSTOP
                            #         # 
                            #         #   --->      ( Thread 15312 (of Process 14900) ) ---PROCESSSTOP ---> ( Process 14900 )
                            #         #   i.e. Process 14900 using Thread 15312, stops itself (Process 14900)  


                        if proc_key in {'PROCESSSTART', 
                                        'PROCESSSTOP', 
                                        'JOBSTART', 
                                        'JOBSTOP', 
                                        'CPUBASEPRIORITYCHANGE', 'CPUPRIORITYCHANGE', 'IOPRIORITYCHANGE', 'PAGEPRIORITYCHANGE',
                                        'IMAGELOAD', 'IMAGEUNLOAD' # basically these 2 are newly added @ 2023-03-14 -- above were all already here.
                                        }:
                           # edge-direction: SRC(THREAD) ----> TAR(PROC-NODE)
                            proc_SRC_TAR_sortdict.update( {event:proc_ts})

                        else: # 'THREADSTART', 'THREADSTOP', etc.
                            # edge-direction: TAR(PROC-NODE) ----> SRC(THREAD)
                            proc_TAR_SRC_sortdict.update({event:proc_ts})
                            new_arr2.append(event)           
                    
                    if (len(edge_event_list) == 1):
                        
                        if proc_key in {'PROCESSSTART', 
                                        'PROCESSSTOP', 
                                        'JOBSTART', 
                                        'JOBSTOP', 
                                        'CPUBASEPRIORITYCHANGE', 'CPUPRIORITYCHANGE', 'IOPRIORITYCHANGE', 'PAGEPRIORITYCHANGE',
                                        'IMAGELOAD', 'IMAGEUNLOAD' # basically these 2 are newly added @ 2023-03-14 -- above were all already here.
                                        }:
                            # edge-direction: SRC(THREAD) ----> TAR(PROC-NODE)                        
                            G.remove_edge(src, tar)
                            G.add_edge(src, tar, name = str([event]))   
                        
                        else: # 'THREADSTART', 'THREADSTOP', etc.
                            # edge-direction: TAR(PROC-NODE) ----> SRC(THREAD)


                            G.remove_edge(src, tar)
                            G.add_edge(tar, src, name = str([event]))


            # IF FILE-EVENT -------------------------------------------------------------------------------------------------------
            if file_key != None: 
                if (len(edge_event_list) > 1): # more than one events on edge
                    
                    if file_key in {'READ', 'QUERYINFORMATION', 'QUERYSECURITY', 'QUERYEA', 'DIRENUM', 'DIRNOTIFY'}: 
                        # edge-direction: TAR(FILE-NODE) ----> SRC(THREAD)
                        file_TAR_SRC_sortdict.update({event:file_ts})
                        new_arr1.append(event) 
                    else: # e.g. WRITE, etc.
                        # edge-direction: SRC(THREAD) ----> TAR(FILE-NODE) 
                        file_SRC_TAR_sortdict.update({event:file_ts})           
                
                if (len(edge_event_list) == 1):
                    if file_key in {'READ', 'QUERYINFORMATION', 'QUERYSECURITY', 'QUERYEA', 'DIRENUM', 'DIRNOTIFY'}:
                        # edge-direction: TAR(FILE-NODE) ----> SRC(THREAD)
                        G.remove_edge(src,tar)
                        G.add_edge(tar,src,name = str([event]))
                    else: # e.g. WRITE, etc.
                        # edge-direction: SRC(THREAD) ----> TAR(FILE-NODE) 
                        G.remove_edge(src,tar)
                        G.add_edge(src,tar,name = str([event]))


            # IF NETWORK-EVENT -------------------------------------------------------------------------------------------------------
            if net_key!= None: 

                # QUESTION: What does Opcode 11, 15, 43, 13 exactly indicate?
                #           Based on my observation alot of Network-events with 'Task Name' also had Opcode == 42
                #           Would this (resulting edge-direction) relate to the reason why we didn't get Network nodes in subgraphs? 

                if (len(edge_event_list) > 1):

                    # https://app.box.com/file/847198342265
                    #  11 (DataReceived), 
                    #  15 (Connectionaccepted)
                    #  43 (DatareceivedoverUDPprotocol)
                    #  13 (Disconnectissued) 
                    if net_key in {11, 15, 43, 13}:
                        # edge-direction: TAR(NET-NODE) ----> SRC(THREAD) 
                        net_TAR_SRC_sortdict.update({event:net_ts})
                        new_arr3.append(event)
                    else:
                        # edge-direction: SRC(THREAD) ----> TAR(NET-NODE)
                        net_SRC_TAR_sortdict.update({event:net_ts})    
                
                if (len(edge_event_list) == 1):
                    if net_key in {11, 15, 43, 13}:
                        # edge-direction: TAR(NET-NODE) ----> SRC(THREAD) 
                        G.remove_edge(src,tar)
                        G.add_edge(tar,src,name = str([event]))
                    else:
                        # edge-direction: SRC(THREAD) ----> TAR(NET-NODE) 
                        G.remove_edge(src,tar)
                        G.add_edge(src,tar,name = str([event]))


            # IF REGISTRY-EVENT -------------------------------------------------------------------------------------------------------
            if reg_key != None:  
                if (len(edge_event_list) > 1): 
                    
                    # https://app.box.com/file/847198342265
                    # 35 (QueryKey)
                    # 38 (QueryValuekey)
                    # 39 (EnumerateKey)
                    # 40 (EnumerateValuekey
                    if reg_key in {35, 38, 39, 40}:
                        # edge-direction: TAR(REG-NODE) ----> SRC(THREAD)                    
                        reg_TAR_SRC_sortdict.update({event:reg_ts})
                        new_arr4.append(event)
                    else: # e.g. operation that corresponds to CREATE
                        # edge-direction: SRC(THREAD) ----> TAR(REG-NODE)                     
                        reg_SRC_TAR_sortdict.update({event:reg_ts})
                
                if (len(edge_event_list) == 1):
                    if reg_key in {35, 38, 39, 40}:
                        # edge-direction: TAR(REG-NODE) ----> SRC(THREAD)                    
                        G.remove_edge(src,tar)
                        G.add_edge(tar,src,name = str([event]))
                    else: # e.g. operation that corresponds to CREATE
                        # edge-direction: SRC(THREAD) ----> TAR(REG-NODE)                     
                        G.remove_edge(src,tar)
                        G.add_edge(src,tar,name = str([event]))


        ########################################################################################################################################


        # for Projection2 there is not need to sort F/R/N events, but for the sake of rest of the Projection algo's this code will be reusable
        if (len(edge_event_list) > 1):

            # IF PROCESS-EVENT -------------------------------------------------------------------------------------------------------
            if proc_key != None: 
                
                if proc_SRC_TAR_sortdict != {}:
                    res = sorted(proc_SRC_TAR_sortdict, key=lambda key: proc_SRC_TAR_sortdict[key])
                    G.add_edge(src,tar,name = str(res))
                
                if proc_TAR_SRC_sortdict != {}:
                    res = sorted(proc_TAR_SRC_sortdict, key=lambda key: proc_TAR_SRC_sortdict[key])
                    G.add_edge(tar,src,name = str(res))
                    if len(edge_event_list) == len(new_arr2):
                        G.remove_edge(src,tar)


            # IF FILE-EVENT -------------------------------------------------------------------------------------------------------
            if file_key != None: 
                
                if file_TAR_SRC_sortdict != {}:
                    res = sorted(file_TAR_SRC_sortdict, key=lambda key: file_TAR_SRC_sortdict[key]) # instead of lamda function, we can use reverse by making TS as key and name attr as values in dict 
                    G.add_edge(tar,src,name = str(res)) 
                    if len(edge_event_list) == len(new_arr1): # *Only satisfy when there is no event on file_src_tar_sort_dict
                        G.remove_edge(src,tar)
                
                if file_SRC_TAR_sortdict != {}:
                    res = sorted(file_SRC_TAR_sortdict, key=lambda key: file_SRC_TAR_sortdict[key])
                    G.add_edge(src,tar,name = str(res))


            # IF NETWORK-EVENT -------------------------------------------------------------------------------------------------------
            if net_key != None: 
                
                if net_TAR_SRC_sortdict != {}:
                    res = sorted(net_TAR_SRC_sortdict, key=lambda key: net_TAR_SRC_sortdict[key])
                    G.add_edge(tar,src,name = str(res))
                    if len(edge_event_list) == len(new_arr3):
                        G.remove_edge(src,tar)
                
                if net_SRC_TAR_sortdict != {}:
                    res = sorted(net_SRC_TAR_sortdict, key=lambda key: net_SRC_TAR_sortdict[key])
                    G.add_edge(src,tar,name =str(res))

             # IF REGISTRY-EVENT -------------------------------------------------------------------------------------------------------
            if reg_key != None:
                
                if reg_TAR_SRC_sortdict != {}:
                    res = sorted(reg_TAR_SRC_sortdict, key=lambda key: reg_TAR_SRC_sortdict[key])
                    G.add_edge(tar,src,name = str(res))
                    if len(edge_event_list) == len(new_arr4):
                        G.remove_edge(src,tar)
                
                if reg_SRC_TAR_sortdict != {}:
                    res = sorted(reg_SRC_TAR_sortdict, key=lambda key: reg_SRC_TAR_sortdict[key])
                    G.add_edge(src,tar,name =str(res))


    #f = open(os.path.join(graph_root,"net_edges.txt"),"w")
    #f.write(str(G.edges(data=True)) )
    #nx.draw(G, with_labels=True, edge_labels= True, font_weight='bold')
    #G.vs["taint"]=math.inf
    #nx.set_node_attributes(G, math.inf, "taint")
    nx.write_graphml(G, os.path.join(graph_root,'graph.GraphML'))
    del G
    #return G


# if __name__ == "__main__":
#     main()
