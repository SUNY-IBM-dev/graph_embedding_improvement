import os
import profile
#from memory_profiler import profile
#from model_v1 import Realtime_FirstStep as fs
import networkx as nx
import math
from collections import Counter
# fp=open('memory_profiler.log','a+')
# @profile(stream=fp)

def set_edge_dir(graph_root, file_data, net_data, reg_data, proc_data): # 
    
    G = nx.read_graphml(os.path.join(graph_root,"Union.GraphML"))
    edgelist = list(G.edges(data = True, keys=True)) # JY @ 2023-05-21 : added "keys = True" for MultiDiGraph remove_edge is not enough with only src,tar anymore.
    
    # JY @ 2023-05-21 : added "keys = True" for MultiDiGraph remove_edge is not enough with only src,tar anymore.
    #                   because two nodes can have multiple edges (each representing an event), 
    #                   and to figure out the correct edge to remove, need not only src,tar, but also key
    # "key" (https://networkx.org/documentation/stable/reference/classes/generated/networkx.MultiDiGraph.remove_edge.html#networkx.MultiDiGraph.remove_edge)
    # hashable identifier, optional (default=None)
    # Used to distinguish multiple edges between a pair of nodes. If None, remove a single edge between u and v. If there are multiple edges, removes the last edge added in terms of insertion order

    # # for debugging
    # edgenames = [ e[3]['name'] for e in edgelist ]; 
    # count = Counter(edgenames) ; 
    # count.most_common()
    
    # edge_iter_cnt = 0
    for src , tar , key, edge in edgelist: 
        # edge_iter_cnt+= 1; 
        # print(f"edge_iter_cnt: {edge_iter_cnt} out of total {len(edgelist)}", flush=True)
        # "SRC" - THREAD node 
        # "TAR" - Other nodes (F/R/N/P)
        
        #edge_event_list= edge["name"].split(',')
        
        event = edge["name"]   # In multi-graph, edge == event

        #for event in edge_event_list:
            
        proc_key = proc_data.get( event, {}).get("Task Name")
        #proc_ts = proc_data.get( event, {}).get("TimeStamp")

        net_key = net_data.get( event, {}).get("Opcode")
        #net_ts = net_data.get( event, {}).get("TimeStamp")

        file_key = file_data.get( event, {}).get("Task Name")
        #file_ts = file_data.get( event, {}).get("TimeStamp")

        reg_key = reg_data.get( event, {}).get("Opcode")
        #reg_ts = reg_data.get( event, {}).get("TimeStamp")            
            

        # IF PROCESS-EVENT -------------------------------------------------------------------------------------------------------
        if proc_key != None: 

            # Added by JY @ 2023-05-20
            # type(G)
            # <class 'networkx.classes.multidigraph.MultiDiGraph'>
            # https://networkx.org/documentation/stable/reference/classes/generated/networkx.MultiDiGraph.has_edge.html#networkx.MultiDiGraph.has_edge


            #if G.has_edge(src,tar): # JY @ 2023-05-20: As now there are so many edges that share src and tar, we also need to match for event
                                     # however, MultiDigraph doesn't support "G.has_edge(src,tar, event) (X)".
                                     # so below is hack
            # edge_we_are_finding = [(u, v, k, d) for u, v, k, d in G.edges(keys=True, data=True) \
            #                         if (u == src) and (v == tar) and (d['name'] == event)]                        
            # if len(edge_we_are_finding) == 1:

            if proc_key in {'PROCESSSTART', 
                            'PROCESSSTOP', 
                            'JOBSTART', 
                            'JOBSTOP', 
                            'CPUBASEPRIORITYCHANGE', 'CPUPRIORITYCHANGE', 'IOPRIORITYCHANGE', 'PAGEPRIORITYCHANGE',
                            'IMAGELOAD', 'IMAGEUNLOAD' # basically these 2 are newly added @ 2023-03-14 -- above were all already here.
                            }:
                # edge-direction: SRC(THREAD) ----> TAR(PROC-NODE)
                # edge_to_remove = [(u, v, k, d) for u, v, k, d in G.edges(keys=True, data=True) \
                #                 if (u == src) and (v == tar) and (d['name'] == event)]                        
                # edge_to_remove = [(u, v, k, d) for u, v, k, d in G.edges(keys=True, data=True) if (d['name'] == event)]  # no need to also match (u==src) and (v==tar) due to efficiency                   
                # G.remove_edges_from(edge_to_remove)
                # #G.remove_edge(src, tar) # -- with MultiDiGraph , 'src' and 'tar' is not enough
                # # still works
                # G.add_edge(src, tar, name = event)

                pass    # JY @ 2023-05-20 : actually dont need to do anything

            else: # 'THREADSTART', 'THREADSTOP', etc.
                    # edge-direction: TAR(PROC-NODE) ----> SRC(THREAD)
                # In NetworkX's MultiDiGraph, there is no built-in function that directly removes an edge based on all three pieces of information: name, source, and target
                # edge_to_remove = [(u, v, k, d) for u, v, k, d in G.edges(keys=True, data=True) \
                #                 if (u == src) and (v == tar) and (d['name'] == event)]                        
                # edge_to_remove = [(u, v, k, d) for u, v, k, d in G.edges(keys=True, data=True) if (d['name'] == event)]  # no need to also match (u==src) and (v==tar) due to efficiency                   
                # G.remove_edges_from(edge_to_remove)
                G.remove_edge(src, tar, key = key) # -- with MultiDiGraph , 'src' and 'tar' is not enough
                
                G.add_edge(tar, src, name = event)


        # IF FILE-EVENT -------------------------------------------------------------------------------------------------------
        if file_key != None: 
   
            if file_key in {'READ', 'QUERYINFORMATION', 'QUERYSECURITY', 'QUERYEA', 'DIRENUM', 'DIRNOTIFY'}:
                # edge-direction: TAR(FILE-NODE) ----> SRC(THREAD)
                # edge_to_remove = [(u, v, k, d) for u, v, k, d in G.edges(keys=True, data=True) \
                #                    if (u == src) and (v == tar) and (d['name'] == event)]           
                # edge_to_remove = [(u, v, k, d) for u, v, k, d in G.edges(keys=True, data=True) if (d['name'] == event)]  # no need to also match (u==src) and (v==tar) due to efficiency                                 
                # G.remove_edges_from(edge_to_remove)
                #G.remove_edge(src, tar) # -- with MultiDiGraph , 'src' and 'tar' is not enough
                G.remove_edge(src, tar, key = key) # -- with MultiDiGraph , 'src' and 'tar' is not enough
                

                G.add_edge(tar,src,name = event)
            else: # e.g. WRITE, etc.
                # edge-direction: SRC(THREAD) ----> TAR(FILE-NODE) 
                
                
                # edge_to_remove = [(u, v, k, d) for u, v, k, d in G.edges(keys=True, data=True) \
                #                    if (u == src) and (v == tar) and (d['name'] == event)]        
                # edge_to_remove = [(u, v, k, d) for u, v, k, d in G.edges(keys=True, data=True) if (d['name'] == event)]  # no need to also match (u==src) and (v==tar) due to efficiency                                   
                # G.remove_edges_from(edge_to_remove)
                # #G.remove_edge(src, tar) # -- with MultiDiGraph , 'src' and 'tar' is not enough
                
                
                # G.add_edge(src,tar,name = event)

                pass    # JY @ 2023-05-20 : actually dont need to do anything


        # IF NETWORK-EVENT -------------------------------------------------------------------------------------------------------
        if net_key!= None: 

            # QUESTION: What does Opcode 11, 15, 43, 13 exactly indicate?
            #           Based on my observation alot of Network-events with 'Task Name' also had Opcode == 42
            #           Would this (resulting edge-direction) relate to the reason why we didn't get Network nodes in subgraphs? 

            if net_key in {11, 15, 43, 13}:
                # edge-direction: TAR(NET-NODE) ----> SRC(THREAD) 


                # edge_to_remove = [(u, v, k, d) for u, v, k, d in G.edges(keys=True, data=True) \
                #                    if (u == src) and (v == tar) and (d['name'] == event)]                        
                # edge_to_remove = [(u, v, k, d) for u, v, k, d in G.edges(keys=True, data=True) if (d['name'] == event)]  # no need to also match (u==src) and (v==tar) due to efficiency                   
                # G.remove_edges_from(edge_to_remove)
                #G.remove_edge(src, tar) # -- with MultiDiGraph , 'src' and 'tar' is not enough
                G.remove_edge(src, tar, key = key) # -- with MultiDiGraph , 'src' and 'tar' is not enough


                G.add_edge(tar,src,name = event)
            else:
                # edge-direction: SRC(THREAD) ----> TAR(NET-NODE) 

                # edge_to_remove = [(u, v, k, d) for u, v, k, d in G.edges(keys=True, data=True) \
                #                    if (u == src) and (v == tar) and (d['name'] == event)]                        
                # edge_to_remove = [(u, v, k, d) for u, v, k, d in G.edges(keys=True, data=True) if (d['name'] == event)]  # no need to also match (u==src) and (v==tar) due to efficiency                   
                # G.remove_edges_from(edge_to_remove)
                # #G.remove_edge(src, tar) # -- with MultiDiGraph , 'src' and 'tar' is not enough
                # G.add_edge(src,tar,name = event)

                pass    # JY @ 2023-05-20 : actually dont need to do anything


        # IF REGISTRY-EVENT -------------------------------------------------------------------------------------------------------
        if reg_key != None:  

            if reg_key in {35, 38, 39, 40}:
                # edge-direction: TAR(REG-NODE) ----> SRC(THREAD)                    

                # edge_to_remove = [(u, v, k, d) for u, v, k, d in G.edges(keys=True, data=True) \
                #                    if (u == src) and (v == tar) and (d['name'] == event)]                        
                # edge_to_remove = [(u, v, k, d) for u, v, k, d in G.edges(keys=True, data=True) if (d['name'] == event)]  # no need to also match (u==src) and (v==tar) due to efficiency                   
                # G.remove_edges_from(edge_to_remove)
                #G.remove_edge(src, tar) # -- with MultiDiGraph , 'src' and 'tar' is not enough
                G.remove_edge(src, tar, key = key) # -- with MultiDiGraph , 'src' and 'tar' is not enough


                G.add_edge(tar,src,name = event)
            else: # e.g. operation that corresponds to CREATE
                # edge-direction: SRC(THREAD) ----> TAR(REG-NODE)                     

                # edge_to_remove = [(u, v, k, d) for u, v, k, d in G.edges(keys=True, data=True) \
                #                    if (u == src) and (v == tar) and (d['name'] == event)] 
                # edge_to_remove = [(u, v, k, d) for u, v, k, d in G.edges(keys=True, data=True) if (d['name'] == event)]  # no need to also match (u==src) and (v==tar) due to efficiency                   
                # G.remove_edges_from(edge_to_remove)
                # #G.remove_edge(src, tar) # -- with MultiDiGraph , 'src' and 'tar' is not enough


                # G.add_edge(src,tar,name = event)

                pass    # JY @ 2023-05-20 : actually dont need to do anything



        ########################################################################################################################################

    
    # # for debugging
    # edgenames_after = [ e[3]['name'] for e in list(G.edges(data = True, keys=True)) ]
    # count_after = Counter(edgenames_after) ; 
    
    # count_after.most_common()
    # print("count_after.most_common()[0:20]:", flush= True)
    # print(*count_after.most_common()[0:20], sep="\n", flush= True)    

    nx.write_graphml(G, os.path.join(graph_root,'graph.GraphML'))
    del G
    #return G


# if __name__ == "__main__":
#     main()
