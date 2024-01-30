#import graphlib
import os
#import FirstStep as fs
import networkx as nx
import igraph
from igraph import *
import pickle
import json
import math
import re
import pandas as pd

import ast
from dateutil import parser
# # benign revised @ 2023-02-04
# def benign(root_path, idx, ids, edge_data, proc_node, proc_thread_node): 
#     i=0
#     backup = []
#     G= Graph.Read_GraphML(os.path.join(root_path,"graph.GraphML"))
    
#     # JY @ 2023-02-04: Sorted THE G.es by Timestamp order before iterating it to avoid unnecessary generations of subgraphs
#     #                  e.g. To avoid incidents where we get the "chrome.exe subgraph" that has root of child-process chrome.exe,
#     #                       before getting "chrome.exe subgraph" of root-process chrome.exe 
#     ProcessStart_edges = dict()
#     for edge in G.es:
#         if "PROC-EDGE" in edge['name']:
#             procedge_name = edge['name']
#             #for procedge_name in ast.literal_eval(edge['name']):
#             if 'processstart' in edge_data[procedge_name]['Task Name'].lower():
#                 ProcessStart_edges[edge] = edge_data[procedge_name]
#     # Now sort it to avoid generating subset-subgraphs before its superset subgraph.
#     # ProcessStart_edges = sorted(ProcessStart_edges.items(), key= lambda item: item[1]['TimeStamp'])
#     names_of_all_proc_nodes_from_all_subgraphs_generated_so_far = set() # set that stores ALL nodes from ALL subgraphs generated so far.

#     for edge_tuple in ProcessStart_edges:
#         edge = edge_tuple # edge-object;   edge_tuple[1] corresponds to edge_data[procedge_name] 
#         print(f"PROCESSSTART -- subgraph root: {G.vs[edge.target]['name']}", flush=True)
#         # name_list = ast.literal_eval(edge['name'])
#         name_list = edge['name']
#         # to save out with some information
#         ImageName = edge_data[name_list]["ImageName"]
#         if "\\" in  ImageName:
#             ImageName = ImageName.split('\\')[-1]
#         ProcessID = edge_data[name_list]["ProcessID"]
#         ProcessId = edge_data[name_list]["ProcessId"]
#         ThreadId = edge_data[name_list]["ThreadId"]
#         TimeStamp = edge_data[name_list]["TimeStamp"]

#         if edge.target not in ids:
#             # To avoid generating subset-subgraphs which are part of already generated superset subgraph. 
#             # if this root-process-node (of not yet generated subgraph) is already part of an already generated subgraph.
#             if G.vs[edge.target]['name'] not in names_of_all_proc_nodes_from_all_subgraphs_generated_so_far:

#                 ids.add(edge.target)
#                 G.vs["taint"]=math.inf
#                 #subgraph_path = os.path.join(root_path,"Benign_Sample_P3_"+idx+"_"+str(i))
#                 subgraph_path = os.path.join(root_path,f"Benign_Sample_P3_{idx}_{ImageName}_PID{ProcessID}_PId{ProcessId}_TId{ThreadId}_TS{TimeStamp}")
#                 if not os.path.exists(subgraph_path):
#                     print(f"made directory {subgraph_path}")
#                     os.makedirs(subgraph_path)
#                 ts = edge_data[name_list]["TimeStamp"] #Processstart root node timestamp
#                 G.vs[edge.target]["taint"] = ts          # root node taint time
#                 current_list=[]
#                 next_list=[edge.target]
#                 tainted_nodes={edge.target}
#                 sub_edge_set=set()
#                 edge_min_ts={} # mapping of edge with cutoff time
#                 tainted_subgraph(subgraph_path,G,current_list,next_list,tainted_nodes,edge_data,edge_min_ts,sub_edge_set,proc_node,proc_thread_node)
#                 #print("calling attributes")
#                 subgraph = attributes(root_path,subgraph_path)

#                 new_subgraph_all_proc_node_names = {v['name'] for v in subgraph.vs if "PROC-NODE" in v['name'] }
#                 names_of_all_proc_nodes_from_all_subgraphs_generated_so_far.update(new_subgraph_all_proc_node_names)
#             else:
#                 print(f"subgraph root: {G.vs[edge.target]['name']} is already part of another subgraph (superset).")
                
               

# def malware_org(root_path,idx,mal_PID,ids,edge_data,proc_node,proc_thread_node): 
#     G= Graph.Read_GraphML(os.path.join(root_path,"graph.GraphML"))
#     for a in G.es:
#         name_list= a["name"].replace("[","").replace("]","").replace(" ","").replace("\'","").split(',')
#         key1 = edge_data.get(name_list[0],{}).get("Task Name")
#         key2 = edge_data.get(name_list[0],{}).get("ImageName")
#         key3 = edge_data.get(name_list[0],{}).get("ProcessID")
#         if key2 != None:
#             k= re.search("malware.exe",key2)
#             if k != None:
#                 if (key1 == "PROCESSSTART") and (key3 == mal_PID):
#                     print(a.target)
#                     if a.target not in ids:
#                         ids.add(a.target)
#                         G.vs["taint"]=math.inf
#                         subgraph_path = os.path.join(root_path,"Malware_Sample_P3_"+idx)
#                         if not os.path.exists(subgraph_path):
#                             os.mkdir(subgraph_path)
#                         ts = edge_data.get(name_list[0],{}).get("TimeStamp") #Processstart root node timestamp
#                         #with open('out_mal1.txt', 'a') as f:
#                             #print("root node:",a.target,", taint time:", ts, file=f)
#                         G.vs[a.target]["taint"] = ts            # root node taint time
#                         current_list=[]
#                         next_list=[a.target]
#                         tainted_nodes={a.target}
#                         sub_edge_set=set()
#                         edge_min_ts={} # mapping of edge with cutoff time
#                         tainted_subgraph(subgraph_path,G,current_list,next_list,tainted_nodes,edge_data,edge_min_ts,sub_edge_set,proc_node,proc_thread_node)
#                         #print("calling attributes")
#                         attributes(root_path,subgraph_path)
#                     return

'''
JY @ 2024-1-3: Projection code's readability should be improved (e.g. more straightforward variable names, comments)
               because each time I come back, I again need to spend time, which decreases productivity,
               to understand the code due to lack of readability.
'''

# -----------------------------------------------------------------------------------------------------------------------------
def projection(root_path, idx, PID, edge_data):

    print(idx, flush= True)

    ids=set() # no root node should be common in subgraphs

    f1 = open(os.path.join(root_path,"proc_node.json")) 
    f2 = open(os.path.join(root_path,"proc_thread.json"))

    proc_nodes= json.load(f1)
    proc_thread_nodes = json.load(f2)   

    targetted_projection3(root_path,idx, PID, ids, edge_data, proc_nodes, proc_thread_nodes)

# -----------------------------------------------------------------------------------------------------------------------------
def targetted_projection3(root_path, 
                          idx, # elastic-serach idx 
                          Root_PID,  # Targetted PID
                          ids,  # node-ids 
                          edge_data, # edgeattr
                          proc_nodes, # json file loaded
                          proc_thread_nodes # json file loaded
                          ): 
    
    i = 0 
    G= Graph.Read_GraphML(os.path.join(root_path,"graph.GraphML"))

    for edge in G.es:

        name_list= edge["name"].replace("[","").replace("]","").replace(" ","").replace("\'","").split(',')
        TaskName = edge_data.get(name_list[0],{}).get("Task Name")
        ProcessID = edge_data.get(name_list[0],{}).get("ProcessID")

        if (TaskName == "ProcessStart/Start") and (ProcessID == Root_PID):

            print(f"processstart -- subgraph root: {G.vs[edge.target]['name']}", flush=True)

            # START FROM EDGE.SOURCE INSTEAD OF EDGE.TARGET
            
            if edge.target not in ids:

                ids.add(edge.target)
                
                G.vs["taint"]=math.inf
                
                subgraph_path = os.path.join(root_path,"SUBGRAPH_P3_"+idx)
                if not os.path.exists(subgraph_path):
                    print(f"made directory {subgraph_path}")
                    os.makedirs(subgraph_path)
                
                ts = edge_data.get(name_list[0],{}).get("TimeStamp") #Processstart root node timestamp

                G.vs[edge.target]["taint"] = ts            # root node taint time

                current_list=[] # to traverse

                next_list=[edge.target] # next to traverse

                tainted_nodes={edge.target}

                subgraph_edge_set=set()

                edge__min_ts__dict={} # mapping of edge with cutoff time

                tainted_subgraph(subgraph_path, G, current_list, next_list, tainted_nodes, edge_data, edge__min_ts__dict, subgraph_edge_set, proc_nodes, proc_thread_nodes)
                
                attributes(root_path,subgraph_path)
                i=i+1


# -----------------------------------------------------------------------------------------------------------------------------
def tainted_subgraph(subgraph_path, G1, current_list, next_list, tainted_nodes, 
                     
                     edge_data, edge__min_ts__dict, subgraph_edge_set, # 'edge_data', 'edge_min_ts', 'subgraph_edge_set' are only here to pass to nested functions
                     
                     proc_nodes, proc_thread_nodes):
        

        
        while( len(next_list) != 0 ):
            
            current_list = next_list # to traverse

            next_list = [] # next to traverse
            
            while( len(current_list) != 0):
                
                node= current_list.pop(0)

                if (G1.vs[node]["name"] in proc_nodes) or (G1.vs[node]["name"] in proc_thread_nodes):
                    
                    outgoing_edges = G1.incident(node, "out")  
                    
                    for outgoing_edge in outgoing_edges: # G1.es[e]['name']
                    
                        target_node = G1.es[outgoing_edge].target
                    
                        earliest_ts = earliest_time(G1, edge_data, node, outgoing_edge) # here, node is either proc or thread node
                        
                        if G1.vs[target_node]["taint"] > earliest_ts :      # compare target-node's taint time (could be infinity), 4
                                                                            # with earliest_ts__of_outgoing_edge (could be infinity?)
                            
                            G1.vs[target_node]["taint"] = earliest_ts       # assign taint-time
                            
                            next_list.append(target_node)
                            
                            tainted_nodes.add(target_node)  
        
        # Pass the populated 'tainted_nodes' to subgraph() function
        subgraph(subgraph_path, G1, tainted_nodes, 
                 edge_data, edge__min_ts__dict, subgraph_edge_set, 
                 proc_nodes, proc_thread_nodes)

# -----------------------------------------------------------------------------------------------------------------------------
def earliest_time(G2, edge_data, source_P_or_T_node, outgoing_edge):
    
    events_on_edge = G2.es[outgoing_edge]["name"].replace("[","").replace("]","").replace(" ","").replace("\'","").split(',')
    
    earliest_ts__of_outgoing_edge = math.inf # default infinity

    outgoing_edge__First_event__TaskName = edge_data.get(events_on_edge[0],{}).get("Task Name") # only first event on an edge gets checked using "n[0]"
    outgoing_edge__First_event__TimeStamp = edge_data.get(events_on_edge[0],{}).get("TimeStamp") 

    if G2.vs[source_P_or_T_node]["taint"] < outgoing_edge__First_event__TimeStamp:

        if (outgoing_edge__First_event__TaskName in ["ProcessStart/Start","ThreadStart/Start"]):

            earliest_ts__of_outgoing_edge = outgoing_edge__First_event__TimeStamp 

    return earliest_ts__of_outgoing_edge

# -----------------------------------------------------------------------------------------------------------------------------
def subgraph(subgraph_path, G, 
             tainted_nodes, 
             edge_data, 
             edge__min_ts__dict, 
             subgraph_edge_set, 
             proc_nodes, 
             proc_thread_nodes):
    

    while( len(tainted_nodes) != 0): # for all tainted nodes

        node=tainted_nodes.pop()

        # (JY @ 2024-1-3)
        #   ** if tainted node is process or thread node then check all its incoming and outgoing edges 
        #       -- Key of projection-3 
        #           -- For Process/Thread node: all incoming (F/R/N/T or F/R/N/P) + outgoing edges.
        #           -- Including the external node F/R/N (created by other benign process) -> P/T,
        #              only if the timestamp of the external node event is after the P/T node taint-time.
        
        if (G.vs[node]["name"] in proc_nodes) or (G.vs[node]["name"] in proc_thread_nodes):

            edges_all = G.incident(node,"all") # both incoming and outgoing edges w.r.t P/T node

            for edge in edges_all:

                src = G.es[edge].source
                tar = G.es[edge].target
                
                # compare taint-time of source-node and target-node of edge,
                # and set the edge's cutoff time, as the earlier(smaller) taint-time ; note that taint-time of nodes are by initiated as infinite 
                # (i.e. to later cutoff all events before this time of the edge)

                if G.vs[ src ]["taint"] < G.vs[ tar ]["taint"]:

                    edge__min_ts__dict[ edge ] = G.vs[ src ]["taint"]     #  earliest tainted time between src and tar
                                                                          #  edge__min_ts__dict ==  mapping of edge with cutoff time
                
                else:
                    edge__min_ts__dict[ edge ] = G.vs[ tar ]["taint"]
                
                subgraph_edge_set.add(edge)


    raw_subgraph = G.subgraph_edges(subgraph_edge_set)

    # pass the populated 'edge__min_ts__dict' and generated 'raw_subgraph' 
    exclude_events_based_on_cutoff(subgraph_path, 
                                   edge__min_ts__dict, 
                                   edge_data, 
                                   G, 
                                   raw_subgraph) 

# -----------------------------------------------------------------------------------------------------------------------------
def exclude_events_based_on_cutoff(subgraph_path, 
                                   edge__min_ts__dict, 
                                   edge_data, 
                                   G, 
                                   subgraph):
    
    # JY @ 2024-1-4: this function was written considering simple-graph setting (multiple events on an edge),
    #                which led to this concept of 'cutoff time' (cutoff all events on this edge that happened before the edge's cutoff time)
    #                but does not cause problem in multi-graph (event == edge) setting

    eid_list = []
    
    for edge in subgraph.es:

        source_id = subgraph.vs[edge.source]["id"]
        
        target_id = subgraph.vs[edge.target]["id"]
        
        edge_id = G.get_eid(int(source_id[1:]), int(target_id[1:]))        
        edge__cutoff_time = edge__min_ts__dict[edge_id] # get cut-off(min_ts) time of the edge

        names_list__after_cutting_off =[]
        
        names_list= edge["name"].replace("[","").replace("]","").replace(" ","").replace("\'","").split(',')
        
        for name in names_list: # each name corresponds to an event.
           
           # print("edge_id:", z,  ", cutoff time", cutoff_time)
            
            event_ts = edge_data.get(name,{}).get("TimeStamp")
            
            if event_ts >= edge__cutoff_time:
                names_list__after_cutting_off.append(name)


        if len(names_list__after_cutting_off) != 0:
            edge["name"] = str(names_list__after_cutting_off)

        else:
            # if all events on edge are cutted off (i.e. no events left on edge --> need to delete this edge)
            eid= subgraph.get_eid(edge.source, edge.target)
            eid_list.append(eid)
            

    subgraph.delete_edges(eid_list)

    subgraph.write_graphml(os.path.join(subgraph_path,"new_graph.graphml"))
    print(f"saved subgraph at {subgraph_path}", flush= True)



# -----------------------------------------------------------------------------------------------------------------------------
def attributes(root_path,subgraph_path):

    # This function is just to write out attributes

    dic_node = {}
    dic_edge = {}
    with (open(os.path.join(root_path,"global_node_attribute.pickle"), "rb")) as f1:
        object1= pickle.load(f1)  
    with (open(os.path.join(root_path,"global_edge_attribute.pickle"), "rb")) as f2:
        object2= pickle.load(f2) 

    g=Graph.Read_GraphML(os.path.join(subgraph_path,"new_graph.graphml"))
    #print(g.summary())
    
    for v in g.vs:
        dic_node[v["name"]]=object1.get(v["name"],{})
    node = open(os.path.join(subgraph_path,"node_attribute.pickle"), "wb")
    pickle.dump(dic_node,node)
    node.close()

    for a1 in g.es:
        sub_name_list= a1["name"].replace("[","").replace("]","").replace(" ","").replace("\'","").split(',')
        for p in sub_name_list:
            dic_edge[p]=object2.get(p,{})
    edge = open(os.path.join(subgraph_path,"edge_attribute.pickle"), "wb")
    pickle.dump(dic_edge,edge)
    edge.close()

    return g # added by JY @ 2023-02-04






# if __name__ == "__main__":
#     main()
