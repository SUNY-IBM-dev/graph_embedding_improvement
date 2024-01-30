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
# benign revised @ 2023-02-04
def benign(root_path, idx, ids, edge_data, proc_node, proc_thread_node): 
    i=0
    backup = []
    G= Graph.Read_GraphML(os.path.join(root_path,"graph.GraphML"))
    
    # JY @ 2023-02-04: Sorted THE G.es by Timestamp order before iterating it to avoid unnecessary generations of subgraphs
    #                  e.g. To avoid incidents where we get the "chrome.exe subgraph" that has root of child-process chrome.exe,
    #                       before getting "chrome.exe subgraph" of root-process chrome.exe 
    ProcessStart_edges = dict()
    for edge in G.es:
        if "PROC-EDGE" in edge['name']:
            procedge_name = edge['name']
            #for procedge_name in ast.literal_eval(edge['name']):
            if 'processstart' in edge_data[procedge_name]['Task Name'].lower():
                ProcessStart_edges[edge] = edge_data[procedge_name]
    # Now sort it to avoid generating subset-subgraphs before its superset subgraph.
    # ProcessStart_edges = sorted(ProcessStart_edges.items(), key= lambda item: item[1]['TimeStamp'])
    names_of_all_proc_nodes_from_all_subgraphs_generated_so_far = set() # set that stores ALL nodes from ALL subgraphs generated so far.

    for edge_tuple in ProcessStart_edges:
        edge = edge_tuple # edge-object;   edge_tuple[1] corresponds to edge_data[procedge_name] 
        print(f"PROCESSSTART -- subgraph root: {G.vs[edge.target]['name']}", flush=True)
        # name_list = ast.literal_eval(edge['name'])
        name_list = edge['name']
        # to save out with some information
        ImageName = edge_data[name_list]["ImageName"]
        if "\\" in  ImageName:
            ImageName = ImageName.split('\\')[-1]
        ProcessID = edge_data[name_list]["ProcessID"]
        ProcessId = edge_data[name_list]["ProcessId"]
        ThreadId = edge_data[name_list]["ThreadId"]
        TimeStamp = edge_data[name_list]["TimeStamp"]

        if edge.target not in ids:
            # To avoid generating subset-subgraphs which are part of already generated superset subgraph. 
            # if this root-process-node (of not yet generated subgraph) is already part of an already generated subgraph.
            if G.vs[edge.target]['name'] not in names_of_all_proc_nodes_from_all_subgraphs_generated_so_far:

                ids.add(edge.target)
                G.vs["taint"]=math.inf
                #subgraph_path = os.path.join(root_path,"Benign_Sample_P3_"+idx+"_"+str(i))
                subgraph_path = os.path.join(root_path,f"Benign_Sample_P3_{idx}_{ImageName}_PID{ProcessID}_PId{ProcessId}_TId{ThreadId}_TS{TimeStamp}")
                if not os.path.exists(subgraph_path):
                    print(f"made directory {subgraph_path}")
                    os.makedirs(subgraph_path)
                ts = edge_data[name_list]["TimeStamp"] #Processstart root node timestamp
                G.vs[edge.target]["taint"] = ts          # root node taint time
                current_list=[]
                next_list=[edge.target]
                tainted_nodes={edge.target}
                sub_edge_set=set()
                edge_min_ts={} # mapping of edge with cutoff time
                tainted_subgraph(subgraph_path,G,current_list,next_list,tainted_nodes,edge_data,edge_min_ts,sub_edge_set,proc_node,proc_thread_node)
                #print("calling attributes")
                subgraph = attributes(root_path,subgraph_path)

                new_subgraph_all_proc_node_names = {v['name'] for v in subgraph.vs if "PROC-NODE" in v['name'] }
                names_of_all_proc_nodes_from_all_subgraphs_generated_so_far.update(new_subgraph_all_proc_node_names)
            else:
                print(f"subgraph root: {G.vs[edge.target]['name']} is already part of another subgraph (superset).")
                
               

def malware_org(root_path,idx,mal_PID,ids,edge_data,proc_node,proc_thread_node): 
    G= Graph.Read_GraphML(os.path.join(root_path,"graph.GraphML"))
    for a in G.es:
        name_list= a["name"].replace("[","").replace("]","").replace(" ","").replace("\'","").split(',')
        key1 = edge_data.get(name_list[0],{}).get("Task Name")
        key2 = edge_data.get(name_list[0],{}).get("ImageName")
        key3 = edge_data.get(name_list[0],{}).get("ProcessID")
        if key2 != None:
            k= re.search("malware.exe",key2)
            if k != None:
                if (key1 == "PROCESSSTART") and (key3 == mal_PID):
                    print(a.target)
                    if a.target not in ids:
                        ids.add(a.target)
                        G.vs["taint"]=math.inf
                        subgraph_path = os.path.join(root_path,"Malware_Sample_P3_"+idx)
                        if not os.path.exists(subgraph_path):
                            os.mkdir(subgraph_path)
                        ts = edge_data.get(name_list[0],{}).get("TimeStamp") #Processstart root node timestamp
                        #with open('out_mal1.txt', 'a') as f:
                            #print("root node:",a.target,", taint time:", ts, file=f)
                        G.vs[a.target]["taint"] = ts            # root node taint time
                        current_list=[]
                        next_list=[a.target]
                        tainted_nodes={a.target}
                        sub_edge_set=set()
                        edge_min_ts={} # mapping of edge with cutoff time
                        tainted_subgraph(subgraph_path,G,current_list,next_list,tainted_nodes,edge_data,edge_min_ts,sub_edge_set,proc_node,proc_thread_node)
                        #print("calling attributes")
                        attributes(root_path,subgraph_path)
                    return


def malware(root_path,idx,mal_PID,ids,edge_data,proc_node,proc_thread_node): 
    i = 0 
    G= Graph.Read_GraphML(os.path.join(root_path,"graph.GraphML"))
    for edge in G.es:
        name_list= edge["name"].replace("[","").replace("]","").replace(" ","").replace("\'","").split(',')
        key1 = edge_data.get(name_list[0],{}).get("Task Name")
        #key2 = edge_data.get(name_list[0],{}).get("ImageName")
        key3 = edge_data.get(name_list[0],{}).get("ProcessID")
        #if key3 != None:
        #    print("check here")

        # if key2 != None:
            # k= re.search("malware.exe",key2)
            # if k != None:
        #if (key1 == "PROCESSSTART") and (key3 != mal_PID): # JY @ 2023-1-5 : I believe this was for benign-noise
        if (key1 == "ProcessStart/Start") and (key3 == mal_PID):
            print(f"processstart -- subgraph root: {G.vs[edge.target]['name']}", flush=True)
            #print(f"user-activity PID: {mal_PID}")
            #print("")
            #print(f"G.vs[edge.target]['name']: {G.vs[edge.target]['name']}")
            #print(f"edge['name']: {edge['name']}")
            #print(f"G.vs[edge.target]['name']: {G.vs[edge.target]['name']}")
            #print("")
            #cmd_proc= edge['name'][ edge['name'].index("(PID):") + len( "(PID):") : edge['name'].index("_(TID):") ]
            #[ v['name'] for v in G.vs if v['name'].startswith('<<PROC-NODE>>') and cmd_proc in v['name']]
            
            # START FROM EDGE.SOURCE INSTEAD OF EDGE.TARGET
            if edge.target not in ids:
                ids.add(edge.target)
                G.vs["taint"]=math.inf
                subgraph_path = os.path.join(root_path,"SUBGRAPH_P3_"+idx)
                if not os.path.exists(subgraph_path):
                    print(f"made directory {subgraph_path}")
                    os.makedirs(subgraph_path)
                ts = edge_data.get(name_list[0],{}).get("TimeStamp") #Processstart root node timestamp
                #with open('out_mal1.txt', 'a') as f:
                    #print("root node:",a.target,", taint time:", ts, file=f)
                G.vs[edge.target]["taint"] = ts            # root node taint time
                current_list=[]
                next_list=[edge.target]
                tainted_nodes={edge.target}
                sub_edge_set=set()
                edge_min_ts={} # mapping of edge with cutoff time
                tainted_subgraph(subgraph_path,G,current_list,next_list,tainted_nodes,edge_data,edge_min_ts,sub_edge_set,proc_node,proc_thread_node)
                #print("calling attributes")
                attributes(root_path,subgraph_path)
                i=i+1

            '''
            if edge.target not in ids:
                ids.add(edge.target)
                G.vs["taint"]=math.inf
                subgraph_path = os.path.join(root_path,"SUBGRAPH_P3_"+idx)
                if not os.path.exists(subgraph_path):
                    os.mkdir(subgraph_path)
                ts = edge_data.get(name_list[0],{}).get("TimeStamp") #Processstart root node timestamp
                #with open('out_mal1.txt', 'a') as f:
                    #print("root node:",a.target,", taint time:", ts, file=f)
                G.vs[edge.target]["taint"] = ts            # root node taint time
                current_list=[]
                next_list=[edge.target]
                tainted_nodes={edge.target}
                sub_edge_set=set()
                edge_min_ts={} # mapping of edge with cutoff time
                tainted_subgraph(subgraph_path,G,current_list,next_list,tainted_nodes,edge_data,edge_min_ts,sub_edge_set,proc_node,proc_thread_node)
                #print("calling attributes")
                attributes(root_path,subgraph_path)
                i=i+1
            '''
            # return


def earliest_time(G2,edge_data,node,e1):
    n = G2.es[e1]["name"].replace("[","").replace("]","").replace(" ","").replace("\'","").split(',')
    min_ts=math.inf
    event1 =''
    tn = edge_data.get(n[0],{}).get("Task Name") #only first event on an edge gets checked using "n[0]"
    ts = edge_data.get(n[0],{}).get("TimeStamp")
    if G2.vs[node]["taint"] < ts:
        if(tn == "ProcessStart/Start" or tn == "ThreadStart/Start"):
            min_ts =ts 
            event1 = tn 
    return min_ts , event1

def tainted_subgraph(subgraph_path,G1,current_list,next_list,tainted_nodes,edge_data,edge_min_ts,sub_edge_set,proc_node,proc_thread_node):
    while(len(next_list)!=0):
        current_list=next_list
        next_list = []
        while(len(current_list)!=0):
            node= current_list.pop(0)

            if (G1.vs[node]["name"] in proc_node) or (G1.vs[node]["name"] in proc_thread_node):
                edges=G1.incident(node, "out")  
                
                # Added by JY @ 2023-1-13
                #outgoing_edges = [G1.es[e]['name'] for e in edges]
                for e in edges: # G1.es[e]['name']
                    x = G1.es[e].target
                    #print("")
                    #print(f"<**source-of-outgoing-edge>: {G1.vs[node]['name']}")
                    #print(f"<**outgoing-edge (single or multiple events)>: {G1.es[e]['name']}")
                    #target_from_outgoing_edge = [G1.vs[x]['name']]
                    #print(f"<**target-of-outgoing-edge>: {target_from_outgoing_edge}")



                    ts, event =earliest_time(G1, edge_data, node, e)
                    
                    if G1.vs[x]["taint"] > ts :      #if target node taint time is not infinity but it should be minimum
                        G1.vs[x]["taint"] = ts
                        next_list.append(x)
                        tainted_nodes.add(x)  

    subgraph(subgraph_path,G1,tainted_nodes,edge_data,edge_min_ts,sub_edge_set,proc_node,proc_thread_node)

def subgraph(subgraph_path,g,tainted_nodes,edge_data,edge_min_ts,sub_edge_set,proc_node,proc_thread_node):
    while(len(tainted_nodes)!=0):
        node=tainted_nodes.pop()
        # if tainted node is process or thread node then check all its incoming and outgoing edges
        if (g.vs[node]["name"] in proc_node) or (g.vs[node]["name"] in proc_thread_node):

            ###
            # Added by JY for debugging
            if g.vs[node]["name"]  == "<<THREAD>>(TID):3940_(TS):N/A__(PID):6056_(CT):N/A":
                print("found the thread-of-interest")
                pass

            ###

            edges_all = g.incident(node,"all")
            for e in edges_all:
                src = g.es[e].source
                tar = g.es[e].target
                if g.vs[src]["taint"] < g.vs[tar]["taint"]:
                    edge_min_ts[e]=g.vs[src]["taint"] # earliest tainted time between src and tar
                else:
                    edge_min_ts[e]=g.vs[tar]["taint"]
                sub_edge_set.add(e)
    raw_subg = g.subgraph_edges(sub_edge_set)
    exclude_events_based_on_cutoff(subgraph_path,edge_min_ts,edge_data,g,raw_subg) 

def exclude_events_based_on_cutoff(subgraph_path,edge_min_ts,edge_data,G,gph):
    eid_list = []
    #gph=Graph.Read_GraphML(os.path.join(root_path,"graph.graphml"))
    for a1 in gph.es:
        x = gph.vs[a1.source]["id"]
        y = gph.vs[a1.target]["id"]
        z=G.get_eid(int(x[1:]), int(y[1:]))
        l =[]
        t1 =[]
        t2= []
        sub_name_list= a1["name"].replace("[","").replace("]","").replace(" ","").replace("\'","").split(',')
        cutoff_time = edge_min_ts[z]
        for p in sub_name_list:
           # print("edge_id:", z,  ", cutoff time", cutoff_time)
            ts = edge_data.get(p,{}).get("TimeStamp")
            #tn = edge_data.get(p,{}).get("Task Name")
            
            if ts >= cutoff_time:
                #print("source:",x, ",target:",y, ", edge_id:" ,z, ", cutoff_time:", cutoff_time , ", Timestamp:", ts, ", event:", tn)
                l.append(p)
                #t1.append(ts)
                #t2.append(tn)

        if len(l) != 0:
            a1["name"] = str(l)
            #a1["Time"] = str(t1)
            #a1["Task"] = str(t2)
        else:
            eid= gph.get_eid(a1.source, a1.target)
            eid_list.append(eid)
            #print(eid)
            #g.delete_edges(eid)
    gph.delete_edges(eid_list)
    #print(f"start save subgraph at {subgraph_path}", flush= True)
    gph.write_graphml(os.path.join(subgraph_path,"new_graph.graphml"))
    print(f"saved subgraph at {subgraph_path}", flush= True)
    #print(gph.summary())

def attributes(root_path,subgraph_path):
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



def projection(root_path,idx,mal_PID, edge_data):
    print(idx, flush= True)
    #backup=[] 
    ids=set() # no root node should be common in subgraphs
    f1 = open(os.path.join(root_path,"proc_node.json")) 
    f2 = open(os.path.join(root_path,"proc_thread.json"))
    proc_node= json.load(f1)
    proc_thread_node = json.load(f2)   

    if not mal_PID:
        benign(root_path,idx,ids,edge_data,proc_node,proc_thread_node)
    else:
        malware(root_path,idx,mal_PID,ids,edge_data,proc_node,proc_thread_node)


if __name__ == "__main__":
    main()
