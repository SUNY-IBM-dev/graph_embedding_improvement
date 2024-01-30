#import graphlib
import os
#import model_v2.FirstStep as fs
import networkx as nx
import igraph
from igraph import *
import pickle
import json
import math
import re


def benign(projection_root,idx,backup,ids,edge_data,object1,object2,proc_node,proc_thread_node): 
    i=0
    G= Graph.Read_GraphML(os.path.join(projection_root,"graph.GraphML"))
    for a in G.es:
        name_list= a["name"].replace("[","").replace("]","").replace(" ","").replace("\'","").split(',')
        for x in name_list:
            if x not in backup:
                backup.append(x)
                key1 = edge_data.get(x,{}).get("Task Name") 
                if key1 == "PROCESSSTART":
                    print(i, a.target)
                    if a.target not in ids:
                        ids.add(a.target)
                        G.vs["taint"]=math.inf
                        root_path = os.path.join(projection_root,"Benign_Sample_PP2_"+idx+"_"+str(i))
                        if not os.path.exists(root_path):
                            os.mkdir(root_path)
                        ts = edge_data.get(x,{}).get("TimeStamp") #Processstart root node timestamp
                        #with open('out_B2.txt', 'a') as f:
                            #print("root node:",a.target,", taint time:", ts, file=f)
                        G.vs[a.target]["taint"] = ts            # root node taint time
                        current_list=[]
                        next_list=[a.target]
                        tainted_nodes={a.target}
                        sub_edge_set=set()
                        edge_min_ts={} # mapping of edge with cutoff time
                        taint_time(root_path,G,current_list,next_list,tainted_nodes,edge_data,edge_min_ts,sub_edge_set,proc_node,proc_thread_node)
                        #print("calling attributes")
                        attributes(root_path,object1,object2)
                        i=i+1    
                    break

def malware(root_path,idx,mal_PID,ids,edge_data,proc_node,proc_thread_node): 
    G= Graph.Read_GraphML(os.path.join(root_path,"graph.GraphML"))
    for a in G.es:
        name_list= a["name"].replace("[","").replace("]","").replace(" ","").replace("\'","").split(',')
        key1 = edge_data.get(name_list[0],{}).get("Task Name")
        key2 = edge_data.get(name_list[0],{}).get("ImageName")
        key3 = edge_data.get(name_list[0],{}).get("ProcessID")
        if key2 != None:
            k= re.search("malware.exe",key2)
            if k != None:
                if (key1 == "PROCESSSTART") and (int(key3) ==int(mal_PID)):
                    print(a.target)
                    if a.target not in ids:
                        ids.add(a.target)
                        G.vs["taint"]=math.inf
                        subgraph_path = os.path.join(root_path,"Malware_Sample_P2_"+idx)
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

def earliest_time(G2,edge_data,node,e1):
    n = G2.es[e1]["name"].replace("[","").replace("]","").replace(" ","").replace("\'","").split(',')
    min_ts=math.inf
    event1 =''
    tn = edge_data.get(n[0],{}).get("Task Name") #only first event on an edge gets checked using "n[0]"
    ts = edge_data.get(n[0],{}).get("TimeStamp")
    if G2.vs[node]["taint"] < ts:
        if(tn == "PROCESSSTART" or tn == "THREADSTART"):
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
                for e in edges:
                    x = G1.es[e].target
                    ts, event =earliest_time(G1,edge_data,node,e)
                    if (event == "PROCESSSTART" or event == "THREADSTART") and G1.vs[x]["taint"] > ts :      #if target node taint time is not infinity but it should be minimum
                        G1.vs[x]["taint"] = ts
                        next_list.append(x)
                        tainted_nodes.add(x)               
    subgraph(subgraph_path,G1,tainted_nodes,edge_data,edge_min_ts,sub_edge_set)

def subgraph(subgraph_path,g,tainted_nodes,edge_data,edge_min_ts,sub_edge_set):
    while(len(tainted_nodes)!=0):
        node=tainted_nodes.pop()
        edges = g.incident(node,"out")
        for e in edges:
            src = g.es[e].source
            tar = g.es[e].target
            if g.vs[src]["taint"] < g.vs[tar]["taint"]:
                edge_min_ts[e]=g.vs[src]["taint"]
            else:
                edge_min_ts[e]=g.vs[tar]["taint"]
            sub_edge_set.add(e)
    raw_subg = g.subgraph_edges(sub_edge_set)
    #subg.write_graphml(os.path.join(root_path,"graph.graphml")) 
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
    gph.write_graphml(os.path.join(subgraph_path,"new_graph.graphml"))
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

def projection(root_path,idx,mal_PID,edge_data):
    print(idx)
    #backup=[] 
    ids=set() # no root node should be common in subgraphs
    f1 = open(os.path.join(root_path,"proc_node.json")) 
    f2 = open(os.path.join(root_path,"proc_thread.json"))
    proc_node = json.load(f1)
    proc_thread_node = json.load(f2)   

    # if not idx2:
        # benign(root_path,idx,backup,ids,edge_data,object1,object2,proc_node,proc_thread_node)
    #else:
    malware(root_path,idx,mal_PID,ids,edge_data,proc_node,proc_thread_node)


if __name__ == "__main__":
    main()
