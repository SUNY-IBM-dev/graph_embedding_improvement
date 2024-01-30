
import os
import pickle
import json
import pandas as pd
import igraph
import pprint
import datetime
import warnings
from pathlib import Path
warnings.filterwarnings("ignore")

''' TODO: Go thorugh "benign_node_type_dist_df__TODO_ANALYSIS_THIS.csv".  
         IT is weird that there is hardly any F/R/N nodes, 
         while there are some F-thread / R-thread / N-thread nodes'''


if __name__ == "__main__":

   #new_benign_subgraphs_dirpath = "/home/jgwak1/tabby/new_benign_log_prj3_subgraphs/Benign"
   Bengin_data_collection_dirpath = "/data/d1/jgwak1/tabby/GENERAL_LOG_COLLECTION_SUBGRAPHS_20230203/Benign"
   dirname = os.path.split(Bengin_data_collection_dirpath)[0]

   pp = pprint.PrettyPrinter(indent=3, width=10, sort_dicts=False)
   ##################################################################################################################################################
   benign_node_type_dist_df = pd.DataFrame()




   #Benign_SG_dirs = [ d for d in os.listdir(Bengin_data_collection_dirpath) if d.startswith("Benign_Sample_P3") ]
   Benign_SG_dirs = [ d for d in os.listdir(Bengin_data_collection_dirpath) if d.startswith("Benign_Sample_P3") ]


  
   for Benign_SG_dir in Benign_SG_dirs:  
        #if Benign_SG not in benign_test_SGs:
          #print(f"processing: {Benign_SG}")
          #benignSG_all_node_attributes = pickle.load( open( os.path.join(Benign_data_range_path, Benign_SG, "node_attribute.pickle"), "rb" ) ) 

          #benignSG_obj = igraph.Graph.Read_GraphML( os.path.join(Bengin_data_collection_dirpath, Benign_SG_dir, "new_graph.graphml") )
   
          #benignSG_obj = igraph.Graph.Read_GraphML( os.path.join(Bengin_data_collection_dirpath, Benign_SG_dir, f"SUBGRAPH_P3_{Benign_SG_dir}", "new_graph.graphml") )
          benignSG_obj = igraph.Graph.Read_GraphML( os.path.join(Bengin_data_collection_dirpath, Benign_SG_dir, "new_graph.graphml") )

          procnode_cnt = len( {v for v in benignSG_obj.vs if "PROC-NODE" in v['name']} )
          filenode_cnt = len( {v for v in benignSG_obj.vs if "FILE-NODE" in v['name']} ) 
          netnode_cnt = len( {v for v in benignSG_obj.vs if "NET-NODE" in v['name']} ) 
          regnode_cnt = len( {v for v in benignSG_obj.vs if "REG-NODE" in v['name']} )

          threadnode_cnt = len( {v for v in benignSG_obj.vs if "THREAD" in v['name']} )

          procedge_cnt = len({e['name'] for e in benignSG_obj.es if "PROC-EDGE" in e['name']})
          fileedge_cnt = len({e['name'] for e in benignSG_obj.es if "FILE-EDGE" in e['name']})
          netedge_cnt = len({e['name'] for e in benignSG_obj.es if "NET-EDGE" in e['name']})
          regedge_cnt = len({e['name'] for e in benignSG_obj.es if "REG-EDGE" in e['name']})


          counts = {"Benign_SG": Benign_SG_dir, 
                    "SG_node_cnt": benignSG_obj.vcount(), "SG_edge_cnt": benignSG_obj.ecount(),
                    "SG_event_cnt": len( sum( [e['name'].replace("[","").replace("]","").replace(" ","").replace("\'","").split(',')  for e in benignSG_obj.es ] , [] ) ),
                    "procnode_cnt":procnode_cnt, "filenode_cnt": filenode_cnt, "netnode_cnt": netnode_cnt, "regnode_cnt": regnode_cnt,
                    "threadnode_cnt": threadnode_cnt, 
                    "procedge_cnt":procedge_cnt, "fileedge_cnt": fileedge_cnt, "netedge_cnt": netedge_cnt, "regedge_cnt": regedge_cnt}
          

          print(f"{Benign_SG_dir}:\n"); pp.pprint(counts)
          print(f"-"*150)


          benign_node_type_dist_df = benign_node_type_dist_df.append(counts, ignore_index = True)

   benign_node_type_dist_df.to_csv( os.path.join( str(Path(__file__).parent),
                                                  f"{dirname}__NodeEdgeTypeDist_df_generated_{datetime.datetime.now().strftime('%Y%m%d_%H_%M_%S')}.csv") )


   print(benign_node_type_dist_df)

   print(f"{dirname}__NodeEdgeTypeDist_df.csv")

   pass
