import FirstStep as fstep
import SecondStep as sstep
import EdgeDirection_Sorted as ed
#from model_v2 import Projection2_malware as sub1
import Projection2_final as sub2
import Projection3_final as sub3
from datetime import datetime
import sys
import os
import json


def main():
    starttime = datetime.now()
    idx  = sys.argv[1]
    mal_PID = sys.argv[2] 
    malware_path = os.path.expanduser('~/tabby/Projection3_datasets/Malware_data/'+idx)
    
    #malware_path = '/data/d1/pwakodi1/tabby/Projection2-3_data_sets/Prj_3/Malware_data/'+idx
    #benign_path = '~/tabby/Projection2_data_sets_new/Malware_data/'+idx
    #if not os.path.exists(malware_path):
    #    os.mkdir(malware_path)
    #malware_path = os.path.expanduser('~/tabby/SUNY_IBM_Project/data/'+idx)
    print("Igraph generation Phase")
    #fstep.first_step(idx,malware_path)
    f = open(os.path.join(malware_path,"file_edge.json"))
    n = open(os.path.join(malware_path,"net_edge.json"))
    r = open(os.path.join(malware_path,"reg_edge.json"))
    p = open(os.path.join(malware_path,"proc_edge.json")) 
    file_data = json.load(f)
    net_data = json.load(n)
    reg_data = json.load(r)
    proc_data = json.load(p)
    edge_data = {**file_data , **net_data , **reg_data, **proc_data}
    
    print("Edge direction phase")
    #ed.set_edge_dir(malware_path,file_data,net_data,reg_data,proc_data)

    print("Attribute list generating phase")
    sstep.second_step(malware_path)

    #print("Projection-2 phase")
    #sub2.projection(malware_path,idx,mal_PID,edge_data)

    #print("Projection-3 phase")
    #sub3.projection(malware_path,idx,mal_PID,edge_data)
    #sub3.projection3(benign_path,idx,idx2,edge_data)
    #endtime = datetime.now()
    #elapsed_time = str( endtime - starttime )
    #print(f"main-function elapsed time: {elapsed_time}")
main()
