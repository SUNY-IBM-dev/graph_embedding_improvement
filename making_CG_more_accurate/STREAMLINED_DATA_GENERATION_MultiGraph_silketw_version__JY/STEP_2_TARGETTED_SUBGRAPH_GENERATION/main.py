
from datetime import datetime
import sys
import os
import json
import shutil
import time

sys.path.append("/data/d1/jgwak1/tabby/graph_embedding_improvement_JY_git/making_CG_more_accurate/STREAMLINED_DATA_GENERATION_MultiGraph_silketw_version__JY/STEP_2_TARGETTED_SUBGRAPH_GENERATION")
from model_v3_PW import FirstStep as fstep
from model_v3_PW import SecondStep as sstep
from model_v3_PW import EdgeDirection_Sorted_multigraph as ed
from model_v3_PW import Projection3_final as sub3

# from model_v3_JY_SimpleGraph import FirstStep_SimpleGraph as fstep
# from model_v3_JY_SimpleGraph import EdgeDirection_Sorted_SimpleGraph as ed
# from model_v3_JY_SimpleGraph import SecondStep as sstep
# from model_v3_JY_SimpleGraph import Projection3_final as sub3


def run_Targetted_Subgraph_Generation( ESIndex_ProcessID_dict, subgraphs_to_save_dirpath):

    starttime = datetime.now()


    for idx, ProcessID in ESIndex_ProcessID_dict.items():

        idx = idx.lower()
        this_iteration_start = datetime.now()

        print(f"{idx}"+"-"*80, flush=True)


        target_path = os.path.join(subgraphs_to_save_dirpath, idx)

        try:

            
            if os.path.exists(target_path):
                 shutil.rmtree(target_path)
            os.makedirs(target_path)
            # #malware_path = os.path.expanduser('~/tabby/SUNY_IBM_Project/data/'+idx)

            print("Igraph generation Phase", flush= True)

            # # Added by JY @ 2023-03-02: Added 3rd argument "EventTypes_to_Exclude_set"
            fstep.first_step(idx, target_path)


            f = open(os.path.join(target_path,"file_edge.json"))
            n = open(os.path.join(target_path,"net_edge.json"))
            r = open(os.path.join(target_path,"reg_edge.json"))
            p = open(os.path.join(target_path,"proc_edge.json")) 
            file_data = json.load(f)
            net_data = json.load(n)
            reg_data = json.load(r)
            proc_data = json.load(p)
            edge_data = {**file_data , **net_data , **reg_data, **proc_data}

            print("Edge direction phase", flush=True)
            ed.set_edge_dir(target_path,file_data,net_data,reg_data,proc_data)
            print("Edge direction phase DONE", flush=True)

            print("Attribute list generating phase", flush=True)
            sstep.second_step(target_path)
            print("Attribute list generating phase DONE", flush=True)

            #print("Projection-2 phase")
            #sub2.projection(malware_path,idx,mal_PID,edge_data)

            print("Projection-3 phase",flush=True)
            sub3.projection(target_path, idx, str(ProcessID), edge_data) # VERY IMPORTANT THAT IT IS STRINGIFIED!!
            print("Projection-3 phase DONE",flush=True)
       

        except Exception as e:
            print(f"Problem with index {idx} and ProcessID {ProcessID}, {e}", flush = True)
            time.sleep(120)
            if os.path.exists(target_path):
                shutil.rmtree(target_path)

        print(f"{idx} -- ELAPSED TIME: {str(datetime.now() - this_iteration_start)}")


    print("DONE", flush=True)
    endtime = datetime.now()
    elapsed_time = str( endtime - starttime )
    print(f"main-function elapsed time: {elapsed_time}", flush=True)

# if __name__ == "__main__":
    # main()
# 