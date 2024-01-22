# %%
#/home/jgwak1/tabby/Baseline_Comparisons/Traditional_ML/Fair_Comparison/randomforest_baseline_faircompare.py
#/home/jgwak1/tabby/Baseline_Comparisons/Traditional_ML/Fair_Comparison/svm_baseline_faircompare.py

# IMPORTANT!! Read from Malware_data level instead of after Processed.
# "/home/jgwak1/tabby/Projection3_datasets/Malware_data/mal0/Malware_Sample_P3_mal0/edge_attribute.pickle"


                # 


# https://www.kaggle.com/code/stuarthallows/using-xgboost-with-scikit-learn

from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import preprocessing
import sklearn

from sklearn.pipeline import make_pipeline
import shap

import random
import pandas as pd
import pickle
import os
from collections import defaultdict
from operator import itemgetter
import pandas as pd
import operator
import re
import json
import glob
import datetime
import shutil
import shap


from nltk import ngrams


ALL_Task_Names = \
    ['CLEANUP', #1
    'CLOSE', #2
    'CREATE', #3
    'CREATENEWFILE', #4
    'DELETEPATH',#5
    'DIRENUM',#6
    'DIRNOTIFY',#7
    'FLUSH',#8
    'FSCTL',#9
    'NAMECREATE',#10
    'NAMEDELETE',#11
    'OPERATIONEND',#12
    'QUERYINFO',#13
    'QUERYINFORMATION',#14
    'QUERYEA',#15
    'QUERYSECURITY',#16
    'READ',#17
    'WRITE',#18
    'SETDELETE',#19
    'SETINFORMATION', #20
    'PAGEPRIORITYCHANGE',#21
    'IOPRIORITYCHANGE',#22
    'CPUBASEPRIORITYCHANGE',#23
    'IMAGEPRIORITYCHANGE',#24
    'CPUPRIORITYCHANGE',#25
    'IMAGELOAD',#26
    'IMAGEUNLOAD',#27
    'PROCESSSTOP',#28
    'PROCESSSTART',#29
    'PROCESSFREEZE',#30
    'PSDISKIOATTRIBUTE',#31
    'PSIORATECONTROL',#32 
    'THREADSTART',#33
    'THREADSTOP',#34
    'THREADWORKONBEHALFUPDATE', #35
    'JOBSTART',#36
    'JOBTERMINATE',#37
    'LOSTEVENT',#38
    'PSDISKIOATTRIBUTION',#39
    'RENAME',#40
    'RENAMEPATH',#41
    'THISGROUPOFEVENTSTRACKSTHEPERFORMANCEOFFLUSHINGHIVES',#42(index in final bit vector)

    "None",

    "UDPIP42_DatasentoverUDPprotocol",
    "UDPIP43_DatareceivedoverUDPprotocol",
    "UDPIP49_UDPconnectionattemptfailed",
    
    "TCPIP10_TCPIPDatasent",
    "TCPIP11_TCPIPDatareceived",
    "TCPIP12_TCPIPConnectionattempted",
    "TCPIP13_TCPIPDisconnectissued",
    "TCPIP14_TCPIPDataretransmitted",
    "TCPIP15_TCPIPConnectionaccepted",
    "TCPIP16_TCPIPReconnectattempted",
    "TCPIP17_TCPIPTCPconnectionattemptfailed",
    "TCPIP18_TCPIPProtocolcopieddataonbehalfofuser",

     "REGISTRY32_CreateKey",
     "REGISTRY33_OpenKey",
     "REGISTRY34_DeleteKey",
     "REGISTRY35_QueryKey",
     "REGISTRY36_SetValueKey",
     "REGISTRY37_DeleteValueKey",
     "REGISTRY38_QueryValueKey",
     "REGISTRY39_EnumerateKey",
     "REGISTRY40_EnumerateValueKey",
     "REGISTRY41_QueryMultipleValueKey",
     "REGISTRY42_SetInformationKey",
     "REGISTRY43_FlushKey",
     "REGISTRY44_CloseKey",
     "REGISTRY45_QuerySecurityKey",
     "REGISTRY46_SetSecurityKey",

     "ELSE"
]


import numpy as np
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

Task_Names = ['CLEANUP', #1
    'CLOSE', #2
    'CREATE', #3
    'CREATENEWFILE', #4
    'DELETEPATH',#5
    'DIRENUM',#6
    'DIRNOTIFY',#7
    'FLUSH',#8
    'FSCTL',#9
    'NAMECREATE',#10
    'NAMEDELETE',#11
    'OPERATIONEND',#12
    'QUERYINFO',#13
    'QUERYINFORMATION',#14
    'QUERYEA',#15
    'QUERYSECURITY',#16
    'READ',#17
    'WRITE',#18
    'SETDELETE',#19
    'SETINFORMATION', #20
    'PAGEPRIORITYCHANGE',#21
    'IOPRIORITYCHANGE',#22
    'CPUBASEPRIORITYCHANGE',#23
    'IMAGEPRIORITYCHANGE',#24
    'CPUPRIORITYCHANGE',#25
    'IMAGELOAD',#26
    'IMAGEUNLOAD',#27
    'PROCESSSTOP',#28
    'PROCESSSTART',#29
    'PROCESSFREEZE',#30
    'PSDISKIOATTRIBUTE',#31
    'PSIORATECONTROL',#32 
    'THREADSTART',#33
    'THREADSTOP',#34
    'THREADWORKONBEHALFUPDATE', #35
    'JOBSTART',#36
    'JOBTERMINATE',#37
    'LOSTEVENT',#38
    'PSDISKIOATTRIBUTION',#39
    'RENAME',#40
    'RENAMEPATH',#41
    'THISGROUPOFEVENTSTRACKSTHEPERFORMANCEOFFLUSHINGHIVES',#42(index in final bit vector)
    # all below 3 task are combined with opcode and having index 43 onwards for all of them in the function TN2int()
    # 'KERNEL_NETWORK_TASK_UDPIP'#index 43 # 42(opcode value) 43,49(https://github.com/repnz/etw-providers-docs/blob/master/Manifests-Win7-7600/Microsoft-Windows-Kernel-Network.xml)
    # 'KERNEL_NETWORK_TASK_TCPIP', # 10-18 (https://github.com/repnz/etw-providers-docs/blob/master/Manifests-Win7-7600/Microsoft-Windows-Kernel-Network.xml)
    # 'MICROSOFT-WINDOWS-KERNEL-REGISTRY', # 32- 46 (https://github.com/repnz/etw-providers-docs/blob/master/Manifests-Win7-7600/Microsoft-Windows-Kernel-Registry.xml)

]
#42+27=69


def TN2int_Revert( TaskNameOpcode_BitVector : list): # index0= none


    if TaskNameOpcode_BitVector.index(1) == 0:
        return "None" # 42

    elif TaskNameOpcode_BitVector.index(1) in range(43,45+1):
        # Important Note : Do not Separate by delimiter " - "
        #                  The countvectorizer fit-transform splits it.

        # https://github.com/repnz/etw-providers-docs/blob/master/Manifests-Win7-7600/Microsoft-Windows-Kernel-Network.xml

        # BitVecIndex_TNOP_map = \
        # {43: "KERNELNETWORKTASKUDPIP42_DatasentoverUDPprotocol",
        #  44: "KERNELNETWORKTASKUDPIP43_DatareceivedoverUDPprotocol",
        #  45: "KERNELNETWORKTASKUDPIP49_UDPconnectionattemptfailed"}
        # return BitVecIndex_TNOP_map[ TaskNameOpcode_BitVector.index(1) ]

        Simplified_BitVecIndex_TNOP_map = \
        {43: "UDPIP42_DatasentoverUDPprotocol",
         44: "UDPIP43_DatareceivedoverUDPprotocol",
         45: "UDPIP49_UDPconnectionattemptfailed"}
        return Simplified_BitVecIndex_TNOP_map[ TaskNameOpcode_BitVector.index(1) ]



    elif TaskNameOpcode_BitVector.index(1) in range(46,54+1):
        # Important Note : Do not Separate by delimiter " - " The countvectorizer fit-transform splits it.                  
        # BitVecIndex_TNOP_map = \
        # {46: "KERNELNETWORKTASKTCPIP10_TCPIPDatasent",
        #  47: "KERNELNETWORKTASKTCPIP11_TCPIPDatareceived",
        #  48: "KERNELNETWORKTASKTCPIP12_TCPIPConnectionattempted",
        #  49: "KERNELNETWORKTASKTCPIP13_TCPIPDisconnectissued",
        #  50: "KERNELNETWORKTASKTCPIP14_TCPIPDataretransmitted",
        #  51: "KERNELNETWORKTASKTCPIP15_TCPIPConnectionaccepted",
        #  52: "KERNELNETWORKTASKTCPIP16_TCPIPReconnectattempted",
        #  53: "KERNELNETWORKTASKTCPIP17_TCPIPTCPconnectionattemptfailed",
        #  54: "KERNELNETWORKTASKTCPIP18_TCPIPProtocolcopieddataonbehalfofuser"}
        # return BitVecIndex_TNOP_map[ TaskNameOpcode_BitVector.index(1) ]

        Simplified_BitVecIndex_TNOP_map = \
        {46: "TCPIP10_TCPIPDatasent",
         47: "TCPIP11_TCPIPDatareceived",
         48: "TCPIP12_TCPIPConnectionattempted",
         49: "TCPIP13_TCPIPDisconnectissued",
         50: "TCPIP14_TCPIPDataretransmitted",
         51: "TCPIP15_TCPIPConnectionaccepted",
         52: "TCPIP16_TCPIPReconnectattempted",
         53: "TCPIP17_TCPIPTCPconnectionattemptfailed",
         54: "TCPIP18_TCPIPProtocolcopieddataonbehalfofuser"}
        return Simplified_BitVecIndex_TNOP_map[ TaskNameOpcode_BitVector.index(1) ]




    elif TaskNameOpcode_BitVector.index(1) in range(55,69+1):
        # Important Note : Do not Separate by delimiter " - "
        #                  The countvectorizer fit-transform splits it. 
        # 
        # https://github.com/repnz/etw-providers-docs/blob/master/Manifests-Win10-18990/Microsoft-Windows-Kernel-Registry.xml
        # BitVecIndex_TNOP_map = \
        # # {55: "MICROSOFTWINDOWSKERNELREGISTRY32_CreateKey",
        # #  56: "MICROSOFTWINDOWSKERNELREGISTRY33_OpenKey",
        # #  57: "MICROSOFTWINDOWSKERNELREGISTRY34_DeleteKey",
        # #  58: "MICROSOFTWINDOWSKERNELREGISTRY35_QueryKey",
        # #  59: "MICROSOFTWINDOWSKERNELREGISTRY36_SetValueKey",
        # #  60: "MICROSOFTWINDOWSKERNELREGISTRY37_DeleteValueKey",
        # #  61: "MICROSOFTWINDOWSKERNELREGISTRY38_QueryValueKey",
        # #  62: "MICROSOFTWINDOWSKERNELREGISTRY39_EnumerateKey",
        # #  63: "MICROSOFTWINDOWSKERNELREGISTRY40_EnumerateValueKey",
        # #  64: "MICROSOFTWINDOWSKERNELREGISTRY41_QueryMultipleValueKey",
        # #  65: "MICROSOFTWINDOWSKERNELREGISTRY42_SetInformationKey",
        # #  66: "MICROSOFTWINDOWSKERNELREGISTRY43_FlushKey",
        # #  67: "MICROSOFTWINDOWSKERNELREGISTRY44_CloseKey",
        # #  68: "MICROSOFTWINDOWSKERNELREGISTRY45_QuerySecurityKey",
        # #  69: "MICROSOFTWINDOWSKERNELREGISTRY46_SetSecurityKey"}
        

        # For readability
        Simplified_BitVecIndex_TNOP_map = \
        {55: "REGISTRY32_CreateKey",
         56: "REGISTRY33_OpenKey",
         57: "REGISTRY34_DeleteKey",
         58: "REGISTRY35_QueryKey",
         59: "REGISTRY36_SetValueKey",
         60: "REGISTRY37_DeleteValueKey",
         61: "REGISTRY38_QueryValueKey",
         62: "REGISTRY39_EnumerateKey",
         63: "REGISTRY40_EnumerateValueKey",
         64: "REGISTRY41_QueryMultipleValueKey",
         65: "REGISTRY42_SetInformationKey",
         66: "REGISTRY43_FlushKey",
         67: "REGISTRY44_CloseKey",
         68: "REGISTRY45_QuerySecurityKey",
         69: "REGISTRY46_SetSecurityKey"}


        return Simplified_BitVecIndex_TNOP_map[ TaskNameOpcode_BitVector.index(1) ]
        # return BitVecIndex_TNOP_map[ TaskNameOpcode_BitVector.index(1) ]

    elif TaskNameOpcode_BitVector.index(1) == 70:
        return "ELSE"

    else:
        for index, value in enumerate(Task_Names, 1):
            if TaskNameOpcode_BitVector.index(1) == index:
                return value



# Added by JY @ 2023-02-28
Registry_EventTypes_to_Drop_set = {"microsoftwindowskernelregistry35_querykey", 
                                    "microsoftwindowskernelregistry33_openkey",
                                    "microsoftwindowskernelregistry38_queryvaluekey",
                                    "microsoftwindowskernelregistry44_closekey",
                                    "microsoftwindowskernelregistry42_setinformationkey",
                                    "microsoftwindowskernelregistry39_enumeratekey",
                                    "microsoftwindowskernelregistry40_enumeratevaluekey",
                                    "microsoftwindowskernelregistry32_createkey",
                                    "microsoftwindowskernelregistry36_setvaluekey",
                                    "microsoftwindowskernelregistry45_querysecuritykey",
                                    "microsoftwindowskernelregistry37_deletevaluekey",
                                    "microsoftwindowskernelregistry41_querymultiplevaluekey",
                                    "microsoftwindowskernelregistry34_deletekey",
                                    "microsoftwindowskernelregistry46_setsecuritykey",
                                    "microsoftwindowskernelregistry43_flushkey"}


# Extract the Malware-Train and Malware-Test indices based on their offline-pickles-directories
# i.e. Extract "malware-index" from "Processed_Malware_Sample_<malware-index>.pickle"
def Extract_SG_list( pickle_files_dirpath : str ):
    pickle_file_list = [file for file in os.listdir(pickle_files_dirpath) if ".pickle" in file]
    subgraph_list = [ pickle_file.removeprefix(f"Processed_").removesuffix(".pickle") for pickle_file in pickle_file_list ]
    return subgraph_list



####################################################################################################################
# Access Each Subgraph and Extract their "TaskName+Opcode" vector (sorted in Timestamp order).
def Extract_TaskNameOpcode_TSsorted_vectors_from_Subgraphs( Subgraphs_dirpath_list : list, 
                                                            target_subgraph_list : list,
                                                            EventTypes_to_Drop_set : set,
                                                                ):
    

    # Make sure all elements in the set are lower-cased
    EventTypes_to_Drop_set = { x.lower() for x in EventTypes_to_Drop_set }

    # Just to make it faster (asymptotically)
    if type(target_subgraph_list) == list:
         target_subgraph_list = set(target_subgraph_list) 

    
    SG_TaskNameOpcodeANDTimeStamp_listdict = defaultdict(list)    # { <Index> : [{"TaskNameOpcode": x, "TimeStamp": y},  <-- event-1
                                                                    #              {"TaskNameOpcode": x, "TimeStamp": y},  <-- event-2
                                                                    #                        ....                             ...
                                                                    #             ]}
    
    for Subgraphs_dirpath in Subgraphs_dirpath_list:    # e.g. Subgraphs_dirpath == tabby/NEW_Projection3_Datasets_Based_On_Modified_CG_Code__20230124/Malware


        subgraphs = [sg_dirname for sg_dirname in os.listdir(Subgraphs_dirpath) if "sample" in sg_dirname.lower() or "subgraph" in sg_dirname.lower() ]   

        for subgraph in subgraphs:  # e.g. subgraph == "Malware_Sample_mal0"
                                    #      subgraph == "Benign_Sample_P3_general_log_collection_60mins_started_20230205_20_02_31_formatted_Code.exe_PID10812_PId8468_TId14060_TS133201190295852283"
                            
            if subgraph in target_subgraph_list:

                print(f"{subgraph}",  flush = True)

                try:

                    # file-pointer to the edge_attribute.pickle file of the subgraph
                    edgeattr_pkl_fp = open( os.path.join(Subgraphs_dirpath, subgraph, "edge_attribute.pickle"), "rb" )
                    SG_edgeattr = pickle.load( edgeattr_pkl_fp )
                    
                    for event_uid in SG_edgeattr:
                        
                        event_TimeStamp = SG_edgeattr[event_uid]['TimeStamp']
                        event_TaskNameOpcode = TN2int_Revert( SG_edgeattr[event_uid]['Task Name'] ) # Revert Bit-vector into TaskNameOpcode


                        # Added by JY @ 2023-02-28 : Experiment with dropping some "Register Provider Event-types with high-frequency counts" 
                        
                        if event_TaskNameOpcode.lower() in EventTypes_to_Drop_set:
                            #print(f"Dropping Event-Type {event_TaskNameOpcode}\n", flush = True)
                            pass
                        else:
                            SG_TaskNameOpcodeANDTimeStamp_listdict[subgraph].append( {"TaskNameOpcode": event_TaskNameOpcode, "TimeStamp": event_TimeStamp} )
                except Exception as e:
                    raise RuntimeError(f"{subgraph}: {e}")
                
                # Sort events (TaskNameOpcode) in the list-dict by TimeStamp order. 
                # i.e. sort the following based on TimeStamp
                # {subgraph: [{"TNOP": eventX_TaskNameOpcode, "TimeStamp":400}, {"TNOP": eventY_TaskNameOpcode, "TimeStamp":20},
                #               ... {"TNOZ": eventY_TaskNameOpcode, "TimeStamp":880}]}
                SG_TaskNameOpcodeANDTimeStamp_listdict[subgraph] = sorted( SG_TaskNameOpcodeANDTimeStamp_listdict[subgraph], key=itemgetter('TimeStamp') )

    # Extract {subgraph: [firstevent_TaskNameOpcode,.. lastevent_TaskNameOpcode]} 
    #  from   {subgraph: [{"TNOP": firstevent_TaskNameOpcode, "TimeStamp":1},... {"TNOP": lastevent_TaskNameOpcode, "TimeStamp":100}]}
    SG_TaskNameOpcodes_dict = { subgraph: list( map(itemgetter('TaskNameOpcode'), sorted_TNOP_TS_dict_list))\
                                                for subgraph, sorted_TNOP_TS_dict_list in SG_TaskNameOpcodeANDTimeStamp_listdict.items() }

    return SG_TaskNameOpcodes_dict




if __name__ == "__main__":

   ###################################################################################################################
   # SET PARAMETERS
    #################################################################################################################
    N = 1
    if N != 1:
        ValueError("This file is for only N == 1")
   ###################################################################################################################
    strongest_X_features = 10    
    
    #EventTypes_to_Drop_set = Registry_EventTypes_to_Drop_set
    EventTypes_to_Drop_set = set()


   ###################################################################################################################
    # Directory to save all results
    print(f"RF-{N}gram-Events + 1gram-5bit-NodeTypes")
    experiment_identifier_dateinfo = datetime.datetime.now().strftime('%Y%m%d_%H-%M-%S')
    Results_save_dirpath = os.path.join("/data/d1/jgwak1/tabby/BASELINE_COMPARISONS/Sequential/RF+Ngrams",
                                        f"RESULTS__RF_{N}gram_events -- flatten_subgraph_psh__at_{experiment_identifier_dateinfo}")

    Malware_Offline_Subgraphs_dirpath_list = ["/data/d1/jgwak1/tabby/OFFLINE_TRAINTEST_DATA/cleaned_data_20230718_ver__basedon__all_malware_psh_samples_asof_20230705_MULTI_GRAPH_5BITplusADHOC_NODEATTR/Malware"]

    
    print(Malware_Offline_Subgraphs_dirpath_list,flush=True)

    #  *** option-1 for manually setting (explicitly split here) ***

    Malware_Train_SG_list = []
    Malware_Test_SG_list = []

    # # # *** option-2 for manually setting (initial way) ***
    # Malware_Offline_Train_pickles_dirpath = \
    #     "/data/d1/jgwak1/tabby/BASELINE_COMPARISONS/Sequential/RF+Ngrams/RESULTS__RF_1gram_flatten_subgraph_psh__at_20230612_21-47-21/GNN_TrainSet_same_as_RFexp/Processed_Malware_ONLY_TaskName_edgeattr"
    # Malware_Offline_Test_pickles_dirpath = \
    #     "/data/d1/jgwak1/tabby/BASELINE_COMPARISONS/Sequential/RF+Ngrams/RESULTS__RF_1gram_flatten_subgraph_psh__at_20230612_21-47-21/GNN_TestSet_same_as_RFexp/Processed_Malware_ONLY_TaskName_edgeattr"
    # Malware_Train_SG_list = Extract_SG_list( Malware_Offline_Train_pickles_dirpath )
    # Malware_Test_SG_list = Extract_SG_list( Malware_Offline_Test_pickles_dirpath )         

    # *** option-2 for manually setting (initial way) ***
    Malware_Offline_Train_pickles_dirpath = \
        "/data/d1/jgwak1/tabby/BASELINE_COMPARISONS/Sequential/RF+Ngrams/RESULTS__RF_1gram_flatten_subgraph_psh__at_20230716_14-10-30__model_seed_42__N=1/GNN_TrainSet_same_as_RFexp/Processed_Malware_ONLY_TaskName_edgeattr"
    Malware_Offline_Test_pickles_dirpath = \
        "/data/d1/jgwak1/tabby/BASELINE_COMPARISONS/Sequential/RF+Ngrams/RESULTS__RF_1gram_flatten_subgraph_psh__at_20230716_14-10-30__model_seed_42__N=1/GNN_TestSet_same_as_RFexp/Processed_Malware_ONLY_TaskName_edgeattr"
    Malware_Train_SG_list = Extract_SG_list( Malware_Offline_Train_pickles_dirpath )
    Malware_Test_SG_list = Extract_SG_list( Malware_Offline_Test_pickles_dirpath )         


    # STRATIFIED-SPLIT VS RANDOM-SPLIT
    MALWARE_STRATIFIED_SPLIT_FLAG = True
    MALWARE_PATTERNS_LIST_FOR_STRATIFIED_SPLIT = ["empire", "invoke_obfuscation", "nishang", "poshc2", "mafia",
                                                  "offsec", "powershellery", "psbits", "pt-toolkit", "randomps", "smallposh",
                                                  "bazaar"]
    train_ratio = 0.8
    test_ratio = 0.2
    
    #========================================================================================================================================
    
    if len(Malware_Train_SG_list) == 0 or len(Malware_Test_SG_list) == 0:
        # If NOT manually-splitted
        Malware_Train_SG_list = []
        Malware_Test_SG_list = []        


        # Get ALL Malware-Offline_SG_list
        ALL_Malware_Offline_SG_list = []    
        for Malware_Offline_Subgraphs_dirpath in Malware_Offline_Subgraphs_dirpath_list:         
            ALL_Malware_Offline_SG_list += [ x for x in os.listdir(Malware_Offline_Subgraphs_dirpath)\
                                            if ("sample" in x.lower() or "subgraph" in x.lower() ) ]

        if MALWARE_STRATIFIED_SPLIT_FLAG:

            from itertools import groupby

            def key_func(s):
                for pattern in MALWARE_PATTERNS_LIST_FOR_STRATIFIED_SPLIT:
                    if pattern in s:
                        return pattern
                return 'other'

            # split the list into groups based on the pattern
            groups = []
            for k, g in groupby(sorted(ALL_Malware_Offline_SG_list, key=key_func), key=key_func):
                groups.append(list(g))

            # stratified-split by group and add it to the 
            for group in groups:

                if len(group) > 1:

                    group_train, group_test = sklearn.model_selection.train_test_split( group, 
                                                                                        train_size = train_ratio, 
                                                                                        test_size = test_ratio )
                    # for each group, append.
                    Malware_Train_SG_list += group_train
                    Malware_Test_SG_list += group_test


                else:
                    # If only one sample in group, just add it to train-set.
                    Malware_Train_SG_list += group_train


        else:   # RANDOM-SPLIT
             
            Malware_Train_SG_list, Malware_Test_SG_list = sklearn.model_selection.train_test_split( ALL_Malware_Offline_SG_list ,
                                                                                                    train_size = train_ratio,
                                                                                                    test_size  = test_ratio )




    Malware_Train_SG_TaskNameOpcodes_dict = Extract_TaskNameOpcode_TSsorted_vectors_from_Subgraphs( 
                                                    Subgraphs_dirpath_list = Malware_Offline_Subgraphs_dirpath_list, 
                                                    target_subgraph_list = Malware_Train_SG_list,
                                                    EventTypes_to_Drop_set = EventTypes_to_Drop_set )

    Malware_Test_SG_TaskNameOpcodes_dict = Extract_TaskNameOpcode_TSsorted_vectors_from_Subgraphs( 
                                                    Subgraphs_dirpath_list = Malware_Offline_Subgraphs_dirpath_list, 
                                                    target_subgraph_list = Malware_Test_SG_list,
                                                    EventTypes_to_Drop_set = EventTypes_to_Drop_set )


    ##############################################################################################################################
    split_random_seed  = 0

    Benign_Offline_Subgraphs_dirpath_list = ["/data/d1/jgwak1/tabby/OFFLINE_TRAINTEST_DATA/cleaned_data_20230718_ver__basedon__all_benign_psh_samples_asof_20230705_MULTI_GRAPH_5BITplusADHOC_NODEATTR/Benign"]


    #  *** option-1 for manually setting (explicitly split here) ***
    Benign_Train_SG_list = []
    Benign_Test_SG_list = []


    # # *** option-2 for manually setting (initial way) ***
    Benign_Offline_Train_pickles_dirpath = \
        "/data/d1/jgwak1/tabby/BASELINE_COMPARISONS/Sequential/RF+Ngrams/RESULTS__RF_1gram_flatten_subgraph_psh__at_20230716_14-10-30__model_seed_42__N=1/GNN_TrainSet_same_as_RFexp/Processed_Benign_ONLY_TaskName_edgeattr"
    Benign_Offline_Test_pickles_dirpath = \
        "/data/d1/jgwak1/tabby/BASELINE_COMPARISONS/Sequential/RF+Ngrams/RESULTS__RF_1gram_flatten_subgraph_psh__at_20230716_14-10-30__model_seed_42__N=1/GNN_TestSet_same_as_RFexp/Processed_Benign_ONLY_TaskName_edgeattr"
    Benign_Train_SG_list = Extract_SG_list( Benign_Offline_Train_pickles_dirpath )
    Benign_Test_SG_list = Extract_SG_list( Benign_Offline_Test_pickles_dirpath )         


    # STRATIFIED-SPLIT VS RANDOM-SPLIT
    BENIGN_STRATIFIED_SPLIT_FLAG = True
    BENIGN_PATTERNS_LIST_FOR_STRATIFIED_SPLIT = ["fleschutz", "jhochwald", "devblackops", "farag2", "jimbrig", "jrussellfreelance", "nickrod518", 
                                                 "redttr", "sysadmin-survival-kit", "stevencohn", "ledrago"]
    train_ratio = 0.8
    test_ratio = 0.2
    
    #========================================================================================================================================
    
    if len(Benign_Train_SG_list) == 0 or len(Benign_Test_SG_list) == 0:
        # If NOT manually-splitted
        Benign_Train_SG_list = []
        Benign_Test_SG_list = []

        # Get ALL Benign-Offline_SG_list
        ALL_Benign_Offline_SG_list = []    
        for Benign_Offline_Subgraphs_dirpath in Benign_Offline_Subgraphs_dirpath_list:         
            ALL_Benign_Offline_SG_list += [ x for x in os.listdir(Benign_Offline_Subgraphs_dirpath)\
                                            if ("sample" in x.lower() or "subgraph" in x.lower() ) ]

        if BENIGN_STRATIFIED_SPLIT_FLAG:

            from itertools import groupby

            def key_func(s):
                for pattern in BENIGN_PATTERNS_LIST_FOR_STRATIFIED_SPLIT:
                    if pattern in s:
                        return pattern
                return 'other'

            # split the list into groups based on the pattern
            groups = []
            for k, g in groupby(sorted(ALL_Benign_Offline_SG_list, key=key_func), key=key_func):
                groups.append(list(g))

            # stratified-split by group and add it to the 
            for group in groups:

                if len(group) > 1:

                    group_train, group_test = sklearn.model_selection.train_test_split( group, 
                                                                                        train_size = train_ratio, 
                                                                                        test_size = test_ratio )
                    # for each group, append.
                    Benign_Train_SG_list += group_train
                    Benign_Test_SG_list += group_test

                else:
                    # If only one sample in group, just add it to train-set
                    Benign_Train_SG_list += group_train


        else:   # RANDOM-SPLIT
             
            Benign_Train_SG_list, Benign_Test_SG_list = sklearn.model_selection.train_test_split( ALL_Benign_Offline_SG_list ,
                                                                                                    train_size = train_ratio,
                                                                                                    test_size  = test_ratio )


    ##################################################################################################################################################################

    Benign_Train_SG_TaskNameOpcodes_dict = Extract_TaskNameOpcode_TSsorted_vectors_from_Subgraphs( 
                                                        Subgraphs_dirpath_list = Benign_Offline_Subgraphs_dirpath_list, 
                                                        target_subgraph_list = Benign_Train_SG_list,
                                                        EventTypes_to_Drop_set = EventTypes_to_Drop_set )

    Benign_Test_SG_TaskNameOpcodes_dict = Extract_TaskNameOpcode_TSsorted_vectors_from_Subgraphs( 
                                                        Subgraphs_dirpath_list = Benign_Offline_Subgraphs_dirpath_list, 
                                                        target_subgraph_list = Benign_Test_SG_list,
                                                        EventTypes_to_Drop_set = EventTypes_to_Drop_set )

    #--------------------------------------------------------------------------------------------------------------------------------------------
    ##############################################################################################################################

    # Now make the directory.
    if not os.path.exists(Results_save_dirpath):
        os.makedirs(Results_save_dirpath)   


    ##############################################################################################################################
    # save these out as well

    with open( os.path.join(Results_save_dirpath, 'Benign_Train_SG_list.txt'), 'w') as file:
        for item in Benign_Train_SG_list:
            file.write(item + '\n')

    with open( os.path.join(Results_save_dirpath, 'Benign_Test_SG_list.txt'), 'w') as file:
        for item in Benign_Test_SG_list:
            file.write(item + '\n')

    with open( os.path.join(Results_save_dirpath, 'Malware_Train_SG_list.txt'), 'w') as file:
        for item in Malware_Train_SG_list:
            file.write(item + '\n')

    with open( os.path.join(Results_save_dirpath, 'Malware_Test_SG_list.txt'), 'w') as file:
        for item in Malware_Test_SG_list:
            file.write(item + '\n')


    # Prepare Directories -------------------

    # Benign-Train
    Train_Processed_Benign_ONLY_TaskName_edgeattr_dirpath = \
        os.path.join(Results_save_dirpath, "GNN_TrainSet_same_as_RFexp", "Processed_Benign_ONLY_TaskName_edgeattr")
    if not os.path.exists(Train_Processed_Benign_ONLY_TaskName_edgeattr_dirpath):
         os.makedirs(Train_Processed_Benign_ONLY_TaskName_edgeattr_dirpath)
    benign_train_pickle_record_fp = open( os.path.join(Train_Processed_Benign_ONLY_TaskName_edgeattr_dirpath,'picklefiles-origin-record.txt'), 'w')
    # Benign-est
    Test_Processed_Benign_ONLY_TaskName_edgeattr_dirpath = \
        os.path.join(Results_save_dirpath, "GNN_TestSet_same_as_RFexp", "Processed_Benign_ONLY_TaskName_edgeattr")
    if not os.path.exists(Test_Processed_Benign_ONLY_TaskName_edgeattr_dirpath):
         os.makedirs(Test_Processed_Benign_ONLY_TaskName_edgeattr_dirpath)
    benign_test_pickle_record_fp = open( os.path.join(Test_Processed_Benign_ONLY_TaskName_edgeattr_dirpath,'picklefiles-origin-record.txt'), 'w')

    # Malwre-Train
    Train_Processed_Malware_ONLY_TaskName_edgeattr_dirpath = \
        os.path.join(Results_save_dirpath, "GNN_TrainSet_same_as_RFexp", "Processed_Malware_ONLY_TaskName_edgeattr")
    if not os.path.exists(Train_Processed_Malware_ONLY_TaskName_edgeattr_dirpath):
         os.makedirs(Train_Processed_Malware_ONLY_TaskName_edgeattr_dirpath)
    malware_train_pickle_record_fp = open( os.path.join(Train_Processed_Malware_ONLY_TaskName_edgeattr_dirpath,'picklefiles-origin-record.txt'), 'w')

    # Malwre-Test
    Test_Processed_Malware_ONLY_TaskName_edgeattr_dirpath = \
        os.path.join(Results_save_dirpath, "GNN_TestSet_same_as_RFexp", "Processed_Malware_ONLY_TaskName_edgeattr")
    if not os.path.exists(Test_Processed_Malware_ONLY_TaskName_edgeattr_dirpath):
         os.makedirs(Test_Processed_Malware_ONLY_TaskName_edgeattr_dirpath)
    malware_test_pickle_record_fp = open( os.path.join(Test_Processed_Malware_ONLY_TaskName_edgeattr_dirpath,'picklefiles-origin-record.txt'), 'w')



    # Following is for "matching" in the for-loop. set-type for constant time lookup.
    Benign_Train_Processed_Pickle_SG_set = set([f"Processed_{x}.pickle" for x in Benign_Train_SG_list])
    Benign_Test_Processed_pickle_SG_set = set([f"Processed_{x}.pickle" for x in Benign_Test_SG_list]) 
    Malware_Train_Processed_pickle_SG_set = set([f"Processed_{x}.pickle" for x in Malware_Train_SG_list]) 
    Malware_Test_Processed_pickle_SG_set = set([f"Processed_{x}.pickle" for x in Malware_Test_SG_list])  

    # Iterate through Benign_Offline_Subgraphs_dirpath_list (1 level above)
    for Benign_Offline_Subgraphs_dirpath in Benign_Offline_Subgraphs_dirpath_list:

        Benign_Offline_PickleFiles_dirpath = \
            os.path.join(os.path.split(Benign_Offline_Subgraphs_dirpath)[0], "Processed_Benign_ONLY_TaskName_edgeattr")

        for Benign_Offline_PickleFile in os.listdir(Benign_Offline_PickleFiles_dirpath):
             
             if Benign_Offline_PickleFile in Benign_Train_Processed_Pickle_SG_set:
                source = os.path.join(Benign_Offline_PickleFiles_dirpath, Benign_Offline_PickleFile)
                destination = os.path.join(Train_Processed_Benign_ONLY_TaskName_edgeattr_dirpath, Benign_Offline_PickleFile)
                shutil.copy(src = source,
                            dst = destination)
                print(f"copied {source} to {destination}\n", flush= True)
                print(f"copied\n{source}\nto\n{destination}\n", flush= True, file = benign_train_pickle_record_fp)

             elif Benign_Offline_PickleFile in Benign_Test_Processed_pickle_SG_set:
                source = os.path.join(Benign_Offline_PickleFiles_dirpath, Benign_Offline_PickleFile)
                destination = os.path.join(Test_Processed_Benign_ONLY_TaskName_edgeattr_dirpath, Benign_Offline_PickleFile)
                shutil.copy(src = source,
                            dst = destination)
                print(f"copied {source} to {destination}\n", flush= True)
                print(f"copied\n{source}\nto\n{destination}\n", flush= True, file = benign_test_pickle_record_fp)




    for Malware_Offline_Subgraphs_dirpath in Malware_Offline_Subgraphs_dirpath_list:
                
        Malware_Offline_PickleFiles_dirpath = \
            os.path.join(os.path.split(Malware_Offline_Subgraphs_dirpath)[0], "Processed_Malware_ONLY_TaskName_edgeattr")

        for Malware_Offline_PickleFile in os.listdir(Malware_Offline_PickleFiles_dirpath):
             
             if Malware_Offline_PickleFile in Malware_Train_Processed_pickle_SG_set:
                source = os.path.join(Malware_Offline_PickleFiles_dirpath, Malware_Offline_PickleFile)
                destination = os.path.join(Train_Processed_Malware_ONLY_TaskName_edgeattr_dirpath, Malware_Offline_PickleFile)
                shutil.copy(src = source,
                            dst = destination)
                print(f"copied {source} to {destination}\n", flush= True)
                print(f"copied\n{source}\nto\n{destination}\n", flush= True, file = malware_train_pickle_record_fp)

             elif Malware_Offline_PickleFile in Malware_Test_Processed_pickle_SG_set:
                source = os.path.join(Malware_Offline_PickleFiles_dirpath, Malware_Offline_PickleFile)
                destination = os.path.join(Test_Processed_Malware_ONLY_TaskName_edgeattr_dirpath, Malware_Offline_PickleFile)
                shutil.copy(src = source,
                            dst = destination)
                print(f"copied {source} to {destination}\n", flush= True)
                print(f"copied\n{source}\nto\n{destination}\n", flush= True, file = malware_test_pickle_record_fp)



    benign_train_pickle_record_fp.close()
    benign_test_pickle_record_fp.close()
    malware_train_pickle_record_fp.close()
    malware_test_pickle_record_fp.close()
    ##############################################################################################################################
    ####################################################################################################################
    # Get 'N-gram frequencies' feature
    
        # CountVectorizer params
        # <max_feautres>  : https://stackoverflow.com/questions/61274499/reduce-dimension-of-word-vectors-from-tfidfvectorizer-countvectorizer
        # If max_features is set to None, then the whole corpus is considered during the TF-IDF transformation. 
        # Otherwise, if you pass, say, 5 to max_features, that would mean creating a feature matrix out of the most 5 frequent words accross text documents.   
        
        # <max_df> & <min_df>
        # max_df is used for removing terms that appear too frequently, also known as "corpus-specific stop words". For example:

        # max_df = 0.50 means "ignore terms that appear in more than 50% of the documents".
        # max_df = 25 means "ignore terms that appear in more than 25 documents".
        # The default max_df is 1.0, which means "ignore terms that appear in more than 100% of the documents". Thus, the default setting does not ignore any terms.

        # min_df is used for removing terms that appear too infrequently. For example:

        # min_df = 0.01 means "ignore terms that appear in less than 1% of the documents".
        # min_df = 5 means "ignore terms that appear in less than 5 documents".
        # The default min_df is 1, which means "ignore terms that appear in less than 1 document". Thus, the default setting does not ignore any terms.
        # https://stackoverflow.com/questions/27697766/understanding-min-df-and-max-df-in-scikit-countvectorizer   
        
        # https://www.reddit.com/r/learnmachinelearning/comments/6evguc/while_building_a_tfidf_determining_a_good_balance/
        # The corpus size is 110k+ documents
        # The difficulty with this is that it is often a case-by-case thing. 
        # But as a rule of thumb, I generally start with min_df to 5-10 and max_df to 30% for a corpus of that size.
        
    print(f"N: {N}",  flush = True)
    
    #      A good rule of thumb for selecting max_df and min_df in the context of n-grams is to consider the distribution of n-gram frequencies in the corpus and the desired level of feature selection.

    # A common practice is to set max_df to a value between 0.8 and 0.95, which means that n-gram features that occur in more than 80-95% of the documents in the corpus will be ignored. This can help to remove commonly occurring n-grams that may not be informative for the classification task.

    # For min_df, a common practice is to set it to a value between 2 and 5, which means that n-gram features that occur in fewer than 2-5 documents in the corpus will be ignored. This can help to remove rare and potentially noisy n-grams that may not provide meaningful information for the model.
    
    
    # countvectorizer = CountVectorizer(ngram_range=(N, N), 
    #                                   max_df= 0.95, 
    #                                   min_df= 5, 
    #                                   max_features= 10000 )  # ngram [use 4-gram or 8-gram] 
    
    countvectorizer = CountVectorizer(ngram_range=(N, N))
    
    # Train Data ---------------------------------------------------------------------------------------------------------------
    Benign_Train_SG_names = [ k for k,v in Benign_Train_SG_TaskNameOpcodes_dict.items()] # list of SG names
    Malware_Train_SG_names = [ k for k,v in Malware_Train_SG_TaskNameOpcodes_dict.items()]
    Train_SG_names = Benign_Train_SG_names + Malware_Train_SG_names

    Benign_Train_data_str = [ ' '.join(v) for k,v in Benign_Train_SG_TaskNameOpcodes_dict.items()] # list of TaskNameOpcodes-strings (each string for each SG)
    Malware_Train_data_str = [ ' '.join(v) for k,v in Malware_Train_SG_TaskNameOpcodes_dict.items()]
    Train_data_str = Benign_Train_data_str + Malware_Train_data_str 
    
    # Get the Train-target (labels)
    Train_target = [0]*len(Benign_Train_data_str) + [1]*len(Malware_Train_data_str)

    # list(zip(Train_data_str, Train_data_vec, Train_target))
    
    BenignTrain_TaskNameOpcodeVector_dict = dict(zip(Benign_Train_SG_names, Benign_Train_data_str))
    MalwareTrain_TaskNameOpcodeVector_dict = dict(zip(Malware_Train_SG_names, Malware_Train_data_str))





    # https://stackoverflow.com/questions/2161752/how-to-count-the-frequency-of-the-elements-in-an-unordered-list
    BenignTrain_TaskNameOpcode_1gram_Count_Dict =  { k : {taskname: freq for taskname, freq in zip( np.unique(v.split(), return_counts=True)[0], np.unique(v.split(), return_counts=True)[1] )} for k,v in BenignTrain_TaskNameOpcodeVector_dict.items()}
    MalwareTrain_TaskNameOpcode_1gram_Count_Dict =  { k : {taskname: freq for taskname, freq in zip( np.unique(v.split(), return_counts=True)[0], np.unique(v.split(), return_counts=True)[1] )} for k,v in MalwareTrain_TaskNameOpcodeVector_dict.items()}



    def unique_tuple_counter( list_of_tuples : list):
        count_dict = {}
        for tup in list_of_tuples:
            if tup in count_dict:
                count_dict[tup] += 1
            else:
                count_dict[tup] = 1
        
        sorted_count_dict = dict( sorted(count_dict.items(), key = lambda x:x[1], reverse= True) )# descending-order
        sorted_count_dict = {" ".join(k) : v for k,v in sorted_count_dict.items()}
        return sorted_count_dict

    BenignTrain_TaskNameOpcode_Ngram_Count_Dict =  { k : unique_tuple_counter( list( ngrams( v.split(), N )) ) for k,v in BenignTrain_TaskNameOpcodeVector_dict.items()  }
    MalwareTrain_TaskNameOpcode_Ngram_Count_Dict =  { k : unique_tuple_counter( list( ngrams( v.split(), N )) ) for k,v in MalwareTrain_TaskNameOpcodeVector_dict.items()  }

    # { k : unique_tuple_counter( list( ngrams( v.split(), N )) ) for k,v in BenignTrain_TaskNameOpcodeVector_dict.items()  }



    # Fit-Transform Train-data  : https://www.meherbejaoui.com/python/counting-words-in-python-with-scikit-learn%27s-countvectorizer/
    Train_NgramEventsDist_vec = countvectorizer.fit_transform(Train_data_str).toarray() # for train-data           ( [ sum(x) for x in X_train ]  )
    # 2023-07-19
    # Turn it to Train-mean-vec
    # Train_data_vec = np.array([x / sum(x) for x in Train_data_vec])


    # Save the fitted countvectorizer : https://stackoverflow.com/questions/45674411/load-pickle-file-for-counvectorizer
    with open( os.path.join(Results_save_dirpath,f"{N}-gram Countvectorizer.pickle") , "wb") as wp:
            pickle.dump(countvectorizer, wp)    

    loaded_countvectorizer = pickle.load( open(os.path.join(Results_save_dirpath,f"{N}-gram Countvectorizer.pickle"), "rb") )   

    # Normalize ( +1e-16 is to avoid zero-division )
    # > https://stackoverflow.com/questions/46160717/two-methods-to-normalise-array-to-sum-total-to-1-0
    # Train_data_vec_normalized = [ sg_ft_vec/ (sum(sg_ft_vec) + 1e-16) for sg_ft_vec in Train_data_vec ]

    list(zip(Train_data_str, Train_NgramEventsDist_vec, Train_target))
    # Save mapping between feature-indx and feature-name
    featureIndices = list(range(len(countvectorizer.get_feature_names_out())))
    featureNames = countvectorizer.get_feature_names_out()
    featureIndexName_map = dict(zip(featureIndices, featureNames))



    # Save the N-gram Train-Dataset
    Train_NgramEventsDist_dataset = pd.DataFrame(Train_NgramEventsDist_vec, columns = featureNames ) 
    Train_NgramEventsDist_dataset["subgraph"] = Train_SG_names  # to use as index 
    Train_NgramEventsDist_dataset.set_index('subgraph', inplace= True)
    # Train_dataset.to_csv(os.path.join(Results_save_dirpath,f"{N}-gram Normalized Train-Dataset.csv"))


    ##############################################################################################################################
    # "tasknames_not_in_data" may not be in the current-data, but still deserve a zero-vector column, for fair comparison.
    tasknames__not_in_data_but_deserve_column_for_fair_comparison = list( set([x.lower() for x in ALL_Task_Names]) - set(featureNames)  )
    for column in tasknames__not_in_data_but_deserve_column_for_fair_comparison:
        Train_NgramEventsDist_dataset[column] = np.zeros(len(Train_NgramEventsDist_dataset))  # Fills the new column with zeros


    ###################################################################################################################################################

    
    # Test Data -----------------------------------------------------------------------------------------------------------------
    #   https://stackoverflow.com/questions/30287371/countvectorizer-matrix-varies-with-new-test-data-for-classification
    #   You should not fit a new CountVectorizer on the test data, you should use the one you fit on the training data and call transfrom(test_data) on it.
        #   1. you can use the same CountVectorizer that you used for your train features like this
        #       cv = CountVectorizer(parameters desired)
        #       X_train = cv.fit_transform(train_data)
        #       X_test = cv.transform(test_data)    

    Benign_Test_SG_names = [ k for k,v in Benign_Test_SG_TaskNameOpcodes_dict.items()] # list of SG names
    Malware_Test_SG_names = [ k for k,v in Malware_Test_SG_TaskNameOpcodes_dict.items()]
    Test_SG_names = Benign_Test_SG_names + Malware_Test_SG_names


    Benign_Test_data_str = [ ' '.join(v) for k,v in Benign_Test_SG_TaskNameOpcodes_dict.items()] # list of TaskNameOpcodes-strings (each string for each SG)
    Malware_Test_data_str = [ ' '.join(v) for k,v in Malware_Test_SG_TaskNameOpcodes_dict.items()]
    Test_data_str = Benign_Test_data_str + Malware_Test_data_str
    # Get the Train-target (labels)
    Test_target = [0]*len(Benign_Test_data_str) + [1]*len(Malware_Test_data_str)

    ##################################################################################################################



    # Transform Test-data  : https://www.meherbejaoui.com/python/counting-words-in-python-with-scikit-learn%27s-countvectorizer/
    Test_NgramEventsDist_vec = countvectorizer.transform(Test_data_str).toarray() # since test-data, transform with the fitted countvectorizer.
    
    # 2023-07-19
    # Turn it to Test-mean-vec
    # Test_data_vec = np.array([x / sum(x) for x in Test_data_vec])

    # Normalize ( +1e-16 is to avoid zero-division )
    # > https://stackoverflow.com/questions/46160717/two-methods-to-normalise-array-to-sum-total-to-1-0
    # Test_data_vec_normalized = [ sg_ft_vec/ (sum(sg_ft_vec) + 1e-16) for sg_ft_vec in Test_data_vec ]
    
    # Save the N-gram Test-Dataset 
    Test_NgramEventsDist_dataset = pd.DataFrame( Test_NgramEventsDist_vec, columns = featureNames )
    Test_NgramEventsDist_dataset["subgraph"] = Test_SG_names    # to use as index
    Test_NgramEventsDist_dataset.set_index('subgraph', inplace= True)
    # Test_dataset.to_csv(os.path.join(Results_save_dirpath, f"{N}-gram Normalized Test-Dataset.csv"))

    ##############################################################################################################################
    # "tasknames_not_in_data" may not be in the current-data, but still deserve a zero-vector column, for fair comparison.
    for column in tasknames__not_in_data_but_deserve_column_for_fair_comparison:
        Test_NgramEventsDist_dataset[column] = np.zeros(len(Test_NgramEventsDist_dataset))  # Fills the new column with zeros


    ####################################################################################################################
    ####################################################################################################################
    ####################################################################################################################
    ####################################################################################################################
    



    Train_dataset = Train_NgramEventsDist_dataset
    Test_dataset = Test_NgramEventsDist_dataset

    #*******************************************************************************************************************************************************#
    #*******************************************************************************************************************************************************#
    #########################################################################################################################################################


    def label_index(index_value):
        if "benign" in index_value:
            return 0
        elif "malware" in index_value:
            return 1
        else:
            raise ValueError("Invalid index value.")

    Train_dataset['label'] = Train_dataset.index.to_series().apply(label_index)
    Test_dataset['label'] = Test_dataset.index.to_series().apply(label_index)
    



    # sklearn.RandomForestClassifier randomness is dictated by paramter 'random_state'
    #        'random_state': Controls both the randomness of the bootstrapping of the samples used
    #                        when building trees (if ``bootstrap=True``) and the sampling of the
    #                        features to consider when looking for the best split at each node
    models_random_seed = 99
    model_random_seeds_txt_fp = open( os.path.join(Results_save_dirpath,f"models_random_seed.txt") , "w")
    print(f"** all models (RF/SVM/Logist/XGB) random_seed: {models_random_seed}", flush=True, file = model_random_seeds_txt_fp)
    print(f"** all models (RF/SVM/Logist/XGB) random_seed: {models_random_seed}", flush=True)

    # Fit the RF-classifier
    random_forest_classifier = RandomForestClassifier(random_state= models_random_seed, bootstrap = True, max_features='sqrt') 

    random_forest_classifier.fit( X = Train_dataset.drop(columns=['label']),                #     # https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier.fit
                                  y = Train_dataset['label'] )             
   
    # Save the RF-classifier :  https://machinelearningmastery.com/save-load-machine-learning-models-python-scikit-learn/
    pickle.dump(random_forest_classifier, open( os.path.join(Results_save_dirpath, f"RF_classifier_{N}gram_flattened_SG.sav"), "wb"))
    loaded_random_forest_classifier = pickle.load(open(os.path.join(Results_save_dirpath, f"RF_classifier_{N}gram_flattened_SG.sav"), 'rb'))

    # Test the RF-classifier
    preds = random_forest_classifier.predict( Test_dataset.drop(columns=['label']) )
    trues = Test_dataset['label']

    # Take record of Test-Results
    triplets = list(zip(Test_SG_names, preds, trues))
    wrong_answer_incidients = [x for x in triplets if x[1] != x[2]] # if pred != true

    test_accuracy = sklearn.metrics.accuracy_score(y_true = trues, y_pred = preds)
    test_f1 = sklearn.metrics.f1_score(y_true = trues, y_pred = preds)
    test_precision = sklearn.metrics.precision_score(y_true = trues, y_pred = preds)
    test_recall = sklearn.metrics.recall_score(y_true = trues, y_pred = preds)
    tn, fp, fn, tp = sklearn.metrics.confusion_matrix(trues, preds).ravel()

    # Save Test-Results
    test_results_fp = open(os.path.join(Results_save_dirpath, f"{random_forest_classifier} {N}-gram Test-Results.txt"), "w")
    print(f"{random_forest_classifier} + {N} gram", flush = True, file= test_results_fp)
    print(f"test accuracy: {test_accuracy}", flush = True, file= test_results_fp)
    print(f"test f1: {test_f1}", flush = True, file= test_results_fp)
    print(f"test precision: {test_precision}", flush = True, file= test_results_fp)
    print(f"test recall: {test_recall}", flush = True, file= test_results_fp)
    print(f"TN: {tn} || FP: {fp} || FN: {fn} || TP: {tp}", flush= True, file= test_results_fp)    
    print(f"wrong_answer_incidents: {wrong_answer_incidients}", flush = True, file= test_results_fp)
    print("", flush = True, file= test_results_fp)

    # -----------------------------------------------------------------------------------------------------------------------------
    # Fit the RF-classifier ver2
    random_forest_classifier = RandomForestClassifier(random_state= models_random_seed, bootstrap = False, max_features='sqrt') 

    random_forest_classifier.fit( X = Train_dataset.drop(columns=['label']),                #     # https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier.fit
                                  y = Train_dataset['label'] )             
   
    # Save the RF-classifier :  https://machinelearningmastery.com/save-load-machine-learning-models-python-scikit-learn/
    pickle.dump(random_forest_classifier, open( os.path.join(Results_save_dirpath, f"RF_classifier_{N}gram_flattened_SG.sav"), "wb"))
    loaded_random_forest_classifier = pickle.load(open(os.path.join(Results_save_dirpath, f"RF_classifier_{N}gram_flattened_SG.sav"), 'rb'))

    # Test the RF-classifier
    preds = random_forest_classifier.predict( Test_dataset.drop(columns=['label']) )
    trues = Test_dataset['label']

    # Take record of Test-Results
    triplets = list(zip(Test_SG_names, preds, trues))
    wrong_answer_incidients = [x for x in triplets if x[1] != x[2]] # if pred != true

    test_accuracy = sklearn.metrics.accuracy_score(y_true = trues, y_pred = preds)
    test_f1 = sklearn.metrics.f1_score(y_true = trues, y_pred = preds)
    test_precision = sklearn.metrics.precision_score(y_true = trues, y_pred = preds)
    test_recall = sklearn.metrics.recall_score(y_true = trues, y_pred = preds)
    tn, fp, fn, tp = sklearn.metrics.confusion_matrix(trues, preds).ravel()

    # Save Test-Results
    test_results_fp = open(os.path.join(Results_save_dirpath, f"{random_forest_classifier} {N}-gram Test-Results.txt"), "w")
    print(f"{random_forest_classifier} + {N} gram", flush = True, file= test_results_fp)
    print(f"test accuracy: {test_accuracy}", flush = True, file= test_results_fp)
    print(f"test f1: {test_f1}", flush = True, file= test_results_fp)
    print(f"test precision: {test_precision}", flush = True, file= test_results_fp)
    print(f"test recall: {test_recall}", flush = True, file= test_results_fp)
    print(f"TN: {tn} || FP: {fp} || FN: {fn} || TP: {tp}", flush= True, file= test_results_fp)    
    print(f"wrong_answer_incidents: {wrong_answer_incidients}", flush = True, file= test_results_fp)
    print("", flush = True, file= test_results_fp)
    
    # -----------------------------------------------------------------------------------------------------------------------------
    # Fit the RF-classifier ver3
    random_forest_classifier = RandomForestClassifier(random_state= models_random_seed, bootstrap = False, max_features=None) 

    random_forest_classifier.fit( X = Train_dataset.drop(columns=['label']),                #     # https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier.fit
                                  y = Train_dataset['label'] )             
   
    # Save the RF-classifier :  https://machinelearningmastery.com/save-load-machine-learning-models-python-scikit-learn/
    pickle.dump(random_forest_classifier, open( os.path.join(Results_save_dirpath, f"RF_classifier_{N}gram_flattened_SG.sav"), "wb"))
    loaded_random_forest_classifier = pickle.load(open(os.path.join(Results_save_dirpath, f"RF_classifier_{N}gram_flattened_SG.sav"), 'rb'))

    # Test the RF-classifier
    preds = random_forest_classifier.predict( Test_dataset.drop(columns=['label']) )
    trues = Test_dataset['label']

    # Take record of Test-Results
    triplets = list(zip(Test_SG_names, preds, trues))
    wrong_answer_incidients = [x for x in triplets if x[1] != x[2]] # if pred != true

    test_accuracy = sklearn.metrics.accuracy_score(y_true = trues, y_pred = preds)
    test_f1 = sklearn.metrics.f1_score(y_true = trues, y_pred = preds)
    test_precision = sklearn.metrics.precision_score(y_true = trues, y_pred = preds)
    test_recall = sklearn.metrics.recall_score(y_true = trues, y_pred = preds)
    tn, fp, fn, tp = sklearn.metrics.confusion_matrix(trues, preds).ravel()

    # Save Test-Results
    test_results_fp = open(os.path.join(Results_save_dirpath, f"{random_forest_classifier} {N}-gram Test-Results.txt"), "w")
    print(f"{random_forest_classifier} + {N} gram", flush = True, file= test_results_fp)
    print(f"test accuracy: {test_accuracy}", flush = True, file= test_results_fp)
    print(f"test f1: {test_f1}", flush = True, file= test_results_fp)
    print(f"test precision: {test_precision}", flush = True, file= test_results_fp)
    print(f"test recall: {test_recall}", flush = True, file= test_results_fp)
    print(f"TN: {tn} || FP: {fp} || FN: {fn} || TP: {tp}", flush= True, file= test_results_fp)    
    print(f"wrong_answer_incidents: {wrong_answer_incidients}", flush = True, file= test_results_fp)
    print("", flush = True, file= test_results_fp)


    ##################################################################################################################3
    ##################################################################################################################3

    test_predprobas_fp = open(os.path.join(Results_save_dirpath, f"RF {N}-gram Predict-Probas.txt"), "w")

    Test_data__vec_str_dict = { x[0]: {'vec':x[1], 'str':x[2]} for x in list(zip( Test_dataset.index,
                                                                                 Test_dataset.drop(columns=['label']).values,
                                                                                 Test_data_str )) }
    # TODO: Predict_proba doesn't seem right. Check this.
    Predict_proba_dict = dict()
    for data_name, data in Test_data__vec_str_dict.items():
        # print(f"{data_name}: {random_forest_classifier.predict_proba([Test_data__vec_str_dict[data_name]['vec']])}",  flush = True)
        # print(f"{data_name}: {random_forest_classifier.predict_proba([Test_data__vec_str_dict[data_name]['vec']])}",  flush = True, file = test_predprobas_fp)
        Predict_proba_dict[data_name] = random_forest_classifier.predict_proba([Test_data__vec_str_dict[data_name]['vec']]).tolist()

    for data_name, data in Predict_proba_dict.items():
        print(f"{data_name}: {Predict_proba_dict[data_name]}",  flush = True)
        print(f"{data_name}: {Predict_proba_dict[data_name]}",  flush = True, file = test_predprobas_fp)



    #wrong_answer_data_name = wrong_answer_incidients[0][0]

    #-----------------------------------------------------------------------------------------------
    # < SHAP Explainer >
    
    # Previously tried LIME, but it appears to not output the n-gram features in its explanation output.
    # Instead, it outputs the unigram output.

    # As an alternative, 
    # trying "SHAP (SHapley Additive exPlanations)"
    #
    # What Shapley does is quantifying the contribution that each player brings to the game. 
    # What SHAP does is quantifying the contribution that each feature brings to the prediction made by the model.
    # It is important to stress that what we called a “game” concerns a single observation. One game: one observation. 
    # Indeed, SHAP is about local interpretability of a predictive model.
    # 
    # The absolute SHAP value shows us how much a single feature affected the prediction, 
    # so Longitude contributed the most, MedInc the second one, AveOccup the third, and Population was the feature with the lowest contribution to the preditcion.
    # (from https://towardsdatascience.com/using-shap-values-to-explain-how-your-machine-learning-model-works-732b3f40e137)
    #
    # > https://medium.com/@kalia_65609/interpreting-an-nlp-model-with-lime-and-shap-834ccfa124e4
    # > https://shap-lrjball.readthedocs.io/en/latest/generated/shap.TreeExplainer.html
    # > https://github.com/slundberg/shap
    # > https://github.com/slundberg/shap/issues/2611
    # > https://towardsdatascience.com/shap-explained-the-way-i-wish-someone-explained-it-to-me-ab81cc69ef30
    # > https://towardsdatascience.com/using-shap-values-to-explain-how-your-machine-learning-model-works-732b3f40e137
    # > https://github.com/slundberg/shap/issues/153 (visualize the first prediction's explanation)
    # --------------------------------------------------------------------------------------------------------------------------------------
    # SHAP Values
    #
    # SHAP value as a united approach to explaining the output of any machine learning model. Three benefits worth mentioning here.
    # 
    # 1. The first one is global interpretability — the collective SHAP values can show how much each predictor contributes, 
    #    either positively or negatively, to the target variable. 
    #    This is like the variable importance plot but it is able to show the positive or negative relationship for each variable with the target 
    #   (see the SHAP value plot below).
    #
    # 2. The second benefit is local interpretability — each observation gets its own set of SHAP values (see the individual SHAP value plot below). 
    #    This greatly increases its transparency. We can explain why a case receives its prediction and the contributions of the predictors. 
    #    Traditional variable importance algorithms only show the results across the entire population but not on each individual case. 
    #    The local interpretability enables us to pinpoint and contrast the impacts of the factors.
    #
    # 3. Third, the SHAP values can be calculated for any tree-based model, 
    #    while other methods use linear regression or logistic regression models as the surrogate models.
    # 
    # ^ source: https://medium.com/dataman-in-ai/explain-your-model-with-the-shap-values-bc36aac4de3d
    #
    # https://stackoverflow.com/questions/73648498/valueerror-max-evals-500-is-too-low-for-the-permutation-explainer-shap-answer
    #
    #shap_values = explainer.shap_values(Test_Data_dict[wrong_answer_data_name]['vec'])
    # how to well visulize shap_values for a single-row
    # > https://shap-lrjball.readthedocs.io/en/latest/generated/shap.summary_plot.html
    #import matplotlib.pyplot as plt
    #shap.summary_plot(shap_values,  X_test,  plot_type="bar", show = False)
    #plt.savefig('/data/d1/jgwak1/tabby/BASELINE_COMPARISONS_20230124/Sequential/Ngrams/d.png')

    ###################################################################################################################################################
    # Global Interpretability Plots

    import matplotlib.pyplot as plt    
    # (A) Variable Importance Plot — Global Interpretability -- summary_plot (bar)
    # https://github.com/slundberg/shap/issues/2073

    # Added by JY @ 2023-2-11
    # > With N == 2, there is an error that seems to be an implementation of code running under the hood 
    #   (https://stackoverflow.com/questions/68233466/shap-exception-additivity-check-failed-in-treeexplainer)
    #   According to the above stateoverflow post,
    #   "A quick workaround (which does not affect much the features ranking by SHAP values in the above scenario where the ) 
    if N == 2:
        check_additivity = False
    else:
        check_additivity = True # default
    shap_values = shap.TreeExplainer(random_forest_classifier).shap_values(np.array(Test_dataset.drop(columns=['label']).values), check_additivity=check_additivity) 
    f = plt.figure()
    shap.summary_plot(shap_values = shap_values[1], # class-1 (positive class ; malware)
                      features = Test_dataset.drop(columns=['label']).values, 
                      feature_names = list(Test_dataset.drop(columns=['label']).columns),      # pipeline[0] == our countvectorizer (n-gram)
                      plot_type = "bar")
    f.savefig( os.path.join(Results_save_dirpath, f"{N}-gram SHAP_global_interpretability_summary_barplot_malware_with_featureName.png"), bbox_inches='tight', dpi=600)

    #shap_values = shap.TreeExplainer(random_forest_classifier).shap_values(X_test)  # pipeline[-1] == our fitted sklearn.ensemble.RandomForestClassifier Instance
    #f = plt.figure()
    #shap.summary_plot(shap_values = shap_values[1], # class-1 (positive class ; malware)
    #                  features = X_test, 
    #                  plot_type = "bar")
    #f.savefig( os.path.join(f"{N}-gram SHAP_global_interpretability_summary_barplot_malware_with_featureIndex.png", bbox_inches='tight', dpi=600) 

    # TODO: Important: Get a feature-importance from shap-values
    # https://stackoverflow.com/questions/65534163/get-a-feature-importance-from-shap-values

    rf_resultX = pd.DataFrame(shap_values[1], columns = list(Test_dataset.drop(columns=['label']).columns))

    vals = np.abs(rf_resultX.values).mean(0)

    shap_importance = pd.DataFrame(list(zip(list(Test_dataset.drop(columns=['label']).columns), vals)),
                                    columns=['col_name','feature_importance_vals']) # Later could make use of the "feature_importance_vals" if needed.
    shap_importance.sort_values(by=['feature_importance_vals'],
                                ascending=False, inplace=True)
    shap_importance.head()  # JY: This is correct. verfiied with plot GREAT!
    
    # Get first 20 
    shap_importance.reset_index()
    shap_importance.reset_index()[0:20]

    FirstX = 500
    Global_FirstXImportant_featureIndex_featureName = dict(zip(shap_importance.reset_index()[0:FirstX]['index'], shap_importance.reset_index()[0:FirstX]['col_name']))
    Global_FirstXImportant_featureNames = [ v for k,v in Global_FirstXImportant_featureIndex_featureName.items() ]

    # Save Global-First20Important Features "Train dataset"
    Global_FirstXImportant_Features_Train_dataset = Train_dataset[ Global_FirstXImportant_featureNames ]
    # Get the SUM of Shap-values by subgraph(row).
    Global_FirstXImportant_Features_Train_dataset = Global_FirstXImportant_Features_Train_dataset.assign(SUM = Global_FirstXImportant_Features_Train_dataset.sum(axis=1)) 
    Global_FirstXImportant_Features_Train_dataset.to_csv(os.path.join(Results_save_dirpath, f"RF {N}-gram Global-SHAP First{FirstX} Important FeatureNames Train-Dataset.csv"))


    # Save Global-First20Important Features "Test dataset"
    Global_FirstXImportant_Features_Test_dataset = Test_dataset[ Global_FirstXImportant_featureNames ] #  여기가 문제인듯
    # Get the SUM of Shap-values by subgraph(row).
    Global_FirstXImportant_Features_Test_dataset = Global_FirstXImportant_Features_Test_dataset.assign(SUM = Global_FirstXImportant_Features_Test_dataset.sum(axis=1)) 
    Global_FirstXImportant_Features_Test_dataset["predict_proba"] = pd.Series(Predict_proba_dict)
    Global_FirstXImportant_Features_Test_dataset.to_csv(os.path.join(Results_save_dirpath, f"RF {N}-gram Global-SHAP First{FirstX} Important FeatureNames Test-Dataset.csv"))



    # Save Global-First20Important Features "ALL dataset" (After Integrating Train and Test)

    Global_FirstXImportant_Features_All_dataset = pd.concat( [ Global_FirstXImportant_Features_Train_dataset, Global_FirstXImportant_Features_Test_dataset ] , axis = 0 )
    Global_FirstXImportant_Features_All_dataset.sort_values(by="subgraph", inplace= True)
    Global_FirstXImportant_Features_All_dataset.to_csv(os.path.join(Results_save_dirpath, f"RF {N}-gram Global-SHAP First{FirstX} Important FeatureNames ALL-Dataset.csv"))

    ###########################################################################

    

    # Train_dataset_All_Features = Train_dataset.assign(SUM = Train_dataset.sum(axis=1)) # Get the SUM by row.
    # Train_dataset_All_Features.to_csv(os.path.join(Results_save_dirpath, f"{N}-gram All_featureNames Train Dataset.csv"))


    # (A) Variable Importance Plot — Global Interpretability -- summary_plot (NO bar)
    #shap_values = shap.TreeExplainer(random_forest_classifier).shap_values(X_test)
    #f = plt.figure()
    #shap.summary_plot(shap_values = shap_values[1],  # class-1 (positive class ; malware)    
    #                  features = X_test,
    #                  feature_names = list(countvectorizer.get_feature_names_out()),      # pipeline[0] == our countvectorizer (n-gram)
    #                 )    # https://shap-lrjball.readthedocs.io/en/latest/generated/shap.summary_plot.html
    #f.savefig(f"{N}-gram SHAP_global_interpretability_summary_nobarplot_malware.png", bbox_inches='tight', dpi=600)


    # LOCAL---------------------------------------------------------------------------------------------------------------------------------------------------------
    # Interpretation of force-plot
    # > https://github.com/slundberg/shap/issues/977


    # JY @ 2023-05-03:
    #                 Get ForcePlot for ALL samples
    #                 But divide into correct/incorrect prediction-dirs 

    FORCE_PLOTS_Local_Explanation_dirpath = os.path.join(Results_save_dirpath, f"FORCE_PLOTS_Local-Explanation_{N}gram_{experiment_identifier_dateinfo}")
    Correct_Predictions_ForcePlots_subdirpath = os.path.join(FORCE_PLOTS_Local_Explanation_dirpath, "Correct_Predictions")
    Mispredictions_ForcePlots_subdirpath = os.path.join(FORCE_PLOTS_Local_Explanation_dirpath, "Mispredictions")
    if not os.path.exists( FORCE_PLOTS_Local_Explanation_dirpath ):
         os.makedirs(FORCE_PLOTS_Local_Explanation_dirpath)
    if not os.path.exists( Correct_Predictions_ForcePlots_subdirpath ):
         os.makedirs( Correct_Predictions_ForcePlots_subdirpath )
    if not os.path.exists( Mispredictions_ForcePlots_subdirpath ):
         os.makedirs( Mispredictions_ForcePlots_subdirpath )


    misprediction_subgraph_names = [wrong_answer_incident[0] for wrong_answer_incident in wrong_answer_incidients]

    # Iterate through all tested-subgraphs
    for Test_SG_name in Test_SG_names:

        explainer = shap.TreeExplainer(random_forest_classifier)


        shap_values = explainer.shap_values(X = Test_data__vec_str_dict[Test_SG_name]['vec'], check_additivity=check_additivity)  # X: numpy.array, pandas.DataFrame or catboost.Pool (for catboost)
                                                                                    #    A matrix of samples (# samples x # features) on which to explain the model’s output.
                                                                                    # https://shap-lrjball.readthedocs.io/en/latest/generated/shap.TreeExplainer.html#shap.TreeExplainer.shap_values
                                                                                    #
                                                                                    # Returns: array or list
                                                                                    #          For models with a single output this returns a matrix of SHAP values (# samples x # features)
                                                                                    #          Each row sums to the difference between the model output for that sample 
                                                                                    #          and the expected value of the model output (which is stored in the expected_value attribute of the explainer when it is constant). 
                                                                                    #          
                                                                                    #          *For models with vector outputs this returns a list of such matrices, one for each output.
                                                                                    # 
                                                                                    #  https://github.com/slundberg/shap/issues/1143
                                                                                    #  My guess is that your model has multiple outputs, perhaps binary classification? 
                                                                                    #  If so the shap_values are a list of explanations for each output, and you will need to select one of this explanations first before plotting.
        
        # base value (the average model output over the training dataset we passed) : https://github.com/slundberg/shap/issues/352

        # https://shap-lrjball.readthedocs.io/en/latest/generated/shap.force_plot.html
        
        # TypeError: Object of type int64 is not JSON serializable                   
        # https://github.com/slundberg/shap/issues/1100

        # https://shap-lrjball.readthedocs.io/en/latest/generated/shap.force_plot.html
        # https://github.com/slundberg/shap/blob/master/shap/plots/_force.py << forceplot() SOURCE-CODE
        
        
        # TODO @ 2023-02-09: Get forceplot for all mispreddictions
        forceplot_out = shap.force_plot(base_value = explainer.expected_value[1], 
                                        shap_values = shap_values[1], 
                                        # Following "int(x)" Because of "TypeError: Object of type int64 is not JSON serializable"
                                        # features = [int(x) for x in Test_data__vec_str_dict[Test_SG_name]['vec']],
                                        # feature_names = list(loaded_countvectorizer.get_feature_names_out()),  

                                        features = Test_dataset.loc[Test_SG_name].drop('label'),

                                        out_names = Test_SG_name,

                                        # link : "identity" or "logit"
                                        #         The transformation used when drawing the tick mark labels. Using logit will change log-odds numbers into probabilities. 
                                        link = "logit",

                                        matplotlib=False, show=False,
                                        text_rotation=True) # https://github.com/slundberg/shap/issues/153
        


        if Test_SG_name in misprediction_subgraph_names:

            shap.save_html( os.path.join(Mispredictions_ForcePlots_subdirpath, f"{N}-gram SHAP_local_interpretability_force_plot_{Test_SG_name}.html"), forceplot_out)
        
        else:
            shap.save_html( os.path.join(Correct_Predictions_ForcePlots_subdirpath, f"{N}-gram SHAP_local_interpretability_force_plot_{Test_SG_name}.html"), forceplot_out)


    # SIMILAR TO PROBLEM THAT I HAVE:    
    # https://stackoverflow.com/questions/54292226/putting-html-output-from-shap-into-the-dash-output-layout-callback

    # https://github.com/slundberg/shap/issues/1252
    # explainer.expected_value[1] : base value (average model output value w.r.t. positive class) w.r.t. positive (malware) class
    # shap_values[1]: shap-value w.r.t to positive (malware) class                 
    # values of the test-data (observation) that we're interested in.
    # 
    # https://github.com/slundberg/shap/issues/352 : Interpretation of Base Value and Predicted Value in SHAP Plots
    # The three arguments to force_plot above represent the expected_value of the first class, 
    # the SHAP values of the first class prediction model, 
    # and the data of the first sample row. 
    # If you want to explain the output of the second class you would need to change the index to 1 for the first two arguments.                 
                                                                        
    # force-plot interpretation:
    #                           these features actually contributed positively to the prediction of the negative class 
    #                           (i.e. these made the passenger more likely to have died.) 
    #                           As you can see by the red "higher" arrow, these can be seen as "pushing" or "forcing" the prediction higher, 
    #                           while the blue "lower" arrows are features that force the prediction lower.
                                                                 



    ###################################################################################################################################################

    #def featureIndex2featureName(feature_Index):
    #    feature_names = list(countvectorizer.get_feature_names_out())
    #    return feature_names[feature_Index]

    #def featureName2featureIndex(feature_Name):
    #    feature_names = list(countvectorizer.get_feature_names_out())
    #    return feature_names.index(feature_Name)




    # WHY RF+Ngram IS WORKING BETTER THAN OUR GAT(neural-network)?
    # > Related sources
    #   https://arxiv.org/pdf/2108.13637.pdf
    #   https://machinelearningmastery.com/impact-of-dataset-size-on-deep-learning-model-skill-and-performance-estimates/
    #   https://blog.frankfurt-school.de/wp-content/uploads/2018/10/Neural-Networks-vs-Random-Forests.pdf (IMPORTANT)
    #   https://www.quora.com/Which-classifier-is-better-random-forests-or-deep-neural-networks
    #   https://dzone.com/articles/3-reasons-to-use-random-forest-over-a-neural-netwo


    # > Are there any Counfounding features? (JY)
    #  >> This should actually be a diffifult task for a simplistic "bag-of-words" style classifier. 
    #     However, simple classifiers perform remarkably well on this task. 
    #     This is because this dataset is "famous" for the presence of many confounding features, for instance the e-mail domains that are present in the header
    #  JY: RF+Ngram 에 confounding feature 이 있나 확인해봐라 

    # Integration of ideas from RF into GAT?
    # > Related sources
    #   https://www.nature.com/articles/s41598-018-34833-6
    #   https://faculty.ist.psu.edu/szw494/publications/NNRF.pdf

    # Small data and overfitting in regards to neural networks and RF
    # >
    #   Overfitting. 
    #   There are inherent limitations when fitting machine learning models to smaller datasets. 
    #   As the training datasets get smaller, the models have fewer examples to learn from, increasing the risk of overfitting.
    #   
    # > Related sources
    #   https://www.quora.com/Why-wont-a-neural-network-overfit-a-small-dataset

    # Why do deep neural nets require so much training data to perform well?
    # > https://www.quora.com/Why-do-deep-neural-nets-require-so-much-training-data-to-perform-well
    
    # Overfitting on small dataset to check if model is good
    # https://stats.stackexchange.com/questions/587762/overfitting-on-small-dataset-to-check-if-model-is-good

    # sklearn.LogisticRegression randomness is dictated by paramter 'random_state'. 
    # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
    from sklearn.linear_model import LogisticRegression
    LogisticRegression_model = LogisticRegression(random_state= models_random_seed)
    
    LogisticRegression_model.fit( X = Train_dataset.drop(columns=['label']),                #     # https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier.fit
                                  y = Train_dataset['label'] )          
    preds = LogisticRegression_model.predict( Test_dataset.drop(columns=['label']) )
    trues = Test_target

    # Take record of Test-Results
    triplets = list(zip(Test_SG_names, preds, trues))
    wrong_answer_incidients = [x for x in triplets if x[1] != x[2]] # if pred != true

    test_accuracy = sklearn.metrics.accuracy_score(y_true = trues, y_pred = preds)
    test_f1 = sklearn.metrics.f1_score(y_true = trues, y_pred = preds)
    test_precision = sklearn.metrics.precision_score(y_true = trues, y_pred = preds)
    test_recall = sklearn.metrics.recall_score(y_true = trues, y_pred = preds)
    tn, fp, fn, tp = sklearn.metrics.confusion_matrix(trues, preds).ravel()

    # Save Test-Results
    test_results_fp = open(os.path.join(Results_save_dirpath, f"Logistic-Regression {N}-gram Test-Results.txt"), "w")
    print(f"{LogisticRegression_model} + {N} gram", flush = True, file= test_results_fp)
    print(f"test accuracy: {test_accuracy}", flush = True, file= test_results_fp)
    print(f"test f1: {test_f1}", flush = True, file= test_results_fp)
    print(f"test precision: {test_precision}", flush = True, file= test_results_fp)
    print(f"test recall: {test_recall}", flush = True, file= test_results_fp)
    print(f"TN: {tn} || FP: {fp} || FN: {fn} || TP: {tp}", flush= True, file= test_results_fp)    
    print(f"wrong_answer_incidents: {wrong_answer_incidients}", flush = True, file= test_results_fp)
    print("", flush = True, file= test_results_fp)


    ####################################################################################################################

    # Top20 Global-Shap 4gram across malware vs benign samples
    # > Signal acr

    TopX = FirstX

    # Get TopX highest (global) shap value features (i.e., features that are most responsible for pushing predictions towards malware)
    Global_TopX_Important_featureIndex_featureName = dict(zip(shap_importance.reset_index()[0:TopX]['index'], shap_importance.reset_index()[0:TopX]['col_name']))
    Global_TopX_Important_featureNames = [ v for k,v in Global_TopX_Important_featureIndex_featureName.items() ]

    # Benign_TaskNameOpcode_Ngram_Count_Dict = BenignTrain_TaskNameOpcode_Ngram_Count_Dict | BenignTest_TaskNameOpcode_Ngram_Count_Dict
    # Malware_TaskNameOpcode_Ngram_Count_Dict = MalwareTrain_TaskNameOpcode_Ngram_Count_Dict | MalwareTest_TaskNameOpcode_Ngram_Count_Dict

    # # make nested dict keys lowercase for matching
    # Benign_TaskNameOpcode_Ngram_Count_Dict = eval(repr(Benign_TaskNameOpcode_Ngram_Count_Dict).lower())
    # Malware_TaskNameOpcode_Ngram_Count_Dict = eval(repr(Malware_TaskNameOpcode_Ngram_Count_Dict).lower())


    def Keep_TopX_Ngram_Count_Dict(Ngram_Count_Dict: dict, TopX_featureNames : list):
        out_dict = dict()
        for sample_name, nested_dict in Ngram_Count_Dict.items():
              out_dict[sample_name] = dict()
              for ngram_key, count in nested_dict.items():
                   if ngram_key in TopX_featureNames:
                        out_dict[sample_name][ngram_key] = count
        return out_dict

    # # make keys into lowercase for matching
    # Benign_TaskNameOpcode_Ngram_Count_Dict_TopXShapFeatures = Keep_TopX_Ngram_Count_Dict(Benign_TaskNameOpcode_Ngram_Count_Dict, Global_TopX_Important_featureNames)
    # Malware_TaskNameOpcode_Ngram_Count_Dict_TopXShapFeatures = Keep_TopX_Ngram_Count_Dict(Malware_TaskNameOpcode_Ngram_Count_Dict, Global_TopX_Important_featureNames)
    # with open( os.path.join(Results_save_dirpath,f"Benign_TaskNameOpcode_{N}gram_Count_Dict_Top{TopX}ShapFeatures.json") , "w") as wp:
    #         json.dump(Benign_TaskNameOpcode_Ngram_Count_Dict_TopXShapFeatures, wp, cls = NpEncoder)      

    # with open( os.path.join(Results_save_dirpath, f"Malware_TaskNameOpcode_{N}gram_Count_Dict_Top{TopX}ShapFeatures.json"), "w") as wp:
    #         json.dump(Malware_TaskNameOpcode_Ngram_Count_Dict_TopXShapFeatures, wp, cls = NpEncoder)     

    shap_importance.reset_index()[0:TopX][['col_name', 'feature_importance_vals']].to_csv(os.path.join(Results_save_dirpath, f"Top{TopX}_SHAP_Global_Features.csv"))

    ####################################################################################################################
    # Support Vector Machines (SVM)
    # "SVM train"

   # sklearn svm.SVC does not necessarily have a randomness involved. 
   # There exists a 'random_state' parameter but is ignored when parameter 'probability' is False which is default.
   # For more details: https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html

    svm_classifier = svm.SVC(kernel = 'rbf')          # kernel{‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’} or callable, default=’rbf’
   
    svm_classifier.fit( X = Train_dataset.drop(columns=['label']),               
                        y = Train_target )             
   
    #pickle.dump(svm_classifier, open(f"/data/d1/jgwak1/tabby/BASELINE_COMPARISONS_20230124/Sequential/Ngrams/SVM_classifier_fitted_on_{n}gram_flattened_SG.sav", "wb"))
    #loaded_model = pickle.load(open(f"/home/jgwak1/tabby/Baseline_Comparisons/Sequential/Ngrams/SVM_classifier_fitted_on_{n}gram_flattened_SG.sav", 'rb'))


    # "SVM test"
    preds = svm_classifier.predict( Test_dataset.drop(columns=['label']) )
    trues = Test_target

    # Take record of Test-Results
    triplets = list(zip(Test_SG_names, preds, trues))
    wrong_answer_incidients = [x for x in triplets if x[1] != x[2]] # if pred != true

    test_accuracy = sklearn.metrics.accuracy_score(y_true = trues, y_pred = preds)
    test_f1 = sklearn.metrics.f1_score(y_true = trues, y_pred = preds)
    test_precision = sklearn.metrics.precision_score(y_true = trues, y_pred = preds)
    test_recall = sklearn.metrics.recall_score(y_true = trues, y_pred = preds)
    tn, fp, fn, tp = sklearn.metrics.confusion_matrix(trues, preds).ravel()

    # Save Test-Results
    test_results_fp = open(os.path.join(Results_save_dirpath, f"SVM {N}-gram Test-Results.txt"), "w")
    print(f"{svm_classifier} + {N} gram", flush = True, file= test_results_fp)
    print(f"test accuracy: {test_accuracy}", flush = True, file= test_results_fp)
    print(f"test f1: {test_f1}", flush = True, file= test_results_fp)
    print(f"test precision: {test_precision}", flush = True, file= test_results_fp)
    print(f"test recall: {test_recall}", flush = True, file= test_results_fp)
    print(f"wrong_answer_incidents: {wrong_answer_incidients}", flush = True, file= test_results_fp)
    print(f"TN: {tn} || FP: {fp} || FN: {fn} || TP: {tp}", flush= True, file= test_results_fp)    
    print("", flush = True, file= test_results_fp)

    ########################################################################################################################

    # from xgboost import XGBClassifier
    # xgb_classifier = XGBClassifier(random_state=42, seed=2, colsample_bytree=0.6, subsample=0.7)
    from sklearn.ensemble import GradientBoostingClassifier # XGB == GraidentBoosting (just use this, for coherence of using sklearn) 
    


    xgb_classifier = GradientBoostingClassifier( random_state = models_random_seed)


    xgb_classifier.fit( X = Train_dataset.drop(columns=['label']),                #     # https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier.fit
                        y = Train_dataset['label'] )   


    # "SVM test"
    preds = xgb_classifier.predict( Test_dataset.drop(columns=['label']) )
    trues = Test_target


    # Take record of Test-Results
    triplets = list(zip(Test_SG_names, preds, trues))
    wrong_answer_incidients = [x for x in triplets if x[1] != x[2]] # if pred != true

    test_accuracy = sklearn.metrics.accuracy_score(y_true = trues, y_pred = preds)
    test_f1 = sklearn.metrics.f1_score(y_true = trues, y_pred = preds)
    test_precision = sklearn.metrics.precision_score(y_true = trues, y_pred = preds)
    test_recall = sklearn.metrics.recall_score(y_true = trues, y_pred = preds)
    tn, fp, fn, tp = sklearn.metrics.confusion_matrix(trues, preds).ravel()

    # Save Test-Results
    test_results_fp = open(os.path.join(Results_save_dirpath, f"XGB {N}-gram Test-Results.txt"), "w")
    print(f"{xgb_classifier} + {N} gram", flush = True, file= test_results_fp)
    print(f"test accuracy: {test_accuracy}", flush = True, file= test_results_fp)
    print(f"test f1: {test_f1}", flush = True, file= test_results_fp)
    print(f"test precision: {test_precision}", flush = True, file= test_results_fp)
    print(f"test recall: {test_recall}", flush = True, file= test_results_fp)
    print(f"TN: {tn} || FP: {fp} || FN: {fn} || TP: {tp}", flush= True, file= test_results_fp)    
    print(f"wrong_answer_incidents: {wrong_answer_incidients}", flush = True, file= test_results_fp)
    print("", flush = True, file= test_results_fp)

    print(f"{xgb_classifier} + {N} gram", flush = True)
    print(f"test accuracy: {test_accuracy}", flush = True)
    print(f"test f1: {test_f1}", flush = True)
    print(f"test precision: {test_precision}", flush = True)
    print(f"test recall: {test_recall}", flush = True)   
    print(f"TN: {tn} || FP: {fp} || FN: {fn} || TP: {tp}", flush= True)    
    print(f"wrong_answer_incidents: {wrong_answer_incidients}", flush = True)
    print("", flush = True)

