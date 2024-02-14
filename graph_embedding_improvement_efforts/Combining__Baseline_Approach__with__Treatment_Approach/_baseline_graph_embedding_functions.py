from sklearn.feature_extraction.text import CountVectorizer
from typing import Tuple, Union
import torch
import re 

taskname_colnames = [
    'None_or_empty', #0 (index 0) 
    'Cleanup', #1
    'Close', #2
    'Create', #3
    'CreateNewFile', #4
    'DeletePath',#5
    'DirEnum',#6
    'DirNotify',#7
    'Flush',#8
    'FSCTL',#9
    'NameCreate',#10
    'NameDelete',#11
    'OperationEnd',#12
    #'QueINFO',#13
    'QueryInformation',#13
    'QueryEA',#14
    'QuerySecurity',#15
    'Read',#16
    'Write',#17
    'SetDelete',#18
    'SetInformation', #19
    'PagePriorityChange',#20
    'IoPriorityChange',#21
    'CpuBasePriorityChange',#22
    #'IMAGEPriorityChange',#24
    'CpuPriorityChange',#23
    'ImageLoad',#24
    'ImageUnload',#25
    'ProcessStop/Stop',#26
    'ProcessStart/Start',#27
    'ProcessFreeze/Start',#28--------
    #'PSDISKIOATTRIBUTE',#31
    #'PSIORATECONTROL',#32 
    'ThreadStart/Start',#29
    'ThreadStop/Stop',#30
    'ThreadWorkOnBehalfUpdate', #31
    'JobStart/Start',#32--------
    'JobTerminate/Stop',#33--------
    #'LOSTEVENT',#38
    #'PSDISKIOATTRIBUTION',#39
    'Rename',#34
    'Renamepath',#35
    'RegPerfOpHiveFlushWroteLogFile',#36------- "Thisgroupofeventstrackstheperformanceofflushinghives": "RegPerfOpHiveFlushWroteLogFile",
    'CreateKey',#37 "EventID(1)":"CreateKey", 
    'OpenKey',#38 "EventID(2)":"OpenKey",
    'DeleteKey',#39 "EventID(3)":"DeleteKey", 
    'QueryKey',#40 "EventID(4)":"QueryKey", 
    'SetValueKey',#41 "EventID(5)":"SetValueKey", 
    'DeleteValueKey',#42 "EventID(6)":"DeleteValueKey", 
    'QueryValueKey',#43 "EventID(7)":"QueryValueKey",  
    'EnumerateKey',#44 "EventID(8)":"EnumerateKey",  
    'EnumerateValueKey',#45 "EventID(9)":"EnumerateValueKey", 
    'QueryMultipleValueKey',#46 "EventID(10)":"QueryMultipleValueKey",
    'SetinformationKey',#47 "EventID(11)":"SetinformationKey", 
    'CloseKeys',#48 "EventID(13)":"CloseKey", 
    'QuerySecurityKey',#49 "EventID(14)":"QuerySecurityKey",
    'SetSecurityKey',#50 "EventID(15)":"SetSecurityKey", 
    'KERNEL_NETWORK_TASK_TCPIP/Datasent.', #51
    'KERNEL_NETWORK_TASK_TCPIP/Datareceived.',#52
    'KERNEL_NETWORK_TASK_TCPIP/Connectionattempted.',#53
    'KERNEL_NETWORK_TASK_TCPIP/Disconnectissued.', #54
    'KERNEL_NETWORK_TASK_TCPIP/Dataretransmitted.',#55
    'KERNEL_NETWORK_TASK_TCPIP/connectionaccepted.' , #56----
    'KERNEL_NETWORK_TASK_TCPIP/Protocolcopieddataonbehalfofuser.', #57
    'KERNEL_NETWORK_TASK_UDPIP/DatareceivedoverUDPprotocol.',#58
    'KERNEL_NETWORK_TASK_UDPIP/DatasentoverUDPprotocol.', #59
    # 'NULL', #remove this entry once second step updations used for subgra
    'Unseen', # 60
    #PW: below events are now different in silketw
    # all below 3 task are combined with opcode and having index 43 onwards for all of them in the function TN2int()
    # 'KERNEL_NETWORK_TASK_UDPIP'#index 43 # 42(opcode value) 43,49(https://github.com/repnz/etw-providers-docs/blob/master/Manifests-Win7-7600/Microsoft-Windows-Kernel-Network.xml)
    # 'KERNEL_NETWORK_TASK_TCPIP', # 10-18 (https://github.com/repnz/etw-providers-docs/blob/master/Manifests-Win7-7600/Microsoft-Windows-Kernel-Network.xml)
    # 'MICROSOFT-WINDOWS-KERNEL-REGISTRY', # 32- 46 (https://github.com/repnz/etw-providers-docs/blob/master/Manifests-Win7-7600/Microsoft-Windows-Kernel-Registry.xml)

]


def get_baseline_3__flattened_graph_Ngram_events__node_type_counts_dict( dataset : list, param2 : Union[int, CountVectorizer] ) -> Union[Tuple[dict, CountVectorizer], dict]:

      # Function being used for "train-set"
      if isinstance(param2, int):
            
            N = param2
            countvectorizer = CountVectorizer(ngram_range=(N, N),
                                              max_df= 1.0,
                                              min_df= 1,
                                              max_features= None)   

            data__node_type_count__dict = dict()
            data__sorted_event_str_sequence__dict = dict()

            train_dataset = dataset

            cnt = 1
            for data in train_dataset:
               print(f"{cnt} / {len(train_dataset)}: {data.name} -- collect 'node-type count' and 'sorted event str sequence'", flush = True)
               # node-type counts
               data.x = data.x[:,:5]
               nodetype_counts = torch.sum(data.x, axis = 0)
               data__node_type_count__dict[re.search(r'Processed_SUBGRAPH_P3_(.*)\.pickle', data.name).group(1) ] = nodetype_counts.tolist()

               # collect sorted event str sequence for each data
               timestamp_sorted_indices = torch.argsort(data.edge_attr[:, -1], descending=False)
               edge_attr__sorted = data.edge_attr[ timestamp_sorted_indices ]
               taskname_indices = torch.nonzero(edge_attr__sorted[:,:-1], as_tuple=False)[:, -1]
               sorted_event_list = [taskname_colnames[i] for i in taskname_indices]
               sorted_event_str_sequence = " ".join(sorted_event_list) # for compabitiblity with countvecotrizer
               data__sorted_event_str_sequence__dict[ re.search(r'Processed_SUBGRAPH_P3_(.*)\.pickle', data.name).group(1) ] = sorted_event_str_sequence # wrap in list for easy compatiblity later

               cnt+=1

            # Now fit countvectorizer based on the accumulated 'data__sorted_event_sequence__dict' 
            # -- Need to fit countvectorizer on entire train-set , instead of one by one.
            train_dataset__sorted_event_str_sequences = [ data__sorted_event_str_sequence for data_name, data__sorted_event_str_sequence\
                                                         in data__sorted_event_str_sequence__dict.items() ]
            countvectorizer.fit( train_dataset__sorted_event_str_sequences )

            # Using the fitted countvectorizer, get the 'Ngram-events counts' for each data
            # -- yes, could have directly utilized "countvecotrizer.fit_transform" in the former-loop.
            #    however, to make sure there is no error in concatenation of 'Ngram-events counts' and 'node-type counts',
            #    just doing the following loop, for explicit connection between 'Ngram-events counts' and 'node-type counts'.
            data__Ngram_events_counts__dict = dict()
            cnt = 1
            for data in train_dataset:
               print(f"{cnt} / {len(dataset)}: {data.name} -- get 'Ngram-events counts'", flush = True)

               # global N-gram
               timestamp_sorted_indices = torch.argsort(data.edge_attr[:, -1], descending=False)
               edge_attr__sorted = data.edge_attr[ timestamp_sorted_indices ]
               taskname_indices = torch.nonzero(edge_attr__sorted[:,:-1], as_tuple=False)[:, -1]
               sorted_event_list = [taskname_colnames[i] for i in taskname_indices]
               sorted_event_str_sequence = " ".join(sorted_event_list) # for compabitiblity with countvecotrizer
               data__Ngram_events_counts = countvectorizer.transform( [ sorted_event_str_sequence] ).toarray().tolist()[0] # use the fitted countvectorizer
               
               data__Ngram_events_counts__dict[ re.search(r'Processed_SUBGRAPH_P3_(.*)\.pickle', data.name).group(1) ] = data__Ngram_events_counts # wrap in list for easy compatiblity later
               cnt+=1

            # Finally, obatin 'data_dict' by concatenating 'data__node_type_count__dict' and 'data__Ngram_events_counts__dict' by matching 'data'
            
            assert set(data__node_type_count__dict.keys()) == set(data__Ngram_events_counts__dict.keys()) , "keys of these two dicts are expected to match"
            data_dict = dict() # integrate after processing the above two separately.
            for data_name in data__node_type_count__dict:
               data_dict[data_name] = data__node_type_count__dict[data_name] + data__Ngram_events_counts__dict[data_name]            


            return data_dict, countvectorizer



      # Function being used for "final-test set"
      elif isinstance(param2, CountVectorizer):

            fitted_countvectorizer = param2

            final_test_dataset = dataset
            data_dict = dict()

            cnt = 1
            for data in final_test_dataset:
               print(f"{cnt} / {len(final_test_dataset)}: {data.name} -- get 'node-type count' and 'Ngram-events counts'", flush = True)
               
               # node-type counts
               data.x = data.x[:,:5]
               nodetype_counts = torch.sum(data.x, axis = 0).tolist()


               # N-gram events counts , using the passed fitted-countvectorizer
               timestamp_sorted_indices = torch.argsort(data.edge_attr[:, -1], descending=False)
               edge_attr__sorted = data.edge_attr[ timestamp_sorted_indices ]
               taskname_indices = torch.nonzero(edge_attr__sorted[:,:-1], as_tuple=False)[:, -1]
               sorted_event_list = [taskname_colnames[i] for i in taskname_indices]
               sorted_event_str_sequence = " ".join(sorted_event_list) # for compabitiblity with countvecotrizer

               data__Ngram_events_counts = fitted_countvectorizer.transform( [ sorted_event_str_sequence] ).toarray().tolist()[0] # use the fitted countvectorizer
               
               
               data_dict[ re.search(r'Processed_SUBGRAPH_P3_(.*)\.pickle', data.name).group(1) ] = nodetype_counts + data__Ngram_events_counts
               cnt+=1

            return data_dict



def get_baseline_2__flattened_graph_Ngram_events_dict( dataset : list, param2 : Union[int, CountVectorizer] ) -> Union[Tuple[dict, CountVectorizer], dict]:

      # Function being used for "train-set"
      if isinstance(param2, int):
         
         N = param2
         countvectorizer = CountVectorizer(ngram_range=(N, N),
                                          max_df= 1.0,
                                          min_df= 1,
                                          max_features= None)   

         train_dataset = dataset

         data__sorted_event_str_sequence__dict = dict()

         cnt = 1
         for data in train_dataset:
            print(f"{cnt} / {len(train_dataset)}: {data.name} -- collect 'sorted event str sequence'", flush = True)

            # collect sorted event str sequence for each data
            timestamp_sorted_indices = torch.argsort(data.edge_attr[:, -1], descending=False)
            edge_attr__sorted = data.edge_attr[ timestamp_sorted_indices ]
            taskname_indices = torch.nonzero(edge_attr__sorted[:,:-1], as_tuple=False)[:, -1]
            sorted_event_list = [taskname_colnames[i] for i in taskname_indices]
            sorted_event_str_sequence = " ".join(sorted_event_list) # for compabitiblity with countvecotrizer
            data__sorted_event_str_sequence__dict[ re.search(r'Processed_SUBGRAPH_P3_(.*)\.pickle', data.name).group(1) ] = sorted_event_str_sequence # wrap in list for easy compatiblity later
            cnt+=1

         # Now fit countvectorizer based on the accumulated 'data__sorted_event_sequence__dict' 
         # and GET the count-vectors
         # -- Need to fit countvectorizer on entire train-set , instead of one by one.  
         train_dataset__sorted_event_str_sequences = [ data__sorted_event_str_sequence for data_name, data__sorted_event_str_sequence\
                                                      in data__sorted_event_str_sequence__dict.items() ]
         train_dataset__count_vectors = countvectorizer.fit_transform( train_dataset__sorted_event_str_sequences ).toarray().tolist()

         assert list(data__sorted_event_str_sequence__dict.keys()) == [ re.search(r'Processed_SUBGRAPH_P3_(.*)\.pickle', data.name).group(1) for data in train_dataset] , "order should match"

         data_dict = dict(zip(list(data__sorted_event_str_sequence__dict.keys()), train_dataset__count_vectors))


         return data_dict, countvectorizer


      # Function being used for "final-test set"
      elif isinstance(param2, CountVectorizer):

         fitted_countvectorizer = param2

         final_test_dataset = dataset
         data_dict = dict()

         cnt = 1
         for data in final_test_dataset:
            print(f"{cnt} / {len(final_test_dataset)}: {data.name} -- get 'Ngram-events counts'", flush = True)

            # N-gram events counts , using the passed fitted-countvectorizer
            timestamp_sorted_indices = torch.argsort(data.edge_attr[:, -1], descending=False)
            edge_attr__sorted = data.edge_attr[ timestamp_sorted_indices ]
            taskname_indices = torch.nonzero(edge_attr__sorted[:,:-1], as_tuple=False)[:, -1]
            sorted_event_list = [taskname_colnames[i] for i in taskname_indices]
            sorted_event_str_sequence = " ".join(sorted_event_list) # for compabitiblity with countvecotrizer

            data__Ngram_events_counts = fitted_countvectorizer.transform( [ sorted_event_str_sequence] ).toarray().tolist()[0] # use the fitted countvectorize

            data_dict[ re.search(r'Processed_SUBGRAPH_P3_(.*)\.pickle', data.name).group(1) ] = data__Ngram_events_counts
            cnt+=1
         return data_dict



def get_baseline_1__simple_counting_dict( dataset : list ) -> dict:

   data_dict = dict()

   cnt = 1

   for data in dataset:
      print(f"{cnt} / {len(dataset)}: {data.name} -- generate graph-embedding", flush = True)

      nodetype_counts = torch.sum(data.x[:,:5], axis = 0)

      data.edge_attr = data.edge_attr[:,:-1] # drop time-scalar
      events_counts = torch.sum(data.edge_attr, dim = 0)

      nodetype_events_counts = torch.cat(( nodetype_counts, events_counts ), dim = 0)

      data_dict[ re.search(r'Processed_SUBGRAPH_P3_(.*)\.pickle', data.name).group(1) ] = nodetype_events_counts.tolist()
      cnt+=1

   return data_dict


