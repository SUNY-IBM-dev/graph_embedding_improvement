import torch
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import re

#============================================================================================================================================

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

#============================================================================================================================================

##########################################################################################################################################################
##########################################################################################################################################################
##########################################################################################################################################################

''' Trial_7__Thread_level_N_grams__N_gt_than_1__Similar_to_PriorGraphEmbedding '''

def pretrain__countvectorizer_on_training_set__before_graph_embedding_generation( 
                                                                                 dataset : list, 
                                                                                 Ngram : int = 4,
                                                                                 only_train_specified_Ngram = False,
                                                                                  ) -> list:
    
      ''' TODO
      start writing pretraining a n-gram countvectorizer on all thread-level event-sequences from every subgraph in entire training set, 
      PRIOR to peforming graph-embedding generation (signal amplication) -- 
      this is similar to what has been done in local ngram standard message passing -- 
      and a reasonable compromiziation althoguh in K-fold CV hypeparameter tuning context strictly speaking should be fitting the countvectorizer of K-1 training-folds
         -- but this will be too much

      # refer to : https://github.com/SUNY-IBM-dev/graph_embedding_improvement/blob/baee25391d90c9631e97f38fa84f1e13ba718cf5/graph_embedding_improvement_efforts/Trial_4__Local_N_gram__standard_message_passing/local_ngram__standard_message_passing__graph_embedding.py#L1277
         
      '''

      if Ngram <= 1:
         ValueError("Ngram should be greater than 1 for this graph-embedding approach.\n-> AsiaCCS submission already handled thread-level 1gram event distirubtion")

      pretrained_countvectorizers_list =[]

      thread_node_tensor = torch.tensor([0, 0, 0, 0, 1])


      graph__to__thread_level_sorted_event_sequences_dict = dict() # Added by JY @ 2024-1-20


      cnt = 1
      for data in dataset:
            
            print(f"{cnt} / {len(dataset)}: {data.name}  |  in 'pretrain__countvectorizer_on_training_set__before_graph_embedding_generation()'", flush = True)

            # Added by JY @ 2023-07-18 to handle node-attr dim > 5  
            # if data.x.shape[1] != 5:
            data_x_first5 = data.x[:,:5]

            thread_node_indices = torch.nonzero(torch.all(torch.eq( data_x_first5, thread_node_tensor ), dim=1), as_tuple=False).flatten().tolist()


            graph__to__thread_level_sorted_event_sequences_dict[data.name] = []

            for thread_node_idx in thread_node_indices:

               edge_src_indices = data.edge_index[0]
               edge_tar_indices = data.edge_index[1]

               # which this node is a target-node (outgoing-edge w.r.t this node)
               outgoing_edges_from_thread_node_idx = torch.nonzero( edge_src_indices == thread_node_idx ).flatten()
               # which this node is a target-node (incoming-edge w.r.t this node)
               incoming_edges_to_thread_node_idx = torch.nonzero( edge_tar_indices == thread_node_idx ).flatten()

               # Following is to deal with edge-attr (event-dist & time-scalar) -------------------------------------------------------------------------------------
               edge_attr_of_incoming_edges_from_thread_node_idx = data.edge_attr[incoming_edges_to_thread_node_idx]
               edge_attr_of_outgoing_edges_from_thread_node_idx = data.edge_attr[outgoing_edges_from_thread_node_idx]



               ''' JY @ 2024-1-20: Get thread-level event-sequence sorted by timestamps '''

               edge_attr_of_both_direction_edges_from_thread_node_idx = torch.cat([edge_attr_of_incoming_edges_from_thread_node_idx, 
                                                                                   edge_attr_of_outgoing_edges_from_thread_node_idx],
                                                                                   dim = 0)


               timestamp_sorted_indices = torch.argsort(edge_attr_of_both_direction_edges_from_thread_node_idx[:, -1], descending=False)

               edge_attr_of_both_direction_edges_from_thread_node_idx__sorted = edge_attr_of_both_direction_edges_from_thread_node_idx[ timestamp_sorted_indices ]

               taskname_indices = torch.nonzero(edge_attr_of_both_direction_edges_from_thread_node_idx__sorted[:,:-1], as_tuple=False)[:, -1]


               # Replace tensor elements with corresponding string values efficiently
               thread_sorted_event_sequence = [taskname_colnames[i] for i in taskname_indices]
               graph__to__thread_level_sorted_event_sequences_dict[data.name].append( thread_sorted_event_sequence )
   
               # --------------------------------------------------------------------------------------------------------------------------------------------------------------------------

            # data_dict[ data.name.lstrip("Processed_SUBGRAPH_P3_").rstrip(".pickle") ] = data_thread_node_both_direction_edges_edge_attrs.tolist()
            cnt+=1

      # Now apply countvectorizer
      # https://github.com/SUNY-IBM-dev/graph_embedding_improvement/blob/c14d7631f95a24d5e0c192d7075a184698af1e13/graph_embedding_improvement_efforts/Trial_4__Local_N_gram__standard_message_passing/local_ngram__standard_message_passing__graph_embedding.py#L1277

      thread_level_sorted_event_sequences__nested_list = [ list_of_thread_event_sequences for data_name, list_of_thread_event_sequences in graph__to__thread_level_sorted_event_sequences_dict.items() ]

      def flatten_list(nested_list):
         return [item for sublist in nested_list for item in sublist]

      all_thread_level_sorted_event_lists = flatten_list( thread_level_sorted_event_sequences__nested_list )



      # https://github.com/SUNY-IBM-dev/graph_embedding_improvement/blob/c14d7631f95a24d5e0c192d7075a184698af1e13/N_gram_hyperparameter_tuning/n_gram_hypeparam_tuning.py#L958
      # for compatiblity with countvectorizer
      # [ 'Threadstart Create Write Create Threadstop',
      #   'Threadstart OpenKey Create CloseKey Close Open Threadstop',
      #     ... ]
      if only_train_specified_Ngram:
         # Only fit for the specified Ngram countvectorizer
         #     e.g. If N == 4, then drop all event-sequences that have less than 4 events, 
         #          because since can't fit a 4gram countvectorizer out of such evnet-sequences.
         all_thread_level_sorted_event_str_sequences = \
            [ ' '.join(thread_sorted_event_list) for thread_sorted_event_list in all_thread_level_sorted_event_lists\
              if len(thread_sorted_event_list) >= Ngram ]

         print(f"Specified Ngram: {Ngram} | only_train_specified_Ngram: {only_train_specified_Ngram}", flush=True)
         countvectorizer = CountVectorizer(ngram_range=(Ngram, Ngram),
                                           max_df= 1.0,
                                           min_df= 1,
                                           max_features= None)   
         countvectorizer.fit( all_thread_level_sorted_event_str_sequences )

         print(f"Fitted {Ngram}-gram countvectorizer", flush=True)


         pretrained_countvectorizers_list.append( countvectorizer )


      else:
         # Fit multiple countvectorizers, to account for event-sequences that have less than than the number of specified Ngram
         #     e.g. Observed a good number of threads having small number of events (some have only 1; such as threadstop)
         #          To account for such cases, not only fit countvectorzier for specified Ngram, 
         #          but fit Ngram countvecotrizer, N-1 gram countvecotrizer, .. , 1 gram countvecotrizer.
         print(f"Specified Ngram: {Ngram} | only_train_specified_Ngram: {only_train_specified_Ngram}", flush=True)

         for N in list(range(1, Ngram+1)):

            # if N > 1:
            #    # max_df and min_df here are necessary and values are conventionally used ones
            #    # necessity is because , say, 61 C 4 will result in 61! / (61-4)! * 4! which is unreasonably large, 
            #    # so necesary to filter out features based on frequency 
            #    # intuition is meaningful ngram features should not appear too little or too frequent
            #    countvectorizer = CountVectorizer(ngram_range=(N, N), 
            #                                     max_df= 0.3, 
            #                                     min_df= 10, 
            #                                     max_features= None )  # ngram [use 4-gram or 8-gram] 
            # else:
            #    # N = 1 <-- we don't want to drop any 1-gram features since we don't have feature-space too big problem
            #    #           Following does not ignore any n-gram feautres
            #    countvectorizer = CountVectorizer(ngram_range=(N, N),
            #                                        max_df= 1.0,
            #                                        min_df= 1,
            #                                        max_features= None)


            all_thread_level_sorted_event_str_sequences = \
               [ ' '.join(thread_sorted_event_list) for thread_sorted_event_list in all_thread_level_sorted_event_lists\
               if len(thread_sorted_event_list) >= N ]

            if len(all_thread_level_sorted_event_str_sequences) == 0:
                print(f"None of the thread-level event-sequences have length greater or equal than {N} -- so continue")
                # this is very unlikely though
                continue

            countvectorizer = CountVectorizer(ngram_range=(N, N),
                                              max_df= 1.0,
                                              min_df= 1,
                                              max_features= None)   
            countvectorizer.fit( all_thread_level_sorted_event_str_sequences )
            print(f"Fitted {N}-gram countvectorizer", flush=True)

            pretrained_countvectorizers_list.append( countvectorizer )


      



      return pretrained_countvectorizers_list

# JY @ 2024-1-20: thread-level N>1 gram events + nodetype-5bits
def get_thread_level_N_gram_events_adjacent_5bit_dist_dict( 
                                                            pretrained_Ngram_countvectorizer_list : list, # TODO        
                                                            dataset : list, 
                                                            pool : str = "sum",
                                                              ):
      
      ''' JY @ 2024-1-20 : Implement this '''


      thread_node_tensor = torch.tensor([0, 0, 0, 0, 1])

      data_dict = dict()

      cnt = 1
      for data in dataset:
            
            print(f"{cnt} / {len(dataset)}: {data.name} -- generate graph-embedding", flush = True)

            # Added by JY @ 2023-07-18 to handle node-attr dim > 5  
            # if data.x.shape[1] != 5:
            data_x_first5 = data.x[:,:5]
            thread_node_indices = torch.nonzero(torch.all(torch.eq( data_x_first5, thread_node_tensor), dim=1), as_tuple=False).flatten().tolist()

            # which this node is a source-node (outgoing-edge w.r.t this node)
  


            data_thread_node_all_unique_adjacent_nodes_5bit_dists = torch.tensor([]) # Added by JY @ 2023-07-19

            data_thread_level_all_Ngram_features = torch.tensor([]) # Added by JY @ 2023-01-20

            for thread_node_idx in thread_node_indices:

               edge_src_indices = data.edge_index[0]
               edge_tar_indices = data.edge_index[1]

               # which this node is a target-node (outgoing-edge w.r.t this node)
               outgoing_edges_from_thread_node_idx = torch.nonzero( edge_src_indices == thread_node_idx ).flatten()
               # which this node is a target-node (incoming-edge w.r.t this node)
               incoming_edges_to_thread_node_idx = torch.nonzero( edge_tar_indices == thread_node_idx ).flatten()

               # Following is to deal with edge-attr (event-dist & time-scalar) -------------------------------------------------------------------------------------
               edge_attr_of_incoming_edges_from_thread_node_idx = data.edge_attr[incoming_edges_to_thread_node_idx]
               edge_attr_of_outgoing_edges_from_thread_node_idx = data.edge_attr[outgoing_edges_from_thread_node_idx]



               ''' JY @ 2024-1-20: Get thread-level event-sequence sorted by timestamps '''
               # Refered to : https://github.com/SUNY-IBM-dev/graph_embedding_improvement/blob/20627016d59466d3dad191ff208efce97b15d35e/graph_embedding_improvement_efforts/Trial_7__Thread_level_N_grams__N_gt_than_1__Similar_to_PriorGraphEmbedding/thread_level_n_gram__n_gt_1__similar_to_asiaccs_graph_embedding.py#L483C1-L491C95

               edge_attr_of_both_direction_edges_from_thread_node_idx = torch.cat([edge_attr_of_incoming_edges_from_thread_node_idx, 
                                                                                   edge_attr_of_outgoing_edges_from_thread_node_idx],
                                                                                   dim = 0)


               timestamp_sorted_indices = torch.argsort(edge_attr_of_both_direction_edges_from_thread_node_idx[:, -1], descending=False)

               edge_attr_of_both_direction_edges_from_thread_node_idx__sorted = edge_attr_of_both_direction_edges_from_thread_node_idx[ timestamp_sorted_indices ]

               taskname_indices = torch.nonzero(edge_attr_of_both_direction_edges_from_thread_node_idx__sorted[:,:-1], as_tuple=False)[:, -1]


               thread_sorted_event_list = [taskname_colnames[i] for i in taskname_indices]

               thread_sorted_event_str_sequence = " ".join(thread_sorted_event_list) # for compabitiblity with countvecotrizer

               # transform and get count -- https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html#sklearn.feature_extraction.text.CountVectorizer.transform
               # Refer to: https://github.com/SUNY-IBM-dev/graph_embedding_improvement/blob/20627016d59466d3dad191ff208efce97b15d35e/graph_embedding_improvement_efforts/Trial_4__Local_N_gram__standard_message_passing/local_ngram__standard_message_passing__graph_embedding.py#L760C41-L760C72

               # Get all Ngrams (N could be multiple, depending on 'only_train_specified_Ngram' parameter value)
               thread_all_Ngram_counts__appended_nparray = np.array([]) # torch.Tensor()
               for pretrained_Ngram_countvectorizer in pretrained_Ngram_countvectorizer_list:
                  # If multiple countvectorizers, supposed to start from 1gram to Ngram
                  #print(f"pretrained_Ngram_countvectorizer.ngram_range: {pretrained_Ngram_countvectorizer.ngram_range}", flush= True)
                  thread_Ngram_counts__portion = pretrained_Ngram_countvectorizer.transform( [ thread_sorted_event_str_sequence ] ).toarray() # needs to be in a list
                  thread_all_Ngram_counts__appended_nparray = np.append(thread_all_Ngram_counts__appended_nparray, thread_Ngram_counts__portion )           
               #print("\n")

               # JY @ 2024-1-20: so that can stack on 'data_thread_level_all_Ngram_features' for all thread's Ngram features within this subgrpah(data)
               thread_all_Ngram_counts__appended_tensor = torch.Tensor(thread_all_Ngram_counts__appended_nparray).view(1,-1) # for Size([1,edge_feat_len])
               data_thread_level_all_Ngram_features = torch.cat((data_thread_level_all_Ngram_features, thread_all_Ngram_counts__appended_tensor), dim=0)                  



               # ==============================================================================================================================================
               # Node-level

               # But also need to consider the multi-graph aspect here. 
               # So here, do not count twice for duplicate adjacent nodes due to multigraph.
               # Just need to get the distribution.
               # Find unique column-wise pairs (dropping duplicates - src/dst pairs that come from multi-graph)
               unique_outgoing_edges_from_thread_node, _ = torch.unique( data.edge_index[:, outgoing_edges_from_thread_node_idx ], dim=1, return_inverse=True)

               # Find unique column-wise pairs (dropping duplicates - src/dst pairs that come from multi-graph)
               unique_incoming_edges_to_thread_node, _ = torch.unique( data.edge_index[:, incoming_edges_to_thread_node_idx ], dim=1, return_inverse=True)

               target_nodes_of_outgoing_edges_from_thread_node = unique_outgoing_edges_from_thread_node[1] # edge-target is index 1
               source_nodes_of_incoming_edges_to_thread_node = unique_incoming_edges_to_thread_node[0] # edge-src is index 0


               #-- Option-1 --------------------------------------------------------------------------------------------------------------------------------------------------------------
               # # Already handled multi-graph case, but how about the bi-directional edge case?
               # # For, T-->F and T<--F, information of F will be recorded twice."Dont Let it happen"
               unique_adjacent_nodes_of_both_direction_edges_of_thread_node = torch.unique( torch.cat( [ target_nodes_of_outgoing_edges_from_thread_node, 
                                                                                                         source_nodes_of_incoming_edges_to_thread_node ] ) )
               integrated_5bit_of_all_unique_adjacent_nodes_to_thread = torch.sum( data_x_first5[unique_adjacent_nodes_of_both_direction_edges_of_thread_node], dim = 0 )
               
               data_thread_node_all_unique_adjacent_nodes_5bit_dists = torch.cat(( data_thread_node_all_unique_adjacent_nodes_5bit_dists,
                                                                                   integrated_5bit_of_all_unique_adjacent_nodes_to_thread.unsqueeze(0) ), dim = 0)
               # --------------------------------------------------------------------------------------------------------------------------------------------------------------------------


            # [ node-type5bit + N>gram events ]
            thread_level_N_gram_events_adjacent_5bit_dist = torch.cat( [data_thread_node_all_unique_adjacent_nodes_5bit_dists, 
                                                                        data_thread_level_all_Ngram_features, 
                                                                        ], dim = 1)
            
            
            
            ''' JY @ 2024-1-20: Now "pool" all thead-node's embedding, to generate the "graph embedding". '''
            if pool == "sum": pool__func = torch.sum
            elif pool == "mean": pool__func = torch.mean

            graph_embedding = pool__func(thread_level_N_gram_events_adjacent_5bit_dist, dim = 0)            
            
            
            
            data_dict[ re.search(r'Processed_SUBGRAPH_P3_(.*?)\.pickle', data.name).group(1) ] = graph_embedding.tolist()

            # data_dict[ data.name.lstrip("Processed_SUBGRAPH_P3_").rstrip(".pickle") ] = data_thread_node_both_direction_edges_edge_attrs.tolist()
            cnt+=1
      return data_dict
##########################################################################################################################################################
##########################################################################################################################################################
##########################################################################################################################################################