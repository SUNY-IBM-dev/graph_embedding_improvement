import os
import pickle
import torch
# import numpy as np
from torch_geometric.data import Data

class LoadGraphs:

    """
    Data Loader class for the graphs
    code works for a graph classification task only
    """

    def __init__(self):
        #add by Meng 4/5/22: not repeatedly load samples of the same size
        #self.data = set()
        #end
        return

    def load_pickle(self, filename):
        """
        loads a pickle

        Args
           filename (str): path to file

        Returns
           data : the pickled data in list format
        """
        data = None
        with open(filename, 'rb') as fp:
            try:
                data = pickle.load(fp)
                return data
            except pickle.UnpicklingError:
            #except (pickle.UnpicklingError, EOFError) as error:
            #    print(error)
                return None

    def parse_single_graph(self, file_path, num_node_attr=1022, num_edge_attr=80, drop_dim_edge_timescalar = False):
        """
        parses a single graph into pytorch-geometric format
        loads the pre-processed graph data

        Args
           file_path (str): abs. file path

        Returns:
           {x, edge_list, y, edge_attr}
        """
        _name = file_path.split("/")[-1]
        data = self.load_pickle(file_path)
        if data is None:
            #print(">>> Error in sample", _name)
            print(">>> pickle.UnpicklingError for sample", _name)
            return -1, -1, -1, -1, -1
        
        x, y, edge_attr, edge_list = data['x'], data['y'], data['edge_attr'], data['edge_list']
        len_x = len(x)
        len_edg = len(edge_list[0])
        # skip extremely large graphs
        if len(x) > 400000:
            print(_name, ">>> #nodes:", len(x), " #edges:", len(edge_attr), " | sample skipped!")
            return -1, -1, -1, -1, -1

        #add by Meng 4/25/22: only load data that have more than 100 nodes
        #add by JY 6/22/22: only load data that have more than 10 nodes  per GIN-experiments excel sheet.
        num_nodes = 10
        if len(x) < num_nodes:
            print(_name, ">>> #nodes:", len(x), " #edges:", len(edge_attr), " | sample skipped!")
            return -1, -1, -1, -1, -1
        #end 
        
        # check for num node attributes
        for _x in x:
            if len(_x) != num_node_attr:
                print(_name, ">>> #node attributes are mismatched, ", len(_x), ' | sample skipped!')
                return -1, -1, -1, -1, -1

        # check for num edge attributes
        for _y in edge_attr:
            if len(_y) != num_edge_attr:
                print(_name, ">>>> #edge attributes mismatched, ", len(_y), " | sample skipped!")
                return -1, -1, -1, -1, -1

        '''#add by Meng 4/5/22: not repeatedly load samples of the same size
        if (len_x, len_edg) in self.data:
            print(_name, ">>> #nodes:", len(x), " #edges:", len(edge_attr), " | sample skipped!")
            return -1, -1, -1, -1, -1
        self.data.add((len_x,len_edg))
        #end'''
            
        # convert all data into tensors
        x = torch.tensor(x, dtype=torch.float)
        y = torch.tensor(y, dtype=torch.long)
        edge_list = torch.tensor(edge_list, dtype=torch.long)
        if drop_dim_edge_timescalar == True:    # Added by JY @ 2023-07-06
            edge_attr = torch.tensor(edge_attr, dtype=torch.float)
            edge_attr = edge_attr[:,:-1]    # drop the last element which is the time-scalar!
        else:
            edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        print("> ", _name, "| #node: ", len_x, " #edge: ", len_edg)

        return x, edge_list, y, edge_attr, _name



    # #####################################################################################################
    # # Added by JY @ 2023-06-08
    # def parse_single_graph_no_edge_feat(self, file_path, num_node_attr=1022, num_edge_attr=None):
    #     """
    #     parses a single graph into pytorch-geometric format
    #     loads the pre-processed graph data

    #     Args
    #        file_path (str): abs. file path

    #     Returns:
    #        {x, edge_list, y, edge_attr}
    #     """
    #     _name = file_path.split("/")[-1]
    #     data = self.load_pickle(file_path)
    #     if data is None:
    #         #print(">>> Error in sample", _name)
    #         print(">>> pickle.UnpicklingError for sample", _name)
    #         return -1, -1, -1, -1, -1
    #     # JY @ 2023-06-08: returned edge_attr will be None        
    #     x, y, edge_attr, edge_list = data['x'], data['y'], data['edge_attr'], data['edge_list']
    #     len_x = len(x)
    #     len_edg = len(edge_list[0])
    #     # skip extremely large graphs
    #     if len(x) > 400000:
    #         print(_name, ">>> #nodes:", len(x), " | sample skipped!")
    #         return -1, -1, -1, -1, -1

    #     #add by Meng 4/25/22: only load data that have more than 100 nodes
    #     #add by JY 6/22/22: only load data that have more than 10 nodes  per GIN-experiments excel sheet.
    #     num_nodes = 10
    #     if len(x) < num_nodes:
    #         print(_name, ">>> #nodes:", len(x), " | sample skipped!")
    #         return -1, -1, -1, -1, -1
    #     #end 
        
    #     # check for num node attributes
    #     for _x in x:
    #         if len(_x) != num_node_attr:
    #             print(_name, ">>> #node attributes are mismatched, ", len(_x), ' | sample skipped!')
    #             return -1, -1, -1, -1, -1

    #     # JY @ 2023-06-08
    #     # # check for num edge attributes
    #     # for _y in edge_attr:
    #     #     if len(_y) != num_edge_attr:
    #     #         print(_name, ">>>> #edge attributes mismatched, ", len(_y), " | sample skipped!")
    #     #         return -1, -1, -1, -1, -1

    #     '''#add by Meng 4/5/22: not repeatedly load samples of the same size
    #     if (len_x, len_edg) in self.data:
    #         print(_name, ">>> #nodes:", len(x), " #edges:", len(edge_attr), " | sample skipped!")
    #         return -1, -1, -1, -1, -1
    #     self.data.add((len_x,len_edg))
    #     #end'''
            
    #     # convert all data into tensors
    #     x = torch.tensor(x, dtype=torch.float)
    #     y = torch.tensor(y, dtype=torch.long)
    #     edge_list = torch.tensor(edge_list, dtype=torch.long)
    #     edge_attr = None # JY @ 2023-06-08

    #     print("> ", _name, "| #node: ", len_x, " #edge: ", len_edg)

    #     return x, edge_list, y, edge_attr, _name
    # #####################################################################################################



    def parse_all_data(self, load_path, num_node_attr, num_edge_attr, drop_dim_edge_timescalar):
        """
        Parses all data samples in 'load_path'

        Args
           load_path (str): the path to load all pickled samples

        Returns
           dataset (list): list of pytorch-geometric Data samples
        """
        dir_contents = os.listdir(load_path)
        
        dataset = []  # will store the datset here
        # loop through all samples and parse
        for idx, filename in enumerate(dir_contents):

            # sanity check
            if '_Sample_' not in filename and '_SUBGRAPH_' not in filename:
                continue

            # for this initial test only load 50 samples at most for Benign or Malware
            # if idx > 100:
            #     break
            
            _path = load_path + "/" + filename  # path to the pickled file
            # modified code to have edge attributes
            
            if num_edge_attr:            
                x, edge_list, y, edge_attr, name = self.parse_single_graph(_path, 
                                                                           num_node_attr=num_node_attr, 
                                                                           num_edge_attr=num_edge_attr,
                                                                           drop_dim_edge_timescalar = drop_dim_edge_timescalar)

                if isinstance(x, int) and x == -1:
                    continue
                # Data(x=x, edge_index=edge_list, edge_weight=edge_weight, y=y)
                dataset.append(Data(x=x, edge_index=edge_list, edge_attr=edge_attr, y=y, name=name))


            # if no edge-feat (num_dge_attr == None)
            else:
                # returned 'edge_attr' is None
                x, edge_list, y, edge_attr, name = self.parse_single_graph_no_edge_feat(_path, num_node_attr=num_node_attr, num_edge_attr=None)

                if isinstance(x, int) and x == -1:
                    continue
                # Data(x=x, edge_index=edge_list, edge_weight=edge_weight, y=y)
                dataset.append(Data(x=x, edge_index=edge_list, edge_attr=edge_attr, y=y, name=name))


        return dataset


    #####################################################################################################


    # def parse_all_data__expand_dim_node_compatible_with_DG(self, 
    #                                                     load_path, 
    #                                                     num_node_attr, 
    #                                                     num_edge_attr):
        
    #     ##############################################################################################
    #     ''' 2023-06-29 '''
    #     # Parse all data in a way that is compatible to the Dynamic Graph scheme.
    #     #   Expand "x" (node-feature-vector) from 5 bit vector to 5+71 bit-vector.
    #     #   This is similar to edge-feat-migrate, but does not actually migrate the edge-features.
    #     #   (so this scheme is not compatible with edge-feat-migrated data)
    #     #   Under this scheme, "x" will start as a concatenation of original "x" and 71-length zero-vector 
    #     #   (71 corresponds to edge-feats after dropping time-info scalar)
    #     #   The 71-length zero-vector will be used to receive information, during the course of information-flow.
    #     ##############################################################################################
        
    #     """
    #     Parses all data samples in 'load_path'

    #     Args
    #        load_path (str): the path to load all pickled samples

    #     Returns
    #        dataset (list): list of pytorch-geometric Data samples
    #     """
    #     dir_contents = os.listdir(load_path)
        
    #     dataset = []  # will store the datset here
    #     # loop through all samples and parse
    #     for idx, filename in enumerate(dir_contents):

    #         # sanity check
    #         if '_Sample_' not in filename and '_SUBGRAPH_' not in filename:
    #             continue

    #         # for this initial test only load 50 samples at most for Benign or Malware
    #         # if idx > 100:
    #         #     break
            
    #         _path = load_path + "/" + filename  # path to the pickled file
    #         # modified code to have edge attributes
    #         x, edge_list, y, edge_attr, name = self.parse_single_graph(_path, num_node_attr=num_node_attr, num_edge_attr=num_edge_attr)

    #         # Expand "x" (node-feature-vector) from 5 bit vector to 5+71 bit-vector,
    #         # for compatibility with Dynamic-Graph scheme
    #         length_of_EdgeFeatVec_after_dropping_time_scalar = edge_attr.size()[1] - 1
    #         zero_vectors = torch.zeros(x.shape[0], length_of_EdgeFeatVec_after_dropping_time_scalar)
    #         expanded_x = torch.cat((x, zero_vectors), dim=1)



    #         edge_attr.size()[1] - 1

    #         if isinstance(x, int) and x == -1:
    #             continue
    #         # Data(x=x, edge_index=edge_list, edge_weight=edge_weight, y=y)
    #         dataset.append(Data(x= expanded_x, 
    #                             edge_index=edge_list, 
    #                             edge_attr=edge_attr, 
    #                             y=y, 
    #                             name=name))

    #     return dataset






    # def parse_all_data__expand_dim_node_compatible_with_SignalAmplification(self, 
    #                                                                         load_path, 
    #                                                                         num_node_attr, 
    #                                                                         num_edge_attr,
    #                                                                         signal_amplification_option: str,                                                                            
    #                                                                         ):
        
    #     ##############################################################################################
    #     ''' 2023-06-29 '''
    #     # Parse all data in a way that is compatible to the Dynamic Graph scheme.
    #     #   Expand "x" (node-feature-vector) from 5 bit vector to 5+71 bit-vector.
    #     #   This is similar to edge-feat-migrate, but does not actually migrate the edge-features.
    #     #   (so this scheme is not compatible with edge-feat-migrated data)
    #     #   Under this scheme, "x" will start as a concatenation of original "x" and 71-length zero-vector 
    #     #   (71 corresponds to edge-feats after dropping time-info scalar)
    #     #   The 71-length zero-vector will be used to receive information, during the course of information-flow.
    #     ##############################################################################################
        
    #     """
    #     Parses all data samples in 'load_path'

    #     Args
    #        load_path (str): the path to load all pickled samples

    #     Returns
    #        dataset (list): list of pytorch-geometric Data samples
    #     """
    #     dir_contents = os.listdir(load_path)
        
    #     dataset = []  # will store the datset here
    #     # loop through all samples and parse
    #     for idx, filename in enumerate(dir_contents):

    #         # sanity check
    #         if '_Sample_' not in filename and '_SUBGRAPH_' not in filename:
    #             continue

    #         # for this initial test only load 50 samples at most for Benign or Malware
    #         # if idx > 100:
    #         #     break
            
    #         _path = load_path + "/" + filename  # path to the pickled file
    #         # modified code to have edge attributes
    #         x, edge_list, y, edge_attr, name = self.parse_single_graph(_path, 
    #                                                                    num_node_attr=num_node_attr, 
    #                                                                    num_edge_attr=num_edge_attr)


    #         length_of_EdgeFeatVec_after_dropping_time_scalar = edge_attr.size()[1] - 1 # -1 to drop time-scalar

    #         if signal_amplification_option == "signal_amplified__event_1gram":
    #             # Tricky-case.
    #             # Expand "x" to 71 bit-vector BUT preserve the first 5 bits (node-type) since needed in message-passing.
    #             nodetype_5bit_x = x[:,:5]
    #             pad_dim = length_of_EdgeFeatVec_after_dropping_time_scalar - 5
    #             pad_zero_vectors =  torch.zeros(x.shape[0], pad_dim)
    #             expanded_x = torch.cat((nodetype_5bit_x, pad_zero_vectors), dim=1)


    #         elif signal_amplification_option == "signal_amplified__event_1gram_nodetype_5bit":
    #             # Expand "x" to 5+71 bit-vector, for compatiblity with "signal_amplified__event_1gram_nodetype_5bit"
    #             nodetype_5bit_x = x[:,:5]
    #             event_1gram_placeholder__zero_vectors =  torch.zeros(x.shape[0], length_of_EdgeFeatVec_after_dropping_time_scalar)
    #             expanded_x = torch.cat((nodetype_5bit_x, event_1gram_placeholder__zero_vectors), dim=1)

    #         elif signal_amplification_option == "signal_amplified__event_1gram_nodetype_5bit_and_Ahoc_Identifier":
    #             # Expand "x" to 46+71 bit-vector, for compatiblity with "signal_amplified__event_1gram_nodetype_5bit_and_Ahoc_Identifier"
    #             nodetype_5bit_and_Ahoc_Identifier_x = x
    #             event_1gram_placeholder__zero_vectors =  torch.zeros(x.shape[0], length_of_EdgeFeatVec_after_dropping_time_scalar)
    #             expanded_x = torch.cat((nodetype_5bit_and_Ahoc_Identifier_x, event_1gram_placeholder__zero_vectors), dim=1)


    #         if isinstance(x, int) and x == -1:
    #             continue
    #         # Data(x=x, edge_index=edge_list, edge_weight=edge_weight, y=y)
    #         dataset.append(Data(x= expanded_x, 
    #                             edge_index= edge_list, 
    #                             edge_attr= edge_attr, 
    #                             y= y, 
    #                             name= name))


    #     return dataset



    # Added by JY @ 2024-1-2
    def parse_all_data__local_ngram__standard_message_pasing(self, load_path, 
                                                             num_node_attr, 
                                                             num_edge_attr,
                                                             drop_dim_edge_timescalar):
        """
        Parses all data samples in 'load_path'

        Args
           load_path (str): the path to load all pickled samples

        Returns
           dataset (list): list of pytorch-geometric Data samples
        """
        dir_contents = os.listdir(load_path)
        
        dataset = []  # will store the datset here
        # loop through all samples and parse
        for idx, filename in enumerate(dir_contents):

            # sanity check
            if '_Sample_' not in filename and '_SUBGRAPH_' not in filename:
                continue

            # for this initial test only load 50 samples at most for Benign or Malware
            # if idx > 100:
            #     break
            
            _path = load_path + "/" + filename  # path to the pickled file
            # modified code to have edge attributes
            
            x, edge_list, y, edge_attr, name = self.parse_single_graph__local_ngram__standard_message_pasing(_path, 
                                                                        num_node_attr=num_node_attr, 
                                                                        num_edge_attr=num_edge_attr,
                                                                        drop_dim_edge_timescalar = drop_dim_edge_timescalar)

            if isinstance(x, int) and x == -1:
                continue
            # Data(x=x, edge_index=edge_list, edge_weight=edge_weight, y=y)
            dataset.append(Data(x=x, edge_index=edge_list, edge_attr=edge_attr, y=y, name=name))



        return dataset    


    def parse_single_graph__local_ngram__standard_message_pasing(self, 
                                                                 file_path, 
                                                                 num_node_attr=1022, 
                                                                 num_edge_attr=80, 
                                                                 drop_dim_edge_timescalar = False):
        """
        parses a single graph into pytorch-geometric format
        loads the pre-processed graph data

        Args
           file_path (str): abs. file path

        Returns:
           {x, edge_list, y, edge_attr}
        """
        _name = file_path.split("/")[-1]
        data = self.load_pickle(file_path)
        if data is None:
            #print(">>> Error in sample", _name)
            print(">>> pickle.UnpicklingError for sample", _name)
            return -1, -1, -1, -1, -1
        
        x, y, edge_attr, edge_list = data['x'], data['y'], data['edge_attr'], data['edge_list']
        len_x = len(x)
        len_edg = len(edge_list[0])
        # skip extremely large graphs
        if len(x) > 400000:
            print(_name, ">>> #nodes:", len(x), " #edges:", len(edge_attr), " | sample skipped!")
            return -1, -1, -1, -1, -1

        #add by Meng 4/25/22: only load data that have more than 100 nodes
        #add by JY 6/22/22: only load data that have more than 10 nodes  per GIN-experiments excel sheet.
        num_nodes = 10
        if len(x) < num_nodes:
            print(_name, ">>> #nodes:", len(x), " #edges:", len(edge_attr), " | sample skipped!")
            return -1, -1, -1, -1, -1
        #end 
        
        # check for num node attributes
        for _x in x:
            if len(_x) != num_node_attr:
                print(_name, ">>> #node attributes are mismatched, ", len(_x), ' | sample skipped!')
                return -1, -1, -1, -1, -1

        # # check for num edge attributes
        # for _y in edge_attr:
        #     if len(_y) != num_edge_attr:
        #         print(_name, ">>>> #edge attributes mismatched, ", len(_y), " | sample skipped!")
        #         return -1, -1, -1, -1, -1

        '''#add by Meng 4/5/22: not repeatedly load samples of the same size
        if (len_x, len_edg) in self.data:
            print(_name, ">>> #nodes:", len(x), " #edges:", len(edge_attr), " | sample skipped!")
            return -1, -1, -1, -1, -1
        self.data.add((len_x,len_edg))
        #end'''
            
        # convert all data into tensors
        x = torch.tensor(x, dtype=torch.float)
        y = torch.tensor(y, dtype=torch.long)
        edge_list = torch.tensor(edge_list, dtype=torch.long)
        # if drop_dim_edge_timescalar == True:    # Added by JY @ 2023-07-06
        #     edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        #     edge_attr = edge_attr[:,:-1]    # drop the last element which is the time-scalar!
        # else:
        #     edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        print("> ", _name, "| #node: ", len_x, " #edge: ", len_edg)

        # JY @ 2024-1-2: Just pass "edge_attr" as it is, list of strings (event-names time-sorted strings) with variable-length
        #                No problem making a pytorch-geoemtric Data object out of it
        return x, edge_list, y, edge_attr, _name
