import sys

import ast
import networkx as nx
import pickle
import numpy as np
from sklearn import preprocessing
class DataProcessor:

    """
    DataProcessor class
    converts the {graph.GrahML, and node/edge attribute pickles} to a pytorch-geometric ready format
    assumes that the input format would not change even after projection
    """

    def __init__(self, node_attribute_list=None, edge_attribute_list=None, debug=False):
        """
        init function

        Args
           node_attribute_list (list(str)): list of node attributes to use as features
           edge_attribute_list (list(str)): list of edge attributes to use as features
        """
        self.debug = debug
        self.node_attribute_list = node_attribute_list
        self.edge_attribute_list = edge_attribute_list

        if self.debug:
            if self.node_attribute_list:
                print('+ node attributes used (# = {}): {}'.format(len(self.node_attribute_list), self.node_attribute_list))
            if self.edge_attribute_list:
                print('+ edge attributes used (# = {}): {}'.format(len(self.edge_attribute_list), self.edge_attribute_list))
        return

    # -------------------
    # main functions
    # -------------------

    def process_single_graph(self, file_path, class_label, 
                             concat_events=False, 
                             compute_order=True):
        """
        processes a single graph into the correct format
        
        Args:
           file_path (str): abs path to the file
           class_label (int): 0 if benign, 1 if malicious
           concat_events (bool): combine events into one edge by taking a frequency count
           compute_order (bool): computes the ordering of events

        Returns:
           A dictionary with the following keys: x, edge_list, y, edge_attr
           where x are node feats and y the class label
        """
        # x, edge_list, y, edge_attr
        #-------------------------------------------------------------------------------------------------------------------
        # 1. get edge_list
        graph = nx.read_graphml(file_path + "/new_graph.graphml")
        edge_dict = self.load_pickle(file_path + "/edge_attribute.pickle")

        # JY @ 2023-05-20: Related to our purpose : CHECK WHAT WE GET FROM events_order.
        # 1.1. get order
        events_order = None
        if compute_order:
            events_order = self.get_events_order(edge_dict)  # 2023-05-20 check here.


        #-------------------------------------------------------------------------------------------------------------------
        # 1.2. getting the edge list
        # there are two types based on if events are concatonated or not
        # current scheme in *_v2= 1-gram model (frequency of events seen for consecutive events across the same two nodes)
        edge_list = None
        edge_names = None
        # if concat_events:
            
        #     # JY @ 2023-05-20:
        #     #   Note that this *_v3 is almost same as dinal's *_v2. 
        #     #   Only change I remember that I made is some network-x or graph library compatibility related issue. 

        #     edge_list, edge_names, freq_vector = self.extract_edge_list_v3(graph, edge_dict)  # new version (edge = concat.events)
        #     # edge_names are now the first event seen

        # else:

        #     # JY @ 2023-05-20: Related to our purpose : CHECK WHAT WE GET FROM edge_list, edge_names.

        #     edge_list, edge_names = self.extract_edge_list(graph)  # old version (edge == events)



        # JY @ 2023-05-20: Even if we don't concatenate events, try using "extract_edge_list_v3",
        #                  since we still want the freq_vector which is no longer freq_vector but just one-hot-encoding

        edge_list, edge_names, task_onehot_vectors_dict = self.extract_edge_list_MultiEdge_Graph(graph, edge_dict)



        if self.debug:
            print("1. loaded edge_list, #edges==#names? {}".format(len(edge_list[0]) == len(edge_names)))

        #-------------------------------------------------------------------------------------------------------------------


        # 2. get node feats (doesnt change)
        x = []  # will store node feats
        node_dict = self.load_pickle(file_path + "/node_attribute.pickle")
        # x = self.extract_node_attr(graph, node_dict)
        x = self.extract_node_attr_5bit(graph, node_dict)
        del node_dict  # to make code memory efficient
        
        if self.debug:
            print("2. loaded node attributes, len(x)==len(nodes)? {}".format(len(x) == len(graph.nodes())))
        #-------------------------------------------------------------------------------------------------------------------
        # 3. get edge feats
        edge_attr = None
        # if concat_events and compute_order:

        #     # JY @ 2023-05-20:
        #     #   Note that this *_v3 is almost same as dinal's *_v2. 
        #     #   Only change I remember that I made is some network-x or graph library compatibility related issue. 

        #     # the only edge attributes = frequency count of tasks + the order-value for the first event seen
        #     edge_attr = self.extract_edge_attr_v3(edge_names, freq_vector, events_order )
        # else:
        #     # JY @ 2023-05-20: 
        #     #   This is not what is needed for MultiEdge graph.
        #     # old version, where all the event attributes are concatonated together
        #     # does not add temporal information
        #     edge_attr = self.extract_edge_attr(edge_names, edge_dict)


        edge_attr = self.extract_edge_attr_MultiEdge_Graph(edge_names, task_onehot_vectors_dict, events_order )





        del edge_dict  # to make code memory efficient
        if self.debug:
            print("3. extracted edge attributes, len(edge_attr)==len(edge_names)? {}".format(len(edge_attr) == len(edge_names)))
            print(f"4. node_dim = {len(x[0])}, edge_dim = {len(edge_attr[0])} (printed for node[0] and edge[0])")
            # prints first node attribute vector and edge attribute vector
            # print(x[0])
            # print(edge_attr[0])
        #-------------------------------------------------------------------------------------------------------------------
        if x == []:
            print("werid")

        return {'x': x, 'edge_list': edge_list, 'y': int(class_label), 'edge_attr': edge_attr}
    
    # -------------------
    # V1/V2
    # node feature extraction does not change
    # -------------------
    
    def extract_node_attr(self, graph, node_dict):
        """
        extracts node attributes
        NOTE: current logic does skip any attributes not in self.node_Attribute_list
        if any other logic is required must be coded here!
        
        Args:
           graph (nx.Graph)
           node_dict (dict): the node attribute dictionary

        Returns:
           x (list(list)): [#nodes, #feat] node attribute matrix
        """
        x = []  # stores node attributes for all nodes [#nodes, #feat]
        for node in graph.nodes(data=True):

            # obtain node name
            node_id, node_data = node
            node_name = node_data['name']
            node_attr = []
            
            # if node name exists, retrieve node data
            if node_name in node_dict:
                # print(node_name, node_dict[node_name])

                attribute_dict = node_dict[node_name]
                for _key in attribute_dict:
                    
                    # skips attributes not in self.node_attribute_list
                    if self.node_attribute_list:
                        if _key not in self.node_attribute_list:
                            continue
                        # else:
                        #     print(_key)
                    
                    # NOTE: its also possible if all attributes are used ->
                    # simplify the code by adding node_attr = list(attribute_dict.values())
                    # instead of looping through it
                    if isinstance(attribute_dict[_key], list):
                        node_attr += attribute_dict[_key]
                        #print(f"{_key}({type(attribute_dict[_key])}) : {len(attribute_dict[_key])}")
                    else:
                        node_attr.append(attribute_dict[_key])
                        #print(f"{_key}({type(attribute_dict[_key])}) : {len(attribute_dict[_key])}")
            else:
                print("> Error in extract_node_attr(): name = {} does not exist in node_attribute.pickle".format(node_name))
                #sys.exit()
            # sys.exit()
            # append node attributes to x
            x.append(node_attr)
            #print("node_name:",node_name)
            #print("length of node attr:", len(node_attr))
            #if len(node_attr) != 270:
                # print("d")
            # print("length of node attr dict keys:", len(attribute_dict.keys()))

        return x



    # JY @ 2023-06-04
    def extract_node_attr_5bit(self, graph, node_dict):
        """
        extracts node attributes
        NOTE: current logic does skip any attributes not in self.node_Attribute_list
        if any other logic is required must be coded here!
        
        Args:
           graph (nx.Graph)
           node_dict (dict): the node attribute dictionary

        Returns:
           x (list(list)): [#nodes, #feat] node attribute matrix
        """
        x = []  # stores node attributes for all nodes [#nodes, #feat]
        for node in graph.nodes(data=True):

            # obtain node name
            node_id, node_data = node
            node_name = node_data['name']
            node_attr = []
            
            # if node name exists, retrieve node data
            if node_name in node_dict:
                # print(node_name, node_dict[node_name])

                if "file" in node_name.lower():
                    node_attr = [1,0,0,0,0]
                elif "reg" in node_name.lower():
                    node_attr = [0,1,0,0,0]
                elif "net" in node_name.lower():
                    node_attr = [0,0,1,0,0]
                elif "proc" in node_name.lower():
                    node_attr = [0,0,0,1,0]
                elif "thread" in node_name.lower():
                    node_attr = [0,0,0,0,1]
                else:
                    print(f"weird 'node_name': {node_name}", flush= True)
                    print(f"set to [0,0,0,0,0]")
                    node_attr = [0,0,0,0,0] # weird-case
            else:
                print("> Error in extract_node_attr(): name = {} does not exist in node_attribute.pickle".format(node_name))
            # append node attributes to x
            x.append(node_attr)
 
        return x



    # ------------------
    # V2 code
    # 1. computing ordering of all events (temporal information)
    # 2. compute the 1-gram frequency count
    # 3. combining frequency count + order as edge features
    # ------------------

    def get_events_order(self, edge_dict):
        """
        Computes the ordering of all events
        1st event order-value = 1/(#events)
        last event order-value = #events / #events = 1.0
        """
        _order = {} # mapping 
        _temp = [] # list of [tuple(name:ts)]
        _sz = len(edge_dict.keys())

        # extract all timestamps
        for _name in edge_dict.keys(): # edge_dict : dict for all edge uids with all attributes -> all event mappings
            _timestamp = edge_dict[_name]["TimeStamp"] # {uid:"taskname","timestamp",}
            _temp.append([_name, _timestamp])

        # sort in ascending order
        _temp.sort(key=lambda x: x[1])

        # add the order
        for i, values in enumerate(_temp):
            # values[0] = name
            _order[values[0]] = (i + 1) / _sz

        # print(_order[list(_order.keys())[0]])
        return _order

    def extract_edge_list_v2(self, graph, edge_dict):
        """
        constructs edge_list in pytorch-geometric format
        concatonates consecutive events into a single edge

        Args:
           graph (nx.Graph)
           edge_dict (dict): attribute dictionary
        
        Returns
           edge_list (list): the edge list from the graph
           edge_names (list): the names for all edges, used to get attributes
           task_freqs (list): frequency count computed
        """
        u, v = [], []  # will hold u,v nodes for all u->v edges
        edge_list, edge_names = [], []  # will hold edge_list = [u, v] and edge_names
        task_freqs = []

        # loop through and extract edges
        # a given edge can hold many edges (I think events) inside it (i.e., > 1 #names)
        for edge in graph.edges(data=True):
            _u, _v, edge_data = edge

            # get u->v node id's, must be int. cannot be n0, n1... for models
            _u = int(_u.strip('n'))
            _v = int(_v.strip('n'))

            # append the edge u->v
            u.append(_u)
            v.append(_v)
            
            # extracting all event names
            _names = edge_data['name']
            _names = ast.literal_eval(_names)  # the names are a str from a list

            # obtain-first event name and first task freq. value
            _temp = [[_names[0], edge_dict[_names[0]]["TimeStamp"]]]  # stores [[name-1,time-1],[name-2,time-2],...,]
            # print(edge_dict[_names[0]]["Task Name"])
            _task_freqs = edge_dict[_names[0]]["Task Name"].copy()

            # add the rest of the timestamps, update the frequency count
            for i in range(len(_names) - 1):
                _temp.append([_names[i + 1], edge_dict[_names[i + 1]]["TimeStamp"]])

                # update frequency count (sum() operation)
                _temp_task = edge_dict[_names[i + 1]]["Task Name"]
                for j in range(len(_task_freqs)):
                    _task_freqs[j] += _temp_task[j]

            # sort and get the first event that occurs w.r.t timstamp
            _temp.sort(key=lambda x: x[1])
            edge_names.append(_temp[0][0])  # appends the event that occured first based on timestamp

            # append task frequencies for current edge
            task_freqs.append(_task_freqs)

        # construct final edge_list
        edge_list.append(u)
        edge_list.append(v)

        return edge_list, edge_names, task_freqs

    def extract_edge_attr_v2(self, edge_names, task_freqs, events_order):
        """
        combines task_freqs with events_order
        """
        edge_attr = []

        for i in range(len(edge_names)):
            # for each edge combines frequency_count with order
            _temp = task_freqs[i] + [events_order[edge_names[i]]]
            edge_attr.append(_temp)
        return edge_attr

    def extract_edge_list_v3(self, graph, edge_dict):
        """
        constructs edge_list in pytorch-geometric format
        concatonates consecutive events into a single edge

        Args:
           graph (nx.Graph)
           edge_dict (dict): attribute dictionary
        
        Returns
           edge_list (list): the edge list from the graph
           edge_names (list): the names for all edges, used to get attributes
           task_freqs (list): frequency count computed
        """
        u, v = [], []  # will hold u,v nodes for all u->v edges
        edge_list, edge_names = [], []  # will hold edge_list = [u, v] and edge_names
        task_freqs = []

        # loop through and extract edges
        # a given edge can hold many edges (I think events) inside it (i.e., > 1 #names)
        for edge in graph.edges(data=True):
            _u, _v, edge_data = edge

            # get u->v node id's, must be int. cannot be n0, n1... for models
            _u = int(_u.strip('n'))
            _v = int(_v.strip('n'))

            # append the edge u->v
            u.append(_u)
            v.append(_v)
            
            # extracting all event names
            _names = edge_data['name']
            _names = ast.literal_eval(_names)  # the names are a str from a list

            # obtain-first event name and first task freq. value
            _temp = [[_names[0], edge_dict[_names[0]]["TimeStamp"]]]  # stores [[name-1,time-1],[name-2,time-2],...,]
            # print(edge_dict[_names[0]]["Task Name"])
            _task_freqs = edge_dict[_names[0]]["Task Name"].copy() # The copy() method returns a new list. It doesn't modify the original list.
            ###----- IRP : with hex value (address) aggregation is not possible-------------####
            #_irp_sum = edge_dict[_names[0]]["Irp"] 
            ### ----------------------------------------###
            


            # add the rest of the timestamps, update the frequency count
            for i in range(len(_names) - 1):
                _temp.append([_names[i + 1], edge_dict[_names[i + 1]]["TimeStamp"]])

                # update frequency count (sum() operation)
                _temp_task = edge_dict[_names[i + 1]]["Task Name"]
                for j in range(len(_task_freqs)):
                    _task_freqs[j] += _temp_task[j]

            # sort and get the first event that occurs w.r.t timstamp
            _temp.sort(key=lambda x: x[1])
            edge_names.append(_temp[0][0])  # appends the event that occured first based on timestamp

            # append task frequencies for current edge
            task_freqs.append(_task_freqs)

            
        # In subgraph-A
        #  Assumming there's 3 edges , E1, E2, E3
        #  X = [ <E1's readopcount>, <E2's readopcount> , <E3's readopcount> ]  <-- raw
        #
        #  [ ( i / L2Norm(X) ) for i in X ]
        #  X_normalized = [ <E1's normalized readopcount>, <E2's normalized readopcount> , <E3's normalized readopcount> ] <<< we have access
        #  
        #  [ i * L2Norm(X) for i in X_normalized ]   f == inverse-(1/L2Norm)
        # 
        #   X = [ 40 ,30 , 20]  
        #   <E1's normalized readopcount> == ( <E1's readopcount> / L2Norm(X) )
        #   
        #  [0.2, 0.3, 0.7]      [40, 60, 140]     [20 30 70] [10 15 35]

        
        
        # not used timestamp as a feature
        freq_vector = {"Task Name":task_freqs}
        # construct final edge_list
        edge_list.append(u)
        edge_list.append(v)

        return edge_list, edge_names, freq_vector

# Note: currently we are normalizing using local vectors (local to edge). However, this local 
# approach looses the global information accross edges.
# e.g., if edge a has max read count = 5000 and edge b has max count = 100
# this local approach will not able to distinguish those edge
# however, as long as count range is somewhat consistent accross edges, this may not be a problem
# Assuming that there is an outlier in an edge, we can consider global normalization
# i.e., normalizing to the global edge event count matrix. but this global approach can also be
# have disavd when the outlier is extream. bcz, the majority typical values will be expressed 
# similarly in terms of value.

    def extract_edge_attr_v3(self, edge_names, freq_vector, events_order):
        """
        combines task_freqs with events_order
        """
        edge_attr = []

        for i in range(len(edge_names)): # looping thru each edge
            # for each edge combines frequency_count with order
            _temp_vector = freq_vector["Task Name"][i]

            # [events_order[edge_names[i]]] : order of all edge based on 1st event on an edge which is sorted by timestamp
            _temp = _temp_vector + [events_order[edge_names[i]]] # edge_names[i] : 1st event uid of edge 'i'
            
            edge_attr.append(_temp)
        return edge_attr


    # ------------------
    # save/load and other helper functions
    # ------------------
    
    def load_pickle(self, filename):
        """
        loads a pickle data
        
        Args
            filename (str): abs.path to load file

        Returns
            data: returns a dict. for node/edge attributes
        """
        data = None
        with open(filename, 'rb') as fp:
            data = pickle.load(fp)
            return data

    def save_pickle(self, filename, data):
        """
        saves processed data into a pickle file

        Args
            filename (str): abs.path to save file
            data (dict): data sample as a dictionary
        """
        with open(filename, 'wb') as fp:
            pickle.dump(data, fp)
        return

    # -------------------------------------------------------------------------------------
    # unused functions from Version-1
    # -------------------------------------------------------------------------------------

    def extract_edge_attr(self, edge_names, edge_dict):
        """
        extracts edge attributes
        NOTE: current logic skips attributes with keys not in self.edge_attribute_list
        
        Args:
           edge_names (list): list of names for all edges with same order as edge_list
           edge_dict (dict): dictionary of edge attributes
        
        Returns
           edge_attr (list): list of edge attributes
        """
        edge_attr = []  # stores the attribute values

        for _name in edge_names:
            # loop through each edge
            if _name in edge_dict:
                _data = edge_dict[_name]
                _values = []  # stores attributes for a given edge
                
                for _key in _data:
                    # add code here to change, skip any attributes

                    if self.edge_attribute_list:
                        if _key not in self.edge_attribute_list:
                            continue

                    if isinstance(_data[_key], list):
                        _values += _data[_key]
                    else:
                        _values.append(_data[_key])

                # append to edge_attr
                edge_attr.append(_values)
            else:
                print('> Error: Edge name = {} does not exist in edge_attribute.pickle'.format(_name))
                #sys.exit()
                # print('> Error: in extract_edge_attr _name = {} does not exist in edge_attribute.pickle'.format(_name))
            #    sys.exit()
                
        return edge_attr
# [[read+ 3]+[]+[]]
    def extract_edge_list(self, graph):
        """
        constructs edge_list in pytorch-geometric format
        Note: I found that loading the Igraph(GraphML) through nx. to be much faster than through Igraph itself?

        Args:
           graph (nx.Graph)
        
        Returns
           edge_list (list): the edge list from the graph
           edge_names (list): the names for all edges, used to get attributes
        """
        u, v = [], []  # will hold u,v nodes for all u->v edges
        edge_list, edge_names = [], []  # will hold edge_list = [u, v] and edge_names

        # loop through and extract edges
        # a given edge can hold many edges (I think events) inside it (i.e., > 1 #names)
        for edge in graph.edges(data=True):
            _u, _v, edge_data = edge

            # get u->v node id's, must be int. cannot be n0, n1... for models
            _u = int(_u.strip('n'))
            _v = int(_v.strip('n'))
            
            # extracting edge names
            _names = edge_data['name']
            _names = ast.literal_eval(_names)  # the names are a str from a list
            for _name in _names:
                u.append(_u)
                v.append(_v)
                edge_names.append(_name)

        # construct final edge_list
        edge_list.append(u)
        edge_list.append(v)
        return edge_list, edge_names



    def extract_edge_list_MultiEdge_Graph(self, graph, edge_dict):

        """
        constructs edge_list in pytorch-geometric format
        concatonates consecutive events into a single edge

        Args:
           graph (nx.Graph)
           edge_dict (dict): attribute dictionary
        
        Returns
           edge_list (list): the edge list from the graph
           edge_names (list): the names for all edges, used to get attributes
           task_onehot_vectors_dict (dict): (1 event , so only 1 bit set)
        """
        u, v = [], []  # will hold u,v nodes for all u->v edges
        edge_list, edge_names = [], []  # will hold edge_list = [u, v] and edge_names
        task_onehot_vectors = []

        # loop through and extract edges
        # a given edge can hold many edges (I think events) inside it (i.e., > 1 #names)
        for edge in graph.edges(data=True):
            _u, _v, edge_data = edge

            # get u->v node id's, must be int. cannot be n0, n1... for models
            _u = int(_u.strip('n'))
            _v = int(_v.strip('n'))

            # append the edge u->v
            u.append(_u)
            v.append(_v)
            
            # extracting event name of this edge (remember that edge == event in multi-edge graph)
            _name = edge_data['name']
            _name = ast.literal_eval(_name)  # the names are a str from a list
            _name = _name[0] # due to ['REG-EDGE_2eaae10d-e2...b8a47faa78'], get string by indexing in.
            edge_names.append(_name)


            task_onehot_vector = edge_dict[_name]["Task Name"].copy() # The copy() method returns a new list. It doesn't modify the original list.
            task_onehot_vectors.append(task_onehot_vector)
        
        
        # not used timestamp as a feature
        task_onehot_vectors_dict = {"Task Name": task_onehot_vectors}
        # construct final edge_list
        edge_list.append(u)
        edge_list.append(v)

        return edge_list, edge_names, task_onehot_vectors_dict        
    



    def extract_edge_attr_MultiEdge_Graph(self, edge_names, task_onehot_vectors_dict, events_order):


        """
        combines task_freqs with events_order
        """
        edge_attr = []

        for i in range(len(edge_names)): # looping thru each edge
            # for each edge combines frequency_count with order
            task_onehot_vector = task_onehot_vectors_dict["Task Name"][i]

            # [events_order[edge_names[i]]] : order of all edge based on 1st event on an edge which is sorted by timestamp
            task_onehot_vector_with_task_event_ordering = task_onehot_vector + [events_order[edge_names[i]]] # edge_names[i] : 1st event uid of edge 'i'
            
            edge_attr.append( task_onehot_vector_with_task_event_ordering )

        return edge_attr    