import os

import sys
sys.path.append("/data/d1/jgwak1/tabby/graph_embedding_improvement_JY_git/making_CG_more_accurate/STREAMLINED_DATA_GENERATION_MultiGraph_silketw_version__JY/STEP_3_processdata_traintestsplit/data_preprocessing/dp_v2_ONLY_TASKNAME_EDGE_ATTR")
# from data_processor_v2_MultiEdge import DataProcessor
from data_processor_v2_MultiEdge_5BitNodeAttr import DataProcessor

def process_all_graphs(data_type, load_path, save_path, data_proc, debug=False, 
                       concat_events=False, 
                       compute_order=True):
    """
    basically loops thrugh all samples and preprocesses them,
    this function is used for both Malware and Benign samples
    NOTE:
    e.g., expects graphs to be like this
    $load_path/
       $data_type/
          $data_type + '_Sample_1'/
             graph.GraphML
             node_attribute.pickle
             edge_attribute.pickle
          $data_type + '_Sample_2'/
             graph.GraphML
             node_attribute.pickle
             edge_attribute.pickle
    
    Args
       load_path (str): the absolute path to the data folder that holds data
       data_type (str): can only be either 'Benign' or 'Malware', the $path above should have these two folders--1) 'Benign' will hold all benign samples with the strict format of 'Benign_Sample_1/', 'Benign_Sample_2' ect. and 2) 'Malware' folder should hold 'Malware_Sample_1/', 'Malware_Sample_2/'
       save_path (str): abs.path to save all the data
       data_proc (DataProcessor): an instance of the data processor class
       concat_events (bool): if True, takes a frequency count for all events in an edge
       compute_order (bool): if True, computes the normalized order of events (temporal information)
    """
    # 1. set class labels, please note that the entire graph is labeled 0-Benign and 1-Malware
    # You will need to change labeling as required after projection
    class_label = None
    if data_type == "Malware":
        class_label = 1
    elif data_type == "Benign":
        class_label = 0
    else:
        print("> Error for data_type={}".format(data_type))

    # 2. make folders to save samples if needed
    _save_path = save_path + "/Processed_" + data_type + "_ONLY_TaskName_edgeattr"
    if os.path.exists(_save_path) is False:
        os.makedirs(_save_path)
    
    # 3. get path for all samples
    _path = load_path + "/" + data_type
    directory_contents = os.listdir(_path)

    # 4. loop through and process them
    for folder in directory_contents:

        # skip folders that don't have data
        if data_type + "_Sample_" not in folder and "SUBGRAPH" not in folder:
            continue

        if debug:
            print("\n processing {}".format(folder))
        
        _sample_path = _path + "/" + folder
        _sample = data_proc.process_single_graph(_sample_path, class_label, 
                                                 concat_events=concat_events, 
                                                 compute_order=compute_order)
        # print(_sample.keys())
        # sample will be dictionary with keys {x: the node feats, y: class_label, edge_list and edge_attr}
        # this can be directly loaded and converted to tensors for the models
        
        # save the graph to file
        _sample_save_path = _save_path + "/Processed_" + folder + ".pickle"
        data_proc.save_pickle(_sample_save_path, _sample)

        if debug:
            print('> len(node-attr): {} | len(edge-attr): {}'.format(len(_sample['x'][0]), len(_sample['edge_attr'][0])))
            print('> file saved @ {}'.format(_sample_save_path))

        #return
    return



def run_data_processor(data_type : str, load_and_save_path: str):
    # [[3,0,0]] --- > 3 read consucutive events 

    # 0. initialize the dataprocessor class
    debug = True

    # for new graphs

    #node_attribute_list = [ 'FileName', 'FileObject', 'KeyObject', 'RelativeName', 'Win32StartAddr', 'daddr']
    node_attribute_list = [] # testing- remove FileName attribute

    # edge_attribute_list = ['FileKey','FilePath', 'ImageBase', 'ImageName', 'Irp', 'Opcode', 'Task Name', 'sport', 'dport', 'saddr']

    #edge_attribute_list = ['Task Name','size', 'ImageSize', 'HandleCount',
    #                    'ReadOperationCount','WriteOperationCount',
    #                    'ReadTransferKiloBytes','WriteTransferKiloBytes','StatusCode',
    #                    'SubProcessTag','ProcessTokenElevationType',
    #                    'ProcessTokenIsElevated','MandatoryLabel','PackageRelativeAppId',
    #                    'Status','Disposition']  # total 16

    edge_attribute_list = ['Task Name']

    # JY @ 2023-05-20 : NO NEED TO CONCAT_EVENTS ANYMORE.
    concat_events = False  # will also concatonate events and take a frequency count 
    
    # JY @ 2023-05-20 : YES WE MAY STILL WANT THIS.
    compute_order = True  # will compute order for all events [1/#events, 2/#events, ...., #events/#events==1.0] 

    # JY @ 2023-05-20 : NO NEED TO CONCAT_EVENTS ANYMORE.
    # (X) A single edge = frequency count for all events + order-value of the first event seen
    # -> For Multi-Edge context, A single edge = bit-vector set for this event + order-value of this event across all events

    data_proc = DataProcessor(node_attribute_list=node_attribute_list, 
                              edge_attribute_list=edge_attribute_list, 
                              debug=debug)

    #load_path = '/data/d1/jgwak1/tabby/NEW_Projection3_Datasets_Based_On_Modified_CG_Code__20230124/GENERAL_LOG_COLLECTION_SUBGRAPHS_20230131'
    #save_path = '/data/d1/jgwak1/tabby/NEW_Projection3_Datasets_Based_On_Modified_CG_Code__20230124/GENERAL_LOG_COLLECTION_SUBGRAPHS_20230131'
    
    load_path = save_path = load_and_save_path

    # 1. process and save all benign samples
    if data_type == "Benign":
        print("\n>> Benign <<")
        process_all_graphs(data_type, load_path, save_path, data_proc, debug=debug, 
                           concat_events=concat_events, 
                           compute_order=compute_order)

    # 2. process and save all malware samples
    if data_type == "Malware":
        #print("\n>> Malware <<")
        process_all_graphs(data_type, load_path, save_path, data_proc, debug=debug, 
                           concat_events=concat_events, 
                           compute_order=compute_order)

    """
    def test():
        dp = DataProcessor(True)

        # filename = "benign_graph.GraphML"

        filename = "./Malware_sample_3"
        class_label = 1
        sample = dp.process_single_graph(filename, class_label)

        print(sample.keys())

    test()
    """
