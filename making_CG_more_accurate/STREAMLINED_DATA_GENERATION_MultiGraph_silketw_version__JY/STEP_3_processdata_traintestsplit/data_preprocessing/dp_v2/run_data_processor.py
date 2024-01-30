import os
from data_processor_v2 import DataProcessor


def process_all_graphs(data_type, load_path, save_path, data_proc, debug=False, concat_events=True, compute_order=True):
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
    _save_path = save_path + "/Processed_" + data_type
    if os.path.exists(_save_path) is False:
        os.makedirs(_save_path)
    
    # 3. get path for all samples
    _path = load_path + "/" + data_type
    directory_contents = os.listdir(_path)

    # 4. loop through and process them
    for folder in directory_contents:

        # skip folders that don't have data
        if data_type + "_Sample_" not in folder:
            continue

        if debug:
            print("\n processing {}".format(folder))
        
        _sample_path = _path + "/" + folder
        _sample = data_proc.process_single_graph(_sample_path, class_label, concat_events=concat_events, compute_order=compute_order)
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

# [[3,0,0]] --- > 3 read consucutive events 

# 0. initialize the dataprocessor class
debug = True

# for new graphs

#node_attribute_list = [ 'FileName', 'FileObject', 'KeyObject', 'RelativeName', 'Win32StartAddr', 'daddr']

node_attribute_list = ['FileName',  'daddr', 'RelativeName'] 

edge_attribute_list = ['Task Name','size', 'ImageSize', 'HandleCount',
                       'ReadOperationCount','WriteOperationCount',
                       'ReadTransferKiloBytes','WriteTransferKiloBytes',
                       'StatusCode', 'Status', 'SubProcessTag',
                       'ProcessTokenElevationType', 'ProcessTokenIsElevated',
                       'MandatoryLabel','PackageRelativeAppId', 'Disposition'] 

edge_attribute_list = ['Task Name']

concat_events = True  # will also concatonate events and take a frequency count 
compute_order = True  # will compute order for all events [1/#events, 2/#events, ...., #events/#events==1.0] 

# A single edge = frequency count for all events + order-value of the first event seen

data_proc = DataProcessor(node_attribute_list=node_attribute_list, edge_attribute_list=edge_attribute_list, debug=debug)

#load_path = os.path.expanduser("/home/pwakodi1/tabby/new_benign_log_prj3_subgraphs/")  # will load from Benign/ and Malware/ folder
#save_path = os.path.expanduser("/home/pwakodi1/tabby/new_benign_log_prj3_subgraphs/")  # will save in Processed_Benign/ and Processed_Malware/ folders


load_path = '/data/d1/jgwak1/tabby/NEW_Projection3_Datasets_Based_On_Modified_CG_Code__20230124'
save_path = '/data/d1/jgwak1/tabby/NEW_Projection3_Datasets_Based_On_Modified_CG_Code__20230124'

#load_path = "/data/d1/jgwak1/tabby/NEW_Projection3_Datasets_Based_On_Modified_CG_Code__20230124/NEW_BENIGN_USER_ACTIVITY_SUBGRPAHS_SET_4"
#save_path = "/data/d1/jgwak1/tabby/NEW_Projection3_Datasets_Based_On_Modified_CG_Code__20230124/NEW_BENIGN_USER_ACTIVITY_SUBGRPAHS_SET_4"

# 1. process and save all benign samples
print("\n>> Benign <<")
data_type = "Benign"
process_all_graphs(data_type, load_path, save_path, data_proc, debug=debug, concat_events=concat_events, compute_order=compute_order)

# 2. process and save all malware samples
#print("\n>> Malware <<")
#data_type = "Malware"
#process_all_graphs(data_type, load_path, save_path, data_proc, debug=debug, concat_events=concat_events, compute_order=compute_order)

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
