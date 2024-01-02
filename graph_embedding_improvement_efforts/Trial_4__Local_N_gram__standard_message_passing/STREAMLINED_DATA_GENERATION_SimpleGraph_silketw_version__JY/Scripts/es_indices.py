from elasticsearch import Elasticsearch
import os
es = Elasticsearch(['http://ocelot.cs.binghamton.edu:9200'],timeout = 300)
idx_list = list( es.indices.get_alias("*") )
curr_dirpath = "/home/pwakodi1/tabby/STREAMLINED_DATA_GENERATION_MultiGraph_silketw_version/"

with open(os.path.join(curr_dirpath ,"silketw_new_indices_list.txt"),"w" ) as fp:
    for i in idx_list:
        fp.write(i+"\n")
    #json.dump(aggregated_filename_dict, fp)
