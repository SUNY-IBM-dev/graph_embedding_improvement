''' 
Event-logs that are directly streamed from pywintrace to elastic-search are unformatted.
(As such event-logs did not go through our formatting steps in "benign/user-4/Code/data_collection.py" and "benign/user-4/Code/Auto_demo_JY.py")

Thus, for such event-logs, we format it here by:
1. Reading those unformatted elastic-search indices here.
2. Format those.
3. Writing it out as formatted elastic-search indices.
'''

import os
from elasticsearch import Elasticsearch, helpers
import json
import time
import sys
#sys.path.append("/data/d2/etw-logs_JY/benign/user-4/Code/")
sys.path.append("/data/d1/jgwak1/tabby/graph_embedding_improvement_JY_git/graph_embedding_improvement_efforts/Trial_4__Local_N_gram__standard_message_passing/STREAMLINED_DATA_GENERATION_SimpleGraph_silketw_version__JY/STEP_1_FORMATTING_ES_INDICES_FILES")
from model import Selection as s
from model import Multi2Single as ms
from model import txt2json as tj


def tailored_mutilevel2singlelevel(log_entry):

#def mutilevel2singlelevel(json1, json2):
#    fp = open(json1, encoding = 'utf-8', errors = 'ignore')
#    f2 = open(json2, 'w', encoding = 'utf-8', errors = 'ignore')
    #f2.write('[')
#    data = tj.get_data(fp)
    #d = eval(data)
    singlelevel_log_entry = {}
    count = 0
        #data = tj.get_data(fp)
        #if data:
    for item in log_entry['_source'].items(): 
        if count == 0:
            count = 1
            item = item[1]
            for item2 in item.items():
                if item2[0] == 'EventDescriptor':
                    item2 = item2[1]
                    for item3 in item2.items():
                        singlelevel_log_entry[item3[0]] = item3[1]
                else:
                    singlelevel_log_entry[item2[0]] = item2[1]
        else:
            singlelevel_log_entry[item[0]] = item[1]

    return singlelevel_log_entry


 
 
 
#from dateutil.parser import parse
def tailored_selection(singlelevel_log_entry):

    selection_processed_log_entry = {}
    for item in singlelevel_log_entry.items():
        if item[0] not in tj.no_attributes:
            if item[0] == 'ThreadId' :
                if isinstance(item[1], str):
                    selection_processed_log_entry[item[0]]= int(item[1], 16)
                else:
                    selection_processed_log_entry[item[0]] = item[1]
            else:
                selection_processed_log_entry[item[0]] = item[1]
    return selection_processed_log_entry


 
 

#if __name__ == "__main__":

def run_format_logentry_on_elasticsearch( unformatted_elasticsearch_indices : list ):

    # Elasticsearch-Indices that are unformatted
    #unformatted_elasticsearch_indices = ["general_log_collection_60mins_started_20230203_23_13_44",
    #                                     "general_log_collection_60mins_started_20230203_20_58_30",
    #                                     "general_log_collection_60mins_started_20230203_22_09_36",
    #                                     "general_log_collection_60mins_started_20230203_19_27_38",
    #                                     "general_log_collection_60mins_started_20230203_18_07_31"
    #                                     ]

    es = Elasticsearch("http://panther.cs.binghamton.edu:9200", timeout=30)
    
    for unformatted_elasticsearch_index in unformatted_elasticsearch_indices:
        
        formatted_elasticsearch_index = f"{unformatted_elasticsearch_index}_formatted"
        
        
        es.indices.create(index=formatted_elasticsearch_index, 
                          body={ 'settings' : { 'index' : { 'number_of_replicas':0 } }})
        
        es.indices.put_settings(index = unformatted_elasticsearch_index, body={'index':{'max_result_window':99999999}})

        #query_body = {"query": {"match_all": {}},"sort" : [{"TimeStamp" : {"order" : "asc"}}]}
        #result = es.search(index = unformatted_elasticsearch_index, doc_type = 'some_type', body = query_body, size = 99999999)
        result = es.search(index = unformatted_elasticsearch_index, doc_type = 'some_type', size = 99999999)
        unformatted_all_log_entries = result['hits']['hits']    # list of dicts
        
        
        formatted_all_log_entries = []
        
        cnt = 0
        print( f"start formatting",flush=True)
        for log_entry in unformatted_all_log_entries:
            
            len(formatted_all_log_entries)
            
            singlelevel_log_entry = tailored_mutilevel2singlelevel(log_entry)
            selection_processed_log_entry = tailored_selection(singlelevel_log_entry)
            
            log_entry_to_upload = {
                "_index": formatted_elasticsearch_index,
                "_type": "some_type",
                "_source": selection_processed_log_entry
            }
            
            formatted_all_log_entries.append(log_entry_to_upload)
            
            
            
            cnt+=1
            #print( f"{cnt/len(unformatted_all_log_entries)}% formatted",flush=True)
        print( f"done formatting",flush=True)

        # just double check.
        assert len(formatted_all_log_entries) == len(unformatted_all_log_entries), "formatted and unformatted log_entries should match in number"
             
        try:
            print("uploading the formatted elastic-search index (with 'formatted' suffix added)", flush= True)
            resp = helpers.bulk(
                client = es,
                actions = formatted_all_log_entries, # yield To_stream
                refresh=True
            )
            print("\nhelpers.bulk() RESPONSE:", resp, flush=True)
            print("RESPONSE TYPE:", type(resp), flush=True)
            print(f"'{formatted_elasticsearch_index}' is uploaded",flush=True)
        except Exception as err:
            print("\nhelpers.bulk() ERROR:", err, flush=True)
            

        try:
            #print("deleting the unformatted elastic-search index", flush= True)
            #es.indices.delete(unformatted_elasticsearch_index, timeout='300s')
            print("for now, not deleting the unformatted elastic-search index", flush= True)
        except Exception as e:
            raise RuntimeError(e)
