from elasticsearch import Elasticsearch
import json

def flatten_dict(d, parent_key='', sep='_'):
    items = {}
    for key, value in d.items():
        new_key = f"{parent_key}{sep}{key}" if parent_key else key
        if isinstance(value, dict):
            items.update(flatten_dict(value, new_key, sep=sep))
        else:
            items[new_key] = value
    return items

def from_elastic(idx):

    es = Elasticsearch(['http://ocelot.cs.binghamton.edu:9200'],timeout = 300)
    es.indices.put_settings(index=idx,
                            body={'index':{
                            'max_result_window':999999999}})
    
    necessary_attributes = [
                            # File-Specific
                            "FileName", 
                            # Network-Specific
                            "daddr",
                            # Registry-Specific
                            'RelativeName',
                            ]
    
    #TODO: change of timestamp to get correct sorted timestamp (done)
    
    query_body = {
        "query": {  "match_all": {} },
        #"_source": necessary_attributes, 
        "sort" : [ { "@timestamp" : { "order" : "asc" } } ]
    }

    # USE SCROLL?
    scroll_size = 10000
    scroll_time = '1m'

    result = es.search(index = idx, body = query_body, 
                      scroll='1m',
                      size = 99999999)
    all_hits = result['hits']['hits']
    all_hits_flatten = []
    for hit in all_hits:
            event_original_data = json.loads(hit['_source']['event']['original'])

            # Flatten the dictionary
            flattened_data = flatten_dict(hit)
            flattened_data['_source_event_original'] = event_original_data  # Add the parsed 'original' data
            flattened_event_original = flatten_dict(event_original_data, 'event_original')
            
                    # Update the flattened_data with the flattened_event_original
            flattened_data.update(flattened_event_original)
            flattened_data.pop('_source_event_original',None)
            all_hits_flatten.append(flattened_data)
    return all_hits_flatten

def main():
    #idx_list = ['abby_finereader','acrobat_pro_dc_x64']
    es = Elasticsearch(['http://ocelot.cs.binghamton.edu:9200'],timeout = 300)
    idx_list = list( es.indices.get_alias("*") )

    FILE_provider = 'edd08927-9cc4-4e65-b970-c2560fb5c289'
    NETWORK_provider = '7dd42a49-5329-4832-8dfd-43d979153a88'
    #PROCESS_provider = '22fb2cd6-0e7b-422b-a0c7-2fad1fd0e716'
    REGISTRY_provider = '70eb4f03-c1de-4f73-a051-33d13d5413bd'
    filename_dict = dict() # {index1:{fileclearname:value},index2----}
    relativename_dict = dict()
    daddr_dict = dict()
    
    idx_cnt = 0
    for index in idx_list:

        try:

            filename_dict[index] = dict()
            relativename_dict[index] = dict()
            daddr_dict[index] = dict()

            all_hits = from_elastic(index)
            for i, log_entry in enumerate(all_hits):
                provider = log_entry.get('_source_ProviderGuid')
                if provider == FILE_provider:
                    logentry_FileName = str(log_entry.get('_source_XmlEventData_FileName')).lower()
                    if logentry_FileName in filename_dict[index]:
                        filename_dict[index][logentry_FileName]+=1
                    else:
                        filename_dict[index][logentry_FileName] = 1

                if provider == REGISTRY_provider:
                    logentry_RelativeName = str(log_entry.get('_source_XmlEventData_RelativeName')).lower()
                    if logentry_RelativeName in relativename_dict[index]:
                        relativename_dict[index][logentry_RelativeName]+=1
                    else:
                        relativename_dict[index][logentry_RelativeName] = 1

                if provider == NETWORK_provider:
                    logentry_destaddr = str(log_entry.get('_source_XmlEventData_daddr')).lower()
                    if logentry_destaddr in daddr_dict[index]:
                        daddr_dict[index][logentry_destaddr] += 1
                    else:
                        daddr_dict[index][logentry_destaddr] = 1
        except Exception as e:
            print(f"Exception occured for {index}\n: {e}\n", flush = True)


        idx_cnt += 1
        print(f"{idx_cnt} / {len(idx_list)} indices are done", flush = True)



    print('check')

    aggregated_filename_dict = { filename_value : sum( inner_dict.get(filename_value, 0) \
                                 for inner_dict in filename_dict.values())\
                                 for filename_value in set().union( *filename_dict.values() ) }
    aggregated_filename_dict = dict(sorted(aggregated_filename_dict.items(), reverse= True, key = lambda item: item[1]))

    aggregated_relativename_dict = { relativename_value : sum( inner_dict.get(relativename_value, 0) \
                                   for inner_dict in relativename_dict.values())\
                                   for relativename_value in set().union( *relativename_dict.values() ) }
    aggregated_relativename_dict = dict(sorted(aggregated_relativename_dict.items(), reverse=True, key = lambda item: item[1]))

    aggregated_daddr_dict = { daddr_value : sum( inner_dict.get(daddr_value, 0) \
                              for inner_dict in daddr_dict.values()) \
                              for daddr_value in set().union( *daddr_dict.values() ) }
    aggregated_daddr_dict = dict( sorted(aggregated_daddr_dict.items(), reverse=True, key = lambda item: item[1])) 
     
    curr_dirpath = "/home/pwakodi1/tabby/STREAMLINED_DATA_GENERATION_MultiGraph_silketw_version"

    import os
    with open(os.path.join(curr_dirpath ,"aggregated_filename_dict.json"),"w" ) as fp:
        json.dump(aggregated_filename_dict, fp)
    with open(os.path.join(curr_dirpath ,"aggregated_relativename_dict.json"),"w" ) as fp:
        json.dump(aggregated_relativename_dict, fp)
    with open(os.path.join(curr_dirpath ,"aggregated_daddr_dict.json"),"w" )as fp:
        json.dump(aggregated_daddr_dict, fp)

    # import os
    # with open(os.path.join(curr_dirpath ,"aggregated_filename_dict__toyexample.json"),"w" ) as fp:
    #     json.dump(aggregated_filename_dict, fp)
    # with open(os.path.join(curr_dirpath ,"aggregated_relativename_dict__toyexample.json"),"w" ) as fp:
    #     json.dump(aggregated_relativename_dict, fp)
    # with open(os.path.join(curr_dirpath ,"aggregated_daddr_dict__toyexample.json"),"w" )as fp:
    #     json.dump(aggregated_daddr_dict, fp)


    # 
    # 1. Cluster the unique log-attribute values by some similarity (e.g. 00001 and 00002 belong to same cluster) -- human-input
    # 2. Based on cluseter, make into bit the more frequent ones 
    #    (reason for not including for all cluseters, is to avoid having an too-big feature-space ; the threshold again human-input)
    # 
    # 'daddr' 

if __name__== '__main__':
    main()