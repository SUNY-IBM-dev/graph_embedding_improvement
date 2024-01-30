import os

import shutil

import datetime

from elasticsearch import Elasticsearch

import numpy as np

import re

import pandas as pd

from dateutil.parser import parse
import matplotlib.pyplot as plt
import datetime

def time_to_seconds(time_str):
    parts = time_str.split(':')
    hours, minutes, seconds = map(int, parts)
    return hours * 3600 + minutes * 60 + seconds

def parse_es_timestamp( logentry_timestamp ):
    return datetime.datetime(1601,1,1)+datetime.timedelta(microseconds=(logentry_timestamp/10)) - datetime.timedelta(hours=5)

datetime_objects=[]

if __name__ == "__main__":



    # TO MANIPULATE

    outputs_parentdir = "/home/pwakodi1/tabby/STREAMLINED_DATA_GENERATION_MultiGraph/STEP_2_Benign_NON_TARGETTED_SUBGRAPH_GENERATION_GeneralLogCollection_subgraphs/model_v3_PW/outputs"



    outputs_dirpath = os.path.join(outputs_parentdir, "tell-joke" )

    diff_min_threshold = datetime.timedelta(seconds=30) # we want to identfiy timestamp-differences that are above this threshold.



    indices_to_investigate = [

'tell-joke'



]



    overall_jumptest_results_df = pd.DataFrame(columns = ["Index", "Index_Recorded_Time_Range(HH:MM:SS)", "# Jump Incidents", "All Jumps(HH:MM:SS)" ])

    





    if os.path.exists(outputs_dirpath):

        shutil.rmtree(outputs_dirpath)

    os.makedirs(outputs_dirpath)    



    # Record to the overall record list file

    # with open(os.path.join( outputs_dirpath, f"Overall_JumpIncidents_Record.txt"), 'w') as ff: 

    #     ff.write(f"Index | #Jump-Incident \n")



    for index in indices_to_investigate:

        print("index",index)



        try:        

            es = Elasticsearch(['http://ocelot.cs.binghamton.edu:9200'],timeout = 300)

            es.indices.put_settings(index=index, body={'index':{'max_result_window':99999999}})

            query_body = {"query": {"match_all": {}}, "sort" : [{"@timestamp" : {"order" : "asc"}}]}

            result = es.search(index = index, body = query_body, size = 99999999)

            #all_log_entries = result['hits']['hits'] 



            f = open(os.path.join( outputs_dirpath, f"{index}.txt"), 'w')



           # benign_index_humanreadable_timestamps = [hit['_source']['TimeStamp'] for hit in result['hits']['hits']]

            timestamp_strings = [(hit['_source']['@timestamp']) for hit in result['hits']['hits']]
            for timestamp in timestamp_strings:
                dt = parse(timestamp)
                datetime_objects.append(dt)
            # x = [datetime.datetime.fromisoformat(i) for i in timestamp_strings]
            # datetime_objects = [parser.isoparse(timestamp) for timestamp in timestamp_strings]
            #print(datetime_objects)
            # Filter out any None values (invalid timestamps)
           # timestamp_strings = [timestamp for timestamp in timestamp_strings if timestamp]

            # Parse the timestamp strings into datetime objects
            #timestamps = [parse_es_timestamp(timestamp_str) for timestamp_str in timestamp_strings if timestamp_str]
            #timestamps = [parse_es_timestamp(timestamp_str) for timestamp_str in timestamp_strings]
           # timestamps = [timestamp for timestamp in v_timestamps if timestamp is not None]

# Check if there are valid timestamps to calculate the intervals
            #if timestamps:

            

            first_index_humanreadable_timestamp = datetime_objects[0]

            last_index_humanreadable_timestamp = datetime_objects[-1]

            index_logentry_timestamp_range = last_index_humanreadable_timestamp - first_index_humanreadable_timestamp



            benign_index_humanreadable_timestamp_diffs = np.diff(datetime_objects).tolist()
            # print(str(benign_index_humanreadable_timestamp_diffs[110].total_seconds())) 
            #x = [time_to_seconds(duration) for duration in benign_index_humanreadable_timestamp_diffs]
            #y_values = range(1, len(benign_index_humanreadable_timestamp_diffs) + 1)
            max_benign_index_humanreadable_timestamp_diffs =max(benign_index_humanreadable_timestamp_diffs)



            above_threshold_humanreadable_timestamp_diffs = sorted( [ diff for diff in benign_index_humanreadable_timestamp_diffs if diff >= diff_min_threshold ], reverse= True )



            benign_index_humanreadable_timestamp_diffs = [0] + benign_index_humanreadable_timestamp_diffs # add [0] in the beginning as there's no diff in very beginning
            #print(benign_index_humanreadable_timestamp_diffs)
            

            #x = [str(dt) for dt in benign_index_humanreadable_timestamp_diffs]
            #x = x[0:100]
            
           # y_values = y_values[0:100]
            
            
            
            # plt.scatter(benign_index_humanreadable_timestamp_diffs, y_values, marker='o', color='blue')
            # plt.xlabel('Time Interval (seconds)')
            # plt.ylabel('Interval Number')
            # plt.title('Time Intervals Between Consecutive Timestamps')
            # plt.grid(True)

            # # Calculate the maximum time interval
            # max_time_interval = max(benign_index_humanreadable_timestamp_diffs)

            # # Print the maximum time interval to the terminal
            # print(f"Maximum Time Interval Between Consecutive Timestamps: {max_time_interval} seconds")

            # # Write the maximum time interval to a file
            # with open('max_time_interval.txt', 'w') as max_time_file:
            #     max_time_file.write(f"Maximum Time Interval: {max_time_interval} seconds\n")

            # # Show the plot
            # plt.show()



            # Record to the overall record list file

            # with open(os.path.join( output_txtfiles_dirpath, f"Overall_JumpIncidents_Record.txt"), 'a') as ff: 

            #     ff.write(f"{index}: {len(above_threshold_humanreadable_timestamp_diffs)}\n")

            # https://stackoverflow.com/questions/18470627/how-do-i-remove-the-microseconds-from-a-timedelta-object

            new_row_series = pd.Series({"Index": index,   

                                       "Index_Recorded_Time_Range(HH:MM:SS)" : str(index_logentry_timestamp_range).split(".")[0],

                                       "# Jump Incidents": len(above_threshold_humanreadable_timestamp_diffs),

                                       "All Jumps(HH:MM:SS)": str([str(dif).split(".")[0] for dif in above_threshold_humanreadable_timestamp_diffs])

                                       })



            overall_jumptest_results_df = pd.concat([overall_jumptest_results_df, new_row_series.to_frame().T], ignore_index = True)



            # record to the record-file for individual samples

            f.write(f"{index} -- MAX TimeDifference-with-Prev-TimeStamp: {str(max_benign_index_humanreadable_timestamp_diffs)}\n\n")



            f.write(f"{index} log-entry timestamp range {str(index_logentry_timestamp_range)}\n")





            f.write(f"Incidents of TimeDifference >= {str(diff_min_threshold)}\n")

            incident_occurence = 1

            for above_threshold_humanreadable_timestamp_diff in above_threshold_humanreadable_timestamp_diffs:

                f.write(f"  {incident_occurence}:  {above_threshold_humanreadable_timestamp_diff}\n")

                incident_occurence+=1

            f.write("\n\n")

            f.write("HumanReadable-LogEntry-TimeStamp  ||  TimeDifference-with-Prev-TimeStamp\n")



            for i in range(len(datetime_objects)):

                f.write(f"{datetime_objects[i]}        ||  {benign_index_humanreadable_timestamp_diffs[i]}\n")        

        

            f.close()



        except Exception as e:

            print(f"Problem with {index} : {e}")



    overall_jumptest_results_df.to_csv( os.path.join(outputs_dirpath, f"Overall_JumpTest_Results_Data.csv"))    

    print("DONE", flush= True)

    





