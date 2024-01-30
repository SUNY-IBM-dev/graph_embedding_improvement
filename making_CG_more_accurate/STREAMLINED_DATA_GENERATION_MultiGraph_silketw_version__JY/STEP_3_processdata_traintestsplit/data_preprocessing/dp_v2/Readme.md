### To Run code:

$ python run_data_processor.py

+ please make sure to change the path as needed
+ please make sure to give the correct attributes as they exist in the dictionaries, the code is case sensitive
+ if more preprocessing is needed for node/edge attributes please change code in data_processor class (i.e., extract_node_attr, extract_edge_attr)

### Changes from Version-1 to Version-2:

1. Adds temporal information:
	- Computes a normalized order for all events based on __TimeDateStamp__
	- step1: Sort order of all events based on __TimeDateStamp__
	- step2: Add the normalized ordering of each event as a new feature (event-i = i / total-#events)

2. Combines consecutive events as single edge
	- Uses the Task Name attribute from the original graphs
	- Computes the frquency for all consecutive events across the same two nodes

3. Now the edge attributes are = event frequency count + order-information for the first event in the edge (if consecutive events are present)

Please note if more attributes have to be combined then the code will need to be changed.
