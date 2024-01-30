### Data Preprocessing (dp_*) code

Code here converts the Igraphs (projected/complete) into pytorch-geometric acceptable formats.
In both **dp_v1** and **dp_v2**, the node features from original igraphs are concatonated into a single vector  based on user input. The main difference between the two versions are the way edges are generated.

1. dp_v1/ (data_preprocessing_version1):
   - edges: all events are considered as seperate edges
   - edge features: same as node features

2. dp_v2/ (data_preprocessing_version2):
   - edges: a single edge can represnt many consecutive events across the same two nodes (not seperated out like v1)
   - edge features:
     1. a frequency count for the number of consecutive events
     2. the normalized order of the events (i.e., the first event in a consecutive bunch of events), the order is comuted by first sorting the events based on TimeDateStamp, then by giving a numeric index (1,..,#total-events) to the sorted result and then normalizing it w.r.t total number of events (#total-events)
