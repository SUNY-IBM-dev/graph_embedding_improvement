# Author : Priti 


+ The model_v2 directory is used to collect Offline malware data 

1) Directory: model_v2
    + Important files are
        1. FirstStep.py
            + This file will generate the raw computation graph by extracting all nodes and egdes from the log entries stored in elastic search
        2. EdgeDirection_Sorted.py
            + This file will change the edge directions of the computation graphs based on the events
            + This will also sort the events on an edge based on the TimeStamp
        3. SecondStep.py
            + This file has all the necessary functions to convert all attribute values into one hot format
        4. Projection1_final.py (Currently not in use)
            + This is the 1st projection algorithm which records all activities after he processstart event
            + This will also record all the connections made by the tainted file, registry, network and process
        5. Projection2_final.py (Currently not in use)
            + The second projection only record the outgoing process events till the File, registry and network
            + It will also record all decendent processes from the current root process node
        6. Projection2_final.py (Currently in use)
            + The third projection records all incoming and outgoing process events till the File, registry and network
            + It will also record all decendent processes from the current root process node

        
2) main.py (/data/d1/pwakodi1/main.py)
    + Usage : python main.py
    + To run all the programs mentioned above in Directory : model_v2 , we used main.py to get the subgraphs
    + Note: There are many scripts created at - /data/d1/pwakodi1/Scripts_for_malware, which will help to run main.py with different malware and its PID.
    + You can create new scripts for new malware samples in similar way. Please dont change script stored at /data/d1/pwakodi1/Scripts_for_malware
    




