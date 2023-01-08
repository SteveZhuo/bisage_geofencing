Note: all configurations (required for running) are in **config.py**.
Current **config.py** is the lab dataset.
To run the shopping mall experiment, change the 'data_name' in **config.py** to 'mall'. 'id' in **config.py** shall also be changed correspondingly.
To run the UJI experiments, change the 'data_name' in **config.py** to 'uji{floor_id}'. 'id' in **config.py** shall also be changed correspondingly.

1. Create a new folder **input** (replacing the old one). Put the input under folder **input**, please check the given file for suggested format and naming. In each row, 0 stands for 'in' and 1 stands for 'out'

2. Run **prep_graph.py**, the output would be folders in **emb_output**. This file processes the signal records and generates corresponding bipartite graph. 
In training, the signal records are used to construct an initial bipartite graph. In testing, new MAC nodes and signal-record nodes will be added into the graph.

3. Run **bisage.py**(requires modifying parameters in config.py, in our example, "lab_train" and "lab_test"), the output would be folders in **emb_output/{your data name}/embedding**. This file processes the 
bipartite graph and generates the embedding for each node using our proposed representation learning algorithm BiSAGE. In training, it generates node embeddings and learns weight matrices through minimizing the loss function. 
In testing, the embedding of each node is quickly inferenced through aggregation.

4. Run **hbos.py**, the result would be in folder **result**, logs files would be in folder **update_output**. This file builds the initial histogram for outlier detection and generates the raw outlier score for 
each embedding using the HBOS outlier detection algorithm. 

5. Run **sigmoid_scaling.py**, the result would be in folder **result**. This file applies sigmoid scaling to the outlier scores and detects outliers if any. It also updates the histograms built in the HBOS model if 
there is highly confident in-boundary signal record. 

For mall, we split the dataset into parts, thus some additional numbering of filename would be required here.
