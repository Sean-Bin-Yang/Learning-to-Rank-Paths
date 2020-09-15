# Learning-to-Rank-Paths

This repository holds the code used in our **TKDE-20** paper: [Context-Aware Path Ranking in Road Networks]().

## Requirements

* Ubuntu OS (16.04)
* Python = 3.6
* Numpy >= 1.16.2
* Pickle
* Tensorflow = 1.12.0

Please refer to the source code to install the required packages in Python.

## Dataset

In the Data folder, there are four files:
* data_DT200915_example_train.pkl is a sample data file. The data format is (x_data, x_temporal,x_driver,y_train,tt_train,fc_train,len_train), here x_data is path. x_temporal is temporal information for specificed paths based on the departure time. x_driver is the additional information of driver. y_train is a path similarity with the ground truth path. tt_train (travel time), fc_train (fuel consumpation) and len_train (travel distance) is the additional information of path.
* driverid_onehot_0823_166.pkl is the onehot embedding for driver IDs.
* road_network_200703_128.pkl is the node embedding of the road network.
* temporalDT_node2vec_0826_new_16.pkl is the temporal node embedding.

For the detailed format of dataset, please refer file "data_DT200915_example_train.pkl" 


## Training

To run the python code, make sure you have related packages.

```bash
cd Learning-to-Rank-Paths/

python train.py
```

## Testing

```bash
python test.py 
```

## Reference

```
@inproceedings{TKDE,
  author    = {Sean Bin Yang and
               Chenjuan Guo and
               Bin Yang},
  title     = {Context-Aware Path Ranking in Road Networks},
  booktitle = {IEEE Transactions on Knowledge and Data Engineering, 2020},
  year      = {2020},
}
```
