# Learning-to-Rank-Paths

This repository holds the code used in our **TKDE-20** paper: [Context-Aware Path Ranking in Road Networks]().

## Requirements

* Ubuntu OS (16.04)
* Python = 3.6
* Numpy >= 1.16.2
* Pickle
* Tensorflow = 1.12.0 (both 0.4 and 1.0 are tested)

Please refer to the source code to install the required packages in Python.

## Dataset

The dataset contains 180 million GPS records for two years period from 166 drivers in Denmark. The sampling rate of the GPS data is 1 Hz. We split the GPS records ino 22612 trajectories representing different trips. Here we just give 64 paths for other user to test since the copyright reason. In the Data file there have four files, each is:
* data_DT200915_example_train.pkl is a sample data file. The data format is (x_data, x_temporal,x_driver,y_train,tt_train,fc_train,len_train), here x_data is path sequence. x_temporal is temporal information for specificed paths based on the departure time. x_driver is the additional information of driver. tt_train (travel time), fc_train (fuel consumpation) and len_train (travel distance) is the additional information of path.
* driverid_onehot_0823_166.pkl is onehot format in terms of dirver id.
* road_network_200703_128.pkl is node embedding of denmark road network.
* temporalDT_node2vec_0826_new_16.pkl is temporal node embedding.

For the detailed format of dataset, please refer file "data_DT200915_example_train.pkl" 


## Training

To run the python code, make sure you have related packages.

```bash
cd Learning-to-Rank-Paths/

python train.py
```

## Testing

```bash
python test-mul.py 
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
