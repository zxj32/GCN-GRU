# GCN-GRU

This is a TensorFlow implementation of the GCN-GRU model as described in our paper:
 
Xujiang Zhao, Feng Chen, Jin-Hee Cho [Deep Learning for Predicting Dynamic Uncertain Opinions in Network Data], Bigdata (2018)

GCN-GRU model are end-to-end trainable deep learning models for dynamic uncertain opinions prediction in dynamic network data. 

![GCN-GRU](git_figure.PNG)


## Installation

1. Clone this repository.
   ```sh
   git clone https://github.com/zxj32/GCN-GRU
   cd GCN-GRU
   ```

2. Install the dependencies. The code should run with TensorFlow 1.0 and newer.
   ```sh
   pip install -r requirements.txt  # or make install
   ```

## Requirements
* TensorFlow (1.0 or later)
* python 2.7
* networkx
* scikit-learn
* scipy

## Run the demo

```bash
python GCN_GRU.py
```

## Data

In order to use your own data, you have to provide 
* an N by N adjacency matrix (N is the number of nodes), and
* an N by D by T feature matrix (D is the number of features per node, T is time length) -- optional

Have a look at the `generate_train_test_dc_noise()` function in `traffic_data/read_dc.py` for an example.

In this example, we load traffic data. The original datasets can be found here: http://inrix.com/publicsector.asp
our paper also inlcude epinion dataset: http://www.trustlet.org/downloaded 
spammer dataset: https://linqs-data.soe.ucsc.edu/public/social_spammer/

## Models

You can choose the following model: 
* `GCN-GRU`: GCN_GRU.py

## Cite

Please cite our paper if you use this code in your own work:

```
@article{xujiang2018gcn_gru,
  title={Deep Learning for Predicting Dynamic Uncertain Opinions in Network Data},
  author={Xujiang Zhao, Feng Chen and Jin-Hee Cho},
  journal={Bigdata},
  year={2018}
}
```
