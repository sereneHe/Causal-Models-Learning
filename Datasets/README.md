# Datasets for Causal Structure Learning

## Synthetic datasets
We provide codes for generating synthetic datasets used in '[Causal Discovery with Reinforement Learning](../research/Causal%20Discovery%20with%20RL)'. Please see the [example notebook](examples_to_generate_synthetic_datasets.ipynb) for further details.

## Real datasets

### Telecom causal dataset
We release a very challenging [dataset](https://github.com/zhushy/causal-datasets/tree/master/Real_Dataset) from real telecommunication networks, to find causal structures based on time series data. 

### Data format
- **real_dataset_processed.csv**: each row counts the numbers of occurrences of the alarms (A_i,i=0,1,...,56) in 10 minutes. The rows are arranged in the time order, i.e., first 10 mins., second 10 mins., etc.
- **true_graph.csv**: the underlying causal relationships, according to expert experience.  `(i,j)=1` implies an edge `i->j`.

### PCIC competition datasets

The real datasets (id:10, 21, 22) used in [PCIC Causal Discovery Competition 2021 ](https://competition.huaweicloud.com/information/1000041487/introduction) have been made available online: [link](https://github.com/gcastle-hub/dataset).

The following table contains F1-scores of some competitive causal discovery algorithms on these datasets. 
