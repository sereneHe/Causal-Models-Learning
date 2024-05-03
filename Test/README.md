# Algorithm Discription:<br>
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Causal Model Discovery Problems in Learning Joint Multiple Dynamical Systems via Non-Commutative Polynomial Optimization(NCPOP)

![ANM-NCPOP](/images/logo.png)

# Input Data:<br>
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------

## 1. Real Data:<br>
For two dimensions causality data, all useful infomation is extracted and saved as .npz file for further analysis.<br>

| File Type | Example Data                                                    | Test File Name                             | File Name              |
| --------- | --------------------------------------------------------------- | ------------------------------------------ | ---------------------- |
| npz       | Telephone                                                       | Telephone.npz                              | Telephone              |
| tar.gz    | 18V_55N_Wireless Data                                           | 18V_55N_Wireless.tar.gz                    | 18V_55N_Wireless       |
| csv       | Process Data                                                    | real_dataset_processed.csv, true_graph.csv | real_dataset_processed |
| tsv       | Krebs_Cycle Time Series                                         | series_list.csv, Krebs_Cycle_TS Folder     | Krebs_Cycle            | 

For more information refer to Input Data Description Section.

## 2. Synthetic Datasets:<br>

2.1. ** Two Dimensions Causality Data**<br>
To generate synthetic data and store as NumPy array x and y under a npz file, use BuiltinDataSet function in /Datasets.<br>

- Raw data(x): causality data with F features, S smples
- Causal_matrix(y): array as shape of (F features, F features)
- Example: linearGauss_6_15.npz<br>

2.2. ** Triple-dimensions Causality Data-Multiple Features Time Series:**<br>
To generate synthetic time series for causality learning, use BuiltinDataSet, ts_generation and data_generation functions in /Datasets.<br>

- Raw data(x_ts): causality data with F features, S smples and T timesets Time Series<br>
- Casaul matrix(y): array as shape of (F features, F features)<br>
- Example: linearGauss_6_15_ts.npz<br>

| File Type | Example Data                                                           | Test File Name                                | File Name                 |
| --------- | ---------------------------------------------------------------------- | --------------------------------------------- | ------------------------- |
| npz       | 6 Notes 15 Edges Synthetic Data with Additive linear Gauss Noise       | linearGauss_6_15.npz                          | linearGauss_6_15          |
| npz       | 6 Notes 15 Edges Synthetic Time Series with Additive linear Gauss Noise| linearGauss_6_15_ts.npz                       | linearGauss_6_15_ts       |

For more information refer to Input Data Description Section.

# To get start:<br>
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------

