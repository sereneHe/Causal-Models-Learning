# Algorithm Discription:<br>
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------
![ANM-NCPOP](/images/logo.png)

# Input Data Discription:<br>
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------

## 1. Two Dimensions Causality Data:<br>

For two dimensions causality data, all useful infomation is extracted and saved as .npz file for further analysis.<br>

| File Type      | Example Data                                                    | Test File Name                             | File Name              |
| -------------- | --------------------------------------------------------------- | ------------------------------------------ | ---------------------- |
| 1.1. npz       | 6 Notes 15 Edges Synthetic Data With Additive linear Gauss Noise| linearGauss_6_15.npz                       | linearGauss_6_15       |
| 1.2. tar.gz    | 18V_55N_Wireless Data                                           | 18V_55N_Wireless.tar.gz                    | 18V_55N_Wireless       |
| 1.3. csv       | Process Data                                                    | real_dataset_processed.csv, true_graph.csv | real_dataset_processed |
| 2.1. tsv       | Krebs_Cycle Time Series                                         | series_list.csv, Krebs_Cycle_TS Folder     | Krebs_Cycle            | 
| 3.1. npz       | 6 Notes 15 Edges Synthetic Data With Additive linear Gauss Noise| linearGauss_6_15.npz                       | linearGauss_6_15       |

For more information refer to Input Data Discription Section.

## 3. Synthetic Datasets:<br>
To generate synthetic data, use BuiltinDataSet function in /Datasets.<br>

3.1. ** Two Dimensions Causality Data**<br>
- Example: 6 Notes 15 Edges Synthetic Data With Additive linear Gauss Noise<br>
- Raw data: x<br>
- Casaul matrix: y<br>
- Test File Name: linearGauss_6_15.npz<br>
- Test codes: 'file_name = linearGauss_6_15'<br>

3.2. ** Triple-dimensions Causality Data-Multiple Features Time Series:**<br>
To generate synthetic time series for causality learning, use BuiltinDataSet, ts_generation and data_generation functions in /Datasets.<br>

# To get start:<br>
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------

