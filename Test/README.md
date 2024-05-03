# Algorithm Discription:<br>
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------
![ANM-NCPOP](/images/logo.png)

# Input Data Discription:<br>
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------

## 1. Two Dimensions Causality Data:<br>

For two dimensions causality data, all useful infomation is extracted and saved as .npz file for further analysis.<br>
1.1. **npz file**<br>
- Example: Telephone<br>
- Raw data: x<br>
- Casaul matrix: y<br>
- Test File Name: Telephone.npz<br>
- Test codes: 'file_name = Telephone'<br>

1.2. **tar.gz file**<br>
- Example: 18V_55N_Wireless Data<br>
- Raw data: 18V_55N_Wireless/Alarm.csv<br>
- Casaul matrix: 18V_55N_Wireless/DAG.npy<br>
- Test File Name: 18V_55N_Wireless.tar.gz<br>
- Test codes: 'file_name = 18V_55N_Wireless'<br>

1.3. **csv files**<br>
- Example: Process Data<br>
- Raw data: real_dataset_processed.csv<br>
- Casaul matrix: true_graph.csv<br>
- Test File Name: real_dataset_processed.csv, true_graph.csv<br>
- Test codes: 'file_name = real_dataset_processed'<br>

## 2. Triple-dimensions Causality Data-Multiple Features Time Series:

For multiple features time series data, all ts will be saved as a (Feature_num, Sample_num, Time) three dimension array for applying ANM-NCPOP.<br>

2.1. **tsv files**
- Example: Krebs_Cycle Time Series<br>
- Raw data:<br>
  Single sample trajectory with multiple features Time Series as shape of (F features, T timeSets) - incluing S smples, i.e. S number of .tsv files<br>
          2.1.1. **Series List**<br>
          - series_list.csv file incluing a column with column_name = Series_num<br>
          2.1.2. **Series Folder**<br>
          - Krebs_Cycle_TS Folder<br>
          series1713874182190.tsv<br>
          series1713874227485.tsv<br>
          series1713874308802.tsv<br>
          series1713874376137.tsv<br>
          series1713874426151.tsv<br>
          series1713874555527.tsv<br>
          series1713874613984.tsv<br>
          ...

- Casaul matrix:<br>
- Test File Name: Krebs_Cycle_TS/series1713874182190.tsv, ..., series_list.csv <br>
- Test codes: 'file_name = Krebs_Cycle'<br>

## Summary Table<br>

| File Type      | Example Data                                                    | Test File Name                             | File Name              |
| -------------- | --------------------------------------------------------------- | ------------------------------------------ | ---------------------- |
| 1.1. npz       | 6 Notes 15 Edges Synthetic Data With Additive linear Gauss Noise| linearGauss_6_15.npz                       | linearGauss_6_15       |
| 1.2. tar.gz    | 18V_55N_Wireless Data                                           | 18V_55N_Wireless.tar.gz                    | 18V_55N_Wireless       |
| 1.3. csv       | Process Data                                                    | real_dataset_processed.csv, true_graph.csv | real_dataset_processed |
| 2.1. tsv       | Krebs_Cycle Time Series                                         | series_list.csv, Krebs_Cycle_TS Folder     | Krebs_Cycle            | 
| 3.1. npz       | 6 Notes 15 Edges Synthetic Data With Additive linear Gauss Noise| linearGauss_6_15.npz                       | linearGauss_6_15       |

For more information refer to Input Data Discription Section above.

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

