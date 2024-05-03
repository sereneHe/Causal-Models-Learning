# Input Data Form:<br>
## 1. Two Dimensions Causality Data:<br>

For two dimensions causality data, all useful infomation is extracted and saved as .npz file for further analysis.<br>
1.1. **npz file**<br>
Example: linearGauss_6_15.npz<br>
Raw data: x<br>
Casaul matrix: y<br>
Test File Name: linearGauss_6_15.npz<br>
Test codes: file_name = linearGauss_6_15<br>

1.2. **tar.gz file**<br>
Example: 18V_55N_Wireless.tar.gz<br>
Raw data: 18V_55N_Wireless/Alarm.csv<br>
Casaul matrix: 18V_55N_Wireless/DAG.npy<br>
Test File Name: 18V_55N_Wireless.tar.gz<br>
Test codes: file_name = 18V_55N_Wireless<br>

1.3. **csv files**<br>
Example: Process Data<br>
Raw data: real_dataset_processed.csv<br>
Casaul matrix: true_graph.csv<br>
Test File Name: real_dataset_processed.csv, true_graph.csv<br>
Test codes: file_name = real_dataset_processed<br>

## 2. Triple-dimensions Causality Data-Multiple Features Time Series:

For multiple features time series data, all ts will be saved as a (Feature_num, Sample_num, Time) three dimension array for applying ANM-NCPOP.<br>

2.1. **tsv files**
Example: Krebs_Cycle Time Series<br>
Raw data:<br>
          Single sample trajectory with multiple features Time Series as shape of (F features, T timeSets) - incluing S smples, i.e. S number of .tsv files<br>
          2.1.1. **Series List**<br>
          series_list.csv file incluing a column with column_name = Series_num<br>
          2.1.2. **Series Folder**<br>
          Krebs_Cycle_TS/<br>
          series1713874182190.tsv<br>
          series1713874227485.tsv<br>
          series1713874308802.tsv<br>
          series1713874376137.tsv<br>
          series1713874426151.tsv<br>
          series1713874555527.tsv<br>
          series1713874613984.tsv<br>

Casaul matrix:<br>
Test File Name: Krebs_Cycle_TS/series1713874182190.tsv, series_list.csv <br>
Test codes: file_name = Krebs_Cycle<br>
| 列1      | 列2      | 列3      |
| -------- | -------- | -------- |
| 内容1    | 内容2    | 内容3    |
| 内容4    | 内容5    | 内容6    |
