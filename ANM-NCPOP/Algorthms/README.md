# Input Data Form:
## 1. Two Dimensions Causality Data:

For two dimensions causality data, all useful infomation is extracted and saved as .npz file for further analysis.
1.1. **npz file**
Example: linearGauss_6_15.npz
Raw data: x
Casaul matrix: y
Test File Name: linearGauss_6_15.npz
Test codes: file_name = linearGauss_6_15

1.2. **tar.gz file**
Example: 18V_55N_Wireless.tar.gz
Raw data: 18V_55N_Wireless/Alarm.csv
Casaul matrix: 18V_55N_Wireless/DAG.npy
Test File Name: 18V_55N_Wireless.tar.gz
Test codes: file_name = 18V_55N_Wireless

1.3. **csv files**
Example: Process Data
Raw data: real_dataset_processed.csv
Casaul matrix: true_graph.csv
Test File Name: real_dataset_processed.csv, true_graph.csv
Test codes: file_name = real_dataset_processed

## 2. Triple-dimensions Causality Data-Multiple Features Time Series:

For multiple features time series data, all ts will be saved as a (Feature_num, Sample_num, Time) three dimension array for applying ANM-NCPOP.

2.1. **tsv files**
Example: Krebs_Cycle Time Series
Raw data:
          Single sample trajectory with multiple features Time Series as shape of (F features, T timeSets) - incluing S smples, i.e. S number of .tsv files
          2.1.1. **Series List**
          series_list.csv file incluing a column with column_name = Series_num
          2.1.2. **Series Folder**
          Krebs_Cycle_TS/
          series1713874182190.tsv
          series1713874227485.tsv
          series1713874308802.tsv
          series1713874376137.tsv
          series1713874426151.tsv
          series1713874555527.tsv
          series1713874613984.tsv

Casaul matrix:
Test File Name: Krebs_Cycle_TS/series1713874182190.tsv, series_list.csv 
Test codes: file_name = Krebs_Cycle
