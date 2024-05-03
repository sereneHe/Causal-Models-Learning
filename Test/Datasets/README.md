# Input Data Description:
## 1. Two Dimensions Causality Data:

1.1. **npz file**<br>
Storing causality Data as NumPy array x and y under a npz file.
- Example: Telephone<br>
- Raw data: x<br>
- Casaul matrix: y<br>
- Test File Name: Telephone.npz<br>

1.2. **tar.gz file**<br>
Archiving and compressing causality files and folders as a tar.gz file.
- Example: 18V_55N_Wireless Data<br>
- Raw data: 18V_55N_Wireless/Alarm.csv<br>
- Casaul matrix: 18V_55N_Wireless/DAG.npy<br>
- Test File Name: 18V_55N_Wireless.tar.gz<br>

1.3. **csv files**<br>
Raw data and casaul matrix are saved as separate csv files.
- Example: Process Data<br>
- Raw data: real_dataset_processed.csv<br>
- Casaul matrix: true_graph.csv<br>
- Test File Name: real_dataset_processed.csv, true_graph.csv<br>

## 2. Triple-dimensions Causality Data-Multiple Features Time Series:

For multiple features time series data, all ts will be saved as a (Feature_num, Sample_num, Time) three dimension array for applying ANM-NCPOP.<br>

2.1. **tsv files**
- Example: Krebs_Cycle Time Series<br>
- Raw data: observation  Time Series data with F features, S smples and T timesets stored as NumPy array x.<br>
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

- Casaul matrix: array as shape of (F features, F features) <br>
- Test File Name: Krebs_Cycle_TS/series1713874182190.tsv, ..., series_list.csv <br>
