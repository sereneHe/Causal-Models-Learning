# Causal Model Discovery Problems in Learning Joint Multiple Dynamical Systems via Non-Commutative Polynomial Optimization(NCPOP)

Welcome to the repository for data standardization in causal discovery!

The data can be any format that is supported by the ANMPOP_ReadData() function, currently including npy, tar.gz, csv and tsv.

# Standardized Data Form:
Causality Data stored as NumPy array x and y under a npz file.

**Raw_data x:**
Causality data with F features, S smples and T timesets Time Series.

**True_dag y:**
causal_matrix array as shape of (F features, F features) - Learned underlying causal relationships between obeservations, according to expert experience or ground true causality. 

# Input Data Form:
Two Dimensions Causality Data:
For two dimensions causality data, all useful infomation is extracted and saved as .npz file for further analysis.
1. **npz file**

storing causality Data as NumPy array x and y under a npz file.
2. **tar.gz file**

archiving and compressing causality files and folders as a tar.gz file.
3. **csv files**

raw data and casaul matrix are saved as separate csv files.
   
Triple-dimensions Causality Data-Multiple Features Time Series:
For multiple features time series data, all ts will be saved as a (Feature_num, Sample_num, Time) three dimension array for applying ANM-NCPOP.
1. **tsv files**

Single sample trajectory with multiple features Time Series as shape of (F features, T timeSets) - incluing S smples, i.e. S number of .tsv files
