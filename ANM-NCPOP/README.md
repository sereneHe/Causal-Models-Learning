# Causal Model Discovery Problems in Learning Joint Multiple Dynamical Systems via Non-Commutative Polynomial Optimization(NCPOP)

Welcome to the repository for generating ANM NCPOP test data! The data can be any format that is supported by the Real_Data_Standardization() function, currently including npy, tar.gz, csv and tsv. For generating artificial data, please refer to Synthetic_Data_Generation part.

# Input Data Form:
---------------------------------------------------------------------------------------------------------------------------------------------------------

## File Types ##

*   npz file

Storing causality Data as NumPy array x and y under a npz file.

*   tar.gz file

Archiving and compressing causality files and folders as a tar.gz file.

*   csv files

Raw data and casaul matrix are saved as separate csv files.
   
*   tsv files

Single sample trajectory with multiple features Time Series as shape of (F features, T timeSets) - incluing S smples, i.e. S number of .tsv files

# Example:
In our example, multiple features time series are saved under Krebs_Cycle.npz as a casaul matrix and an array, separately.

* __x__: is an array in shape(F, S, T), where the number of row F is features_num, the number of column S is smples_num and the number of deep T is timesets.
* __y__: is a nonsymmetric square matrix.


Synthetic_Data_Generation
---------------------------------------------------------------------------------------------------------------------------------------------------------
Weighted random DAG is produced according to num_nodes and num_edges. Test raw datasets is the time series generating from weighted random DAG and SEM type.

# Example:

In our example, Weighted random DAG and raw datasets are saved under LinearGauss_6_15_TS.npz as a casaul matrix and an array, separately.

# Output Data Form
---------------------------------------------------------------------------------------------------------------------------------------------------------
Causality Data stored as NumPy array x and matrix y under a npz file.

* **Raw_data:**

An array saves F features, S smples and T timesets Time Series.

* **True_dag:**

A causal matrix is a nonsymmetric matrix saved as the shape of (F features, F features) 


