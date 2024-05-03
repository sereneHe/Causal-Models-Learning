# Input Data Description:
## Synthetic Datasets:<br>
-------------------------------------------------------------------------------------------------------------------------

## 3. Two Dimensions Causality Data<br>
To generate synthetic data and store as NumPy array x and y under a npz file, use BuiltinDataSet function in /Datasets.<br>

- Raw data(x): causality data with 6 features, 100 smples
- Causal_matrix(y): array as shape of (6 features, 6 features)
- Example: linearGauss_6_15.npz<br>

## 4. Triple-dimensions Causality Data-Multiple Features Time Series<br>
To generate synthetic time series for causality learning, use BuiltinDataSet, ts_generation and data_generation functions in /Datasets.<br>

- Raw data(x_ts): causality data with 6 features, 100 smples and 5 timesets Time Series<br>
- Casaul matrix(y): array as shape of (6 features, 6 features)<br>
- Example: linearGauss_6_15_5_ts.npz<br>

| File Type | Example Data                                                           | Test File Name                                  | File Name                 |
| --------- | ---------------------------------------------------------------------- | ----------------------------------------------- | ------------------------- |
| npz       | 6 Notes 15 Edges Synthetic Data with Additive linear Gauss Noise       | linearGauss_6_15.npz                            | linearGauss_6_15          |
| npz       | 6 Notes 15 Edges Synthetic Time Series with Additive linear Gauss Noise| linearGauss_6_15_5_ts.npz                       | linearGauss_6_15_5_ts     |

