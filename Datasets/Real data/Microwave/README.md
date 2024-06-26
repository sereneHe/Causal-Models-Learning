The real datasets (id:10, 21, 22) used in [PCIC Causal Discovery Competition 2021 ](https://competition.huaweicloud.com/information/1000041487/introduction) have been made available online: [link](https://github.com/gcastle-hub/dataset).
# Datasets
-------------------------------------------------------
Examples: 25V_474N_Microwave.tar.gz, 24V_439N_Microwave.tar.gz, 18V_55N_Wireless.tar.gz.

We provide a total of 24 datasets, 12 with accompanying network topology information and the other 12 without such network topology information. If you download the datasets from our competition site, you will find that K datasets are stored in separated directories named from 1 to K, and each dataset includes the following data files:

* **Alarm.csv:** Historical alarm data

(1)    Format: [alarm_id, device_id, start_timestamp, end_timestamp]

(2)    In the alarm data file we provide historic alarm information. Each row denotes an alarm record which contains the alarm ID (i.e., the alarm type), the device where the alarm occurred, the start timestamp, and the end timestamp. For privacy, every alarm id is encoded to an integer number starting from 0 to N-1, where N is the number of the alarm types. Each device ID is likewise encoded to an integer number starting from 0 to M-1, where M is the number of the devices.




* **Topology.npy:** The connections between devices (only for the datasets with topology).

(1)    Format: an M*M NumPy array, with M being the number of the devices in the network.

(2)    This NumPy file stores the binary symmetric adjacency matrix for the network topology which is an undirected graph. For example, the element which is in the i-th row and j-th column of the matrix equals 1 (0) means the existence (resp. non-existence) of an undirected link between the device i and the device j.

(3)    Example (M=10):



* **DAG.npy:** The true causal graph (only 5 graphs are made public during the competition).

(1)    Format: an N*N NumPy array, where N is the number of the alarm types.

(2)    Similar to the topology, DAG.npy stores the binary adjacency matrix for the true causal alarm graph. The graph is labeled manually by experts or, for the synthetic datasets, the pre-set causal assumptions. For example, the element which is in the i-th row and j-th column of the matrix equals 1 (0) means the existence (resp. non-existence) of an directed edge from the alarm type i to alarm type j.

(3)    Example (N=10):



**Note:**
 For datasets without topology, even though the alarm file record the device where each alarm occurs, these different devices are independent, i.e., there is no known relation between these devices.   
