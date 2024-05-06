method = 'linear'
sem_type = 'gauss'
num_nodes = 6
num_edges = 15
num_datasets = 10
T=20
# Weighted adjacency matrix for the target causal graph
weighted_random_dag = DAG.erdos_renyi(n_nodes=num_nodes, n_edges=num_edges, seed=1)
# _simulate_linear_sem(W =weighted_random_dag, n = num_datasets, sem_type = 'gauss', noise_scale=1.0)
dataset = ANMNCPOP_GenerateData(W=weighted_random_dag, n=num_datasets, T=20, method=method, sem_type=sem_type)
true_dag, data = dataset.B, dataset.XX
# print(weighted_random_dag)
print(true_dag)
print(data.shape)
np.save('ANMNCPOP_GenerateTimeIID.npy', data)
