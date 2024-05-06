if __name__ == "__main__":
    ############################################################################################################
    ############################################ SETTING File_PATH and file_name ###############################
    ############################################################################################################

    File_PATH = "/content/drive/MyDrive/Colab Notebooks/NCPOP-Colab Notebooks/Test_Causality_Datasets/Real_data/Krebs_Cycle/"
    method = 'linear'
    sem_type = 'gauss'
    num_nodes = 6
    num_edges = 15
    num_datasets = 10
    T=20
    # Weighted adjacency matrix for the target causal graph
    weighted_random_dag = DAG.erdos_renyi(n_nodes=num_nodes, n_edges=num_edges, seed=1)
    # _simulate_linear_sem(W =weighted_random_dag, n = num_datasets, sem_type = 'gauss', noise_scale=1.0)
    dataset = GenerateData(W=weighted_random_dag, n=num_datasets, T=20, method=method, sem_type=sem_type)
    true_dag, data = dataset.B, dataset.XX
    sname = method.capitalize()+sem_type.capitalize()+'_'+str(num_nodes)+'_'+str(num_edges)+'_TS.npz'
    np.savez(File_PATH + sname, x=dataset.XX, y=dataset.B)
    print('INFO: Check for '+sname+ '!')
