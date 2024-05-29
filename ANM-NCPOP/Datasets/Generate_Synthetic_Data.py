from BuiltinDataSet import DAG
import numpy as np
import networkx as nx
import logging
# from Generate_SyntheticData import*

class GenerateData(object):
    '''
    Simulate IID datasets for causal structure learning.

    Parameters
    ------------------------------------------------------------------------------------------------
    File_PATH
        save route
    n: int
        Number of samples for standard trainning dataset.
    T: int
        Number of timeseries for standard trainning dataset.
    method: str, (linear or nonlinear), default='linear'
        Distribution for standard trainning dataset.
    sem_type: str
        gauss, exp, gumbel, uniform, logistic (linear);
        mlp, mim, gp, gp-add, quadratic (nonlinear).
    nodes: series
        Notes of samples for standard trainning dataset.
    edges: series
        Edges of samples for standard trainning dataset.
        
    Returns
    ------------------------------------------------------------------------------------------------
    Raw_data: npz
            x：[d, n, T] sample time series 
            y: true_dag

    Examples
    -------------------------------------------------------------------------------------------------
    >>> method = 'linear'
    >>> sem_type = 'gauss'
    >>> nodes = range(6,15,3)
    >>> edges = range(10,20,5)
    >>> T=200
    >>> File_PATH = 'Test/'
    >>> _ts = Generate_Synthetic_Data(File_PATH, n=num_datasets, T, method, sem_type, nodes, edges)
    '''

    def __init__(self, File_PATH, W, n=1000, T=20, method='linear', sem_type='gauss', nodes, edges, noise_scale=1.0):
        nodes_num = len(nodes)
        edges_num = len(edges)
                
        ############################################## Create Artificial Dataset ###################################
        filename = method.capitalize()+'SEM_' + sem_type.capitalize() +'Noise'
        File_PATH_Base = File_PATH +'Results_'+ filename +'/'
        self.File_PATH_Datasets = File_PATH_Base + 'Datasets_'+ filename +'/'
        if not os.path.exists(self.File_PATH_Datasets):
            os.makedirs(self.File_PATH_Datasets)
        print('ANM-NCPOP INFO: Created Datasets_'+ method.capitalize()+ ' File!')

        count = 0
        tqdm_csv=os.listdir(self.File_PATH_Datasets)
        if len(tqdm_csv) != nodes_num* edges_num:
            print('ANM-NCPOP INFO: Generating '+ printname + ' Dataset!')
            if method == 'linear':
                for nn in nodes:
                    for ne in edges:
                        count = count +1
                        w = DAG.erdos_renyi(n_nodes=nn, n_edges=ne, seed=1)
                        self.B = (W != 0).astype(int)
                        self.XX = GenerateData._simulate_linear_sem(W, n, T, sem_type, noise_scale)
                        data_name = filename+'_'+str(nn)+'Nodes_'+str(ne)+'Edges_TS'
                        np.savez(self.File_PATH_Datasets +data_name+'.npz', x=XX , y=B)
                        logging.info('ANM-NCPOP INFO: Finished synthetic dataset')
                        print('ANM-NCPOP INFO: '+ data_name + ' IS DONE!')
                print('ANM-NCPOP INFO: '+ str(count) + ' datasets are generated!')
            elif method == 'nonlinear':
                for nn in nodes:
                    for ne in edges:
                        count = count +1
                        w = DAG.erdos_renyi(n_nodes=nn, n_edges=ne, seed=1)
                        self.B = (W != 0).astype(int)
                        self.XX = GenerateData._simulate_nonlinear_sem(W, n, T, sem_type, noise_scale)
                        data_name = filename+'_'+str(nn)+'Nodes_'+str(ne)+'Edges_TS'
                        np.savez(self.File_PATH_Datasets +data_name+'.npz', x=XX , y=B)
                        logging.info('ANM-NCPOP INFO: Finished synthetic dataset')
                        print('ANM-NCPOP INFO: '+ data_name + ' IS DONE!')
                print('ANM-NCPOP INFO: '+ str(count) + ' datasets are generated!')

        else:
            print('ANM-NCPOP INFO: Finished '+ printname+' dataset generation!')

    @staticmethod
    def _simulate_linear_sem(W, n, T, sem_type, noise_scale):
        """
        Simulate samples from linear SEM with specified type of noise.
        For uniform, noise z ~ uniform(-a, a), where a = noise_scale.

        Parameters
        ----------
        W: np.ndarray
            [d, d] weighted adj matrix of DAG.
        n: int
            Number of samples, n=inf mimics population risk.
        T: int
        Number of timeseries for standard trainning dataset.
        sem_type: str
            gauss, exp, gumbel, uniform, logistic.
        noise_scale: float
            Scale parameter of noise distribution in linear SEM.

        Return
        ------
        XX: np.ndarray
            [T, n, d] sample matrix, [d, d] if n and T=inf
        """
        def _simulate_single_equation(X, w, scale):
            """X: [n, num of parents], w: [num of parents], x: [n]"""
            if sem_type == 'gauss':
                z = np.random.normal(scale=scale, size=T)
                x = X @ w + z
            elif sem_type == 'exp':
                z = np.random.exponential(scale=scale, size=T)
                x = X @ w + z
            elif sem_type == 'gumbel':
                z = np.random.gumbel(scale=scale, size=T)
                x = X @ w + z
            elif sem_type == 'uniform':
                z = np.random.uniform(low=-scale, high=scale, size=T)
                x = X @ w + z
            elif sem_type == 'logistic':
                x = np.random.binomial(1, sigmoid(X @ w)) * 1.0
            else:
                raise ValueError('Unknown sem type. In a linear model, \
                                 the options are as follows: gauss, exp, \
                                 gumbel, uniform, logistic.')
            return x

        d = W.shape[0]
        if noise_scale is None:
            scale_vec = np.ones(d)
        elif np.isscalar(noise_scale):
            scale_vec = noise_scale * np.ones(d)
        else:
            if len(noise_scale) != d:
                raise ValueError('noise scale must be a scalar or has length d')
            scale_vec = noise_scale
        G_nx =  nx.from_numpy_array(W, create_using=nx.DiGraph)
        if not nx.is_directed_acyclic_graph(G_nx):
            raise ValueError('W must be a DAG')
        if np.isinf(T):  # population risk for linear gauss SEM
            if sem_type == 'gauss':
                # make 1/d X'X = true cov
                X = np.sqrt(d) * np.diag(scale_vec) @ np.linalg.inv(np.eye(d) - W)
                return X
            else:
                raise ValueError('population risk not available')
        # empirical risk
        ordered_vertices = list(nx.topological_sort(G_nx))
        assert len(ordered_vertices) == d
        X = np.zeros([T, d])
        XX = np.zeros((T, n, d))
        for j in ordered_vertices:
            parents = list(G_nx.predecessors(j))
            X[:, j] = _simulate_single_equation(X[:, parents], W[parents, j], scale_vec[j])
        for ns in range(n):
            XX[:, ns] = X
        return XX

    @staticmethod
    def _simulate_nonlinear_sem(W, n, T, sem_type, noise_scale):
        """
        Simulate samples from nonlinear SEM.

        Parameters
        ----------
        B: np.ndarray
            [d, d] binary adj matrix of DAG.
        n: int
            Number of samples.
        T: int
            Number of times.
        sem_type: str
            mlp, mim, gp, gp-add, or quadratic.
        noise_scale: float
            Scale parameter of noise distribution in linear SEM.

        Return
        ------
        XX: np.ndarray
            [T, n, d] sample matrix
        """
        if sem_type == 'quadratic':
            return GenerateData._simulate_quad_sem(W, T, noise_scale)

        def _simulate_single_equation(X, scale):
            """X: [n, num of parents], x: [n]"""
            z = np.random.normal(scale=scale, size=n)
            pa_size = X.shape[1]
            if pa_size == 0:
                return z
            if sem_type == 'mlp':
                hidden = 100
                W1 = np.random.uniform(low=0.5, high=2.0, size=[pa_size, hidden])
                W1[np.random.rand(*W1.shape) < 0.5] *= -1
                W2 = np.random.uniform(low=0.5, high=2.0, size=hidden)
                W2[np.random.rand(hidden) < 0.5] *= -1
                x = sigmoid(X @ W1) @ W2 + z
            elif sem_type == 'mim':
                w1 = np.random.uniform(low=0.5, high=2.0, size=pa_size)
                w1[np.random.rand(pa_size) < 0.5] *= -1
                w2 = np.random.uniform(low=0.5, high=2.0, size=pa_size)
                w2[np.random.rand(pa_size) < 0.5] *= -1
                w3 = np.random.uniform(low=0.5, high=2.0, size=pa_size)
                w3[np.random.rand(pa_size) < 0.5] *= -1
                x = np.tanh(X @ w1) + np.cos(X @ w2) + np.sin(X @ w3) + z
            elif sem_type == 'gp':
                from sklearn.gaussian_process import GaussianProcessRegressor
                gp = GaussianProcessRegressor()
                x = gp.sample_y(X, random_state=None).flatten() + z
            elif sem_type == 'gp-add':
                from sklearn.gaussian_process import GaussianProcessRegressor
                gp = GaussianProcessRegressor()
                x = sum([gp.sample_y(X[:, i, None], random_state=None).flatten()
                        for i in range(X.shape[1])]) + z
            else:
                raise ValueError('Unknown sem type. In a nonlinear model, \
                                 the options are as follows: mlp, mim, \
                                 gp, gp-add, or quadratic.')
            return x

        B = (W != 0).astype(int)
        d = B.shape[0]
        if noise_scale is None:
            scale_vec = np.ones(d)
        elif np.isscalar(noise_scale):
            scale_vec = noise_scale * np.ones(d)
        else:
            if len(noise_scale) != d:
                raise ValueError('noise scale must be a scalar or has length d')
            scale_vec = noise_scale
        X = np.zeros([T, d])
        XX = np.zeros((T, n, d))
        G_nx =  nx.from_numpy_array(B, create_using=nx.DiGraph)
        ordered_vertices = list(nx.topological_sort(G_nx))
        assert len(ordered_vertices) == d
        for j in ordered_vertices:
            parents = list(G_nx.predecessors(j))
            X[:, j] = _simulate_single_equation(X[:, parents], scale_vec[j])
        for ns in range(n):
            XX[:, ns] = X

        return XX
