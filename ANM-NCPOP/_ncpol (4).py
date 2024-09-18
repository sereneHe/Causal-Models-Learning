#!/usr/bin/env python
# coding: utf-8

# In[20]:


get_ipython().system('wget https://bootstrap.pypa.io/pip/3.6/get-pip.py')
get_ipython().system('python3 get-pip.py')
get_ipython().system('pip3 install gcastle')
get_ipython().system('pip3 install PyPDF2')


# coding: utf-8
## Xiaoyu He ##

import numpy as np
import pandas as pd
from sklearn.preprocessing import scale
from itertools import combinations
from castle.common import BaseLearner
from independence_tests import hsic_test
from inputlds import*
from functions import*
from ncpol2sdpa import*
from base import*
from math import sqrt
from sympy.physics.quantum import HermitianOperator, Operator

'''
class NCPOLR_Multioperators(object):
    """Estimator based on NCPOP Regressor

    References
    ----------
    Quan Zhou https://github.com/Quan-Zhou/Proper-Learning-of-LDS/blob/master/ncpop/functions.py
    Zhou, Q. and Mareˇcek, J.(2023). Learning of linear dynamical systems as a non-commutative polynomial optimization problem. IEEE Transactions on Automatic Control

    Examples
    --------
    >>> import pandas as pd
    >>> import sys
    >>> sys.path.append("/home/zhouqua1")
    >>> sys.path.append("/home/zhouqua1/NCPOP")
    >>> from inputlds import*
    >>> from functions import*
    >>> from ncpol2sdpa import*
    >>> import numpy as np
    >>> from ..Clustering.datasets import load_heartrate
    >>> from sympy.physics.quantum import HermitianOperator, Operator
    >>> Test_Data = load_heartrate().reshape(600,3)
    >>> level = 1
    >>> N = len(np.transpose(Test_Data))
    >>> NCPOLR2().estimate(Test_Data,N,level)
    """

    def __init__(self, **kwargs):
        super(NCPOLR_Multioperators, self).__init__()


    def generate_multioperators(self, name, n_vars, m_vars, hermitian=None, commutative=False):
        """Generates a two dimensions of commutative or noncommutative operators

        Parameters
        ----------
        name[str type]: The prefix in the symbolic representation of the noncommuting
                    variables. This will be suffixed by a number from 00 to
                    (n_vars-1)(m_vars-1) if n_vars > 1 and m_vars > 1.
        n_vars[int type]: The number of variables row.
        m_vars[int type]: The number of variables column.
        hermitian[bool type]: Optional parameter to request Hermitian variables .
        commutative[bool type]: Optional parameter to request commutative variables.
                            Commutative variables are Hermitian by default.

        Returns
        -------
        Array of class variables[array type]: `sympy.physics.quantum.operator.Operator` or
                  `sympy.physics.quantum.operator.HermitianOperator`

        Examples
        --------
        >>> from sympy.physics.quantum import HermitianOperator, Operator
        >>> generate_multioperators('y', 2, 3, commutative=True)
        #   array([[y00, y01, y02],
        #   [y10, y11, y12]], dtype=object)

        """

        variables = []
        variables1 = []
        variables2 = []
        for i in range(n_vars):
            if n_vars > 1:
                var_name1 = '%s%s' % (name, i)
            else:
                var_name1 = '%s' % name
            if hermitian is not None and hermitian:
                variables1.append(HermitianOperator(var_name1))
            else:
                variables1.append(Operator(var_name1))
            variables1[-1].is_commutative = commutative
        for n in range(len(variables1)):
            for j in range(m_vars):
                if m_vars > 1:
                    var_name = '%s%s' % (variables1[n], j)
                else:
                    var_name = '%s' % variables1[n]
                if hermitian is not None and hermitian:
                    variables2.append(HermitianOperator(var_name))
                else:
                    variables2.append(Operator(var_name))
                variables2[-1].is_commutative = commutative
        var = np.matrix(np.array(variables2).reshape(n_vars,m_vars))
        var = np.array(variables2).reshape(n_vars,m_vars)
        #print(variables1,variables2)
        return var

    def estimate2(self, x, Y):
        """Fit Estimator based on NCPOP Regressor model and predict y or produce residuals.
        The module converts a noncommutative optimization problem provided in SymPy
        format to an SDPA semidefinite programming problem.

        Parameters
        ----------
        x: array
            Variable seen as cause
        y: array
            Variable seen as effect

        Returns
        -------
        y_predict: array
            regression predict values of y or residuals
        """

        level = 2
        n = 2
        # Y = np.transpose(Y)
        T = len(np.transpose(x))
        m = len(x)


        # Decision Variables
        A = self.generate_multioperators("A", n_vars=n, m_vars=n, hermitian=True, commutative=False)
        B = self.generate_multioperators("B", n_vars=m, m_vars=n, hermitian=True, commutative=False)
        # phi = self.generate_multioperators("phi", n_vars=n, m_vars=T+1, hermitian=True, commutative=False)
        w = self.generate_multioperators("w", n_vars=n, m_vars=T, hermitian=True, commutative=False)
        v = self.generate_multioperators("v", n_vars=m, m_vars=T, hermitian=True, commutative=False)
        f = self.generate_multioperators("f", n_vars=m, m_vars=T, hermitian=True, commutative=False)

        # Objective
        obj = sum((Y[mm][t]-f[mm][t])*2 for mm in range(m) for t in range(T))

        # Constraints
        ine1 = [x[nn][t] - A[nn][nn]*x[nn][t-1] - w[nn][t] for nn in range(n) for t in range(1, T)]
        ine2 = [-x[nn][t] + A[nn][nn]*x[nn][t-1] + w[nn][t] for nn in range(n) for t in range(1, T)]
        ine3 = [f[mm][t] - B[mm][nn]*x[nn][t] - v[mm][t] for nn in range(n) for t in range(1, T) for mm in range(m)]
        ine4 = [-f[mm][t] + B[mm][nn]*x[nn][t] + v[mm][t] for nn in range(n) for t in range(1, T) for mm in range(m)]
        ines = ine1+ine2+ine3+ine4 #+ine5

        # Solve the NCPO
        variables = [var for array in [A, B, w, v, f] for var in array.flatten()]
        sdp = SdpRelaxation(variables=variables, verbose=1)
        # sdp = SdpRelaxation(variables = flatten([A,B,w,v,f]),verbose = 1)
        sdp.get_relaxation(level, objective=obj, inequalities=ines)
        sdp.solve(solver='mosek')
        
        #with sdp.SolverFactory("mosek") as solver:
            ## options - MOSEK parameters dictionary, using strings as keys (optional)
            ## tee - write log output if True (optional)
            ## soltype - accepts three values : bas, itr and itg for basic,
            ## interior point and integer solution, respectively. (optional)
            #solver.solve(model, options = {'dparam.optimizer_max_time':  100.0,
            #                               'iparam.intpnt_solve_form':   int(mosek.solveform.dual)},
            #                    tee = True, soltype='itr')

            ## Save data to file (after solve())
            #solver._solver_model.writedata("dump.task.gz")
            
        #sdp.solve(solver='sdpa', solverparameters={"executable":"sdpa_gmp","executable": "C:/Users/zhouq/Documents/sdpa7-windows/sdpa.exe"})
        print(sdp.primal, sdp.dual, sdp.status)

        if(sdp.status != 'infeasible'):
            print('ok.')
            est_noise = []
            for i in range(T):
                est_noise.append(sdp[q[i]])
            print(est_noise)
            return est_noise
        else:
            print('Cannot find feasible solution.')
            return

    def estimate2_old(self, X, Y):
        """Fit Estimator based on NCPOP Regressor model and predict y or produce residuals.
        The module converts a noncommutative optimization problem provided in SymPy
        format to an SDPA semidefinite programming problem.
        Define a function for solving the NCPO problems with
        given standard deviations of process noise and observtion noise,
        length of  estimation data and required relaxation level.

        Parameters
        ----------
        X : array
            Variable seen as cause
        Y: array
            Variable seen as effect

        Returns
        -------
        obj: num
            Objective value in optima
        y_predict: array
            regression predict values of y or residuals
        """
        n = 2
        m = 3
        level = 1
        T= len(Y)
        Y = np.transpose(Y)
        print(n, m, T)

        # Decision Variables
        A = self.generate_multioperators("A", n_vars=n, m_vars=n, hermitian=True, commutative=False)
        B = self.generate_multioperators("B", n_vars=m, m_vars=n, hermitian=True, commutative=False)
        phi = self.generate_multioperators("phi", n_vars=n, m_vars=T+1, hermitian=True, commutative=False)
        w = self.generate_multioperators("w", n_vars=n, m_vars=T, hermitian=True, commutative=False)
        v = self.generate_multioperators("v", n_vars=m, m_vars=T, hermitian=True, commutative=False)
        # f = self.generate_multioperators("f", n_vars=m, m_vars=T, hermitian=True, commutative=False)


        # Objective
        #obj = sum((Y[i]-f[i])**2 for i in range(T)) + 0.0005*sum(p[i]**2 for i in range(T)) + 0.0001*sum(q[i]**2 for i in range(T))
        obj = sum((Y[mm][t]-f[mm][t])*2 for mm in range(m) for t in range(T))
        # + 0.0001*sum(w[nn][t]*2 for t in range(T) for nn in range(n))
        # + 0.0005*sum(v[mm][t]*2 for t in range(T) for mm in range(m))

        # Constraints
        ine1 = [phi[nn][t+1] - A[nn][nn]*phi[nn][t] - w[nn][t] for nn in range(n) for t in range(T)]
        ine2 = [-phi[nn][t+1] + A[nn][nn]*phi[nn][t] + w[nn][t] for nn in range(n) for t in range(T)]
        ine3 = [Y[mm][t] - B[mm][nn]*phi[nn][t+1] - v[mm][t] for nn in range(n) for t in range(T) for mm in range(m)]
        ine4 = [-Y[mm][t] + B[mm][nn]*phi[nn][t+1] + v[mm][t] for nn in range(n) for t in range(T) for mm in range(m)]
        ine5 = [w[nn][t]-v[mm][t] for nn in range(n) for t in range(T) for mm in range(m)]
        ine6 = [w[nn][t]+v[mm][t] for nn in range(n) for t in range(T) for mm in range(m)]
        ines = ine1+ine2+ine3+ine4+ine5+ine6

        # Solve the NCPO
        AA = Operator(np.asarray(A).reshape(-1))
        BB = Operator(np.asarray(B).reshape(-1))
        # f = Operator(np.asarray(f).reshape(-1))
        w = Operator(np.asarray(w).reshape(-1))
        v = Operator(np.asarray(v).reshape(-1))
        phi = Operator(np.asarray(phi).reshape(-1))
        #print([Operator(AA),Operator(BB),Operator(f),Operator(p),Operator(phi),Operator(q)])

        sdp = SdpRelaxation(variables = flatten([AA,BB,w,phi,v]),verbose = 1)
        sdp.get_relaxation(level, objective=obj, inequalities=ines)
        sdp.solve(solver='mosek')

        sdp.write_to_file("solutions.csv")
        sdp.write_to_file('example.dat-s')
        sdp.find_solution_ranks()
        #sdp.solve(solver='sdpa', solverparameters={"executable":"sdpa_gmp","executable": "C:/Users/zhouq/Documents/sdpa7-windows/sdpa.exe"})
        print(sdp.primal, sdp.dual,sdp.status)
        return sdp.primal
'''    


class NCPOLR(object):
    """Estimator based on NCPOP Regressor

    References
    ----------
    Quan Zhou https://github.com/Quan-Zhou/Proper-Learning-of-LDS/blob/master/ncpop/functions.py
    
    Examples
    --------
    >>> import numpy as np
    >>> Y=[1,2,3]
    >>> X=[1,2,3]
    >>> ncpolr = NCPOLR()
    >>> y_pred = ncpolr.estimate(X, Y)
    >>> print(y_pred)
    """
    
    def __init__(self, **kwargs):
        super(NCPOLR, self).__init__()

    def estimate(self, X, Y):
        """Fit Estimator based on NCPOP Regressor model and predict y or produce residuals.
        The module converts a noncommutative optimization problem provided in SymPy
        format to an SDPA semidefinite programming problem.

        Parameters
        ----------
        X : array
            Variable seen as cause
        Y: array
            Variable seen as effect

        Returns
        -------
        y_predict: array
            regression predict values of y or residuals
        """
        
        T = len(Y)
        level = 1
    
        # Decision Variables
        # f=G*x+n以前是最小化n**2，现在就直接最小化p
        # G是系数
        G = generate_operators("G", n_vars=1, hermitian=True, commutative=False)[0]
        # f是y的估计值
        f = generate_operators("f", n_vars=T, hermitian=True, commutative=False)
        # n是残差
        n = generate_operators("m", n_vars=T, hermitian=True, commutative=False)
        # p是n的绝对值
        p = generate_operators("p", n_vars=T, hermitian=True, commutative=False)

        # Objective
        obj = sum((Y[i]-f[i])**2 for i in range(T)) + 0.5*sum(p[i] for i in range(T))
        
        # Constraints
        ine1 = [f[i] - G*X[i] - n[i] for i in range(T)]
        ine2 = [-f[i] + G*X[i] + n[i] for i in range(T)]
        # fp和n的关系通过加新的限制条件p>n 和p>-n来实现
        ine3 = [p[i]-n[i] for i in range(T)]
        ine4 = [p[i]+n[i] for i in range(T)]
        ines = ine1+ine2+ine3+ine4

        # Solve the NCPO
        sdp = SdpRelaxation(variables = flatten([G,f,n,p]),verbose = 1)
        sdp.get_relaxation(level, objective=obj, inequalities=ines)
        sdp.solve(solver='mosek')
        #sdp.solve(solver='sdpa', solverparameters={"executable":"sdpa_gmp","executable": "C:/Users/zhouq/Documents/sdpa7-windows/sdpa.exe"})
        print(sdp.primal, sdp.dual, sdp.status)

        if(sdp.status != 'infeasible'):
            print('ok.')
            est_noise = []
            for i in range(T):
                est_noise.append(sdp[n[i]])
            print(est_noise)
            return est_noise
        else:
            print('Cannot find feasible solution.')
            return



class ANM_NCPO(BaseLearner):
    """
    Nonlinear causal discovery with additive noise models

    Use Estimator based on NCPOP Regressor and independent Gaussian noise,
    For the independence test, we implemented the HSIC with a Gaussian kernel,
    where we used the gamma distribution as an approximation for the
    distribution of the HSIC under the null hypothesis of independence
    in order to calculate the p-value of the test result.
    
    References
    ----------
    Hoyer, Patrik O and Janzing, Dominik and Mooij, Joris M and Peters,
    Jonas and Schölkopf, Bernhard,
    "Nonlinear causal discovery with additive noise models", NIPS 2009

    Parameters
    ----------
    alpha : float, default 0.05
        significance level be used to compute threshold

    Attributes
    ----------
    causal_matrix : array like shape of (n_features, n_features)
        Learned causal structure matrix.
    
    Examples
    --------
    >>> # from castle.algorithms.ncpol._ncpol import NCPOLR,ANM_NCPOP
    >>> from castle.common import GraphDAG
    >>> from castle.metrics import MetricsDAG
    >>> import numpy as np

    >>> rawdata = np.load('dataset/linear_gauss_6nodes_15edges.npz', allow_pickle=True)
    >>> data = rawdata['x'][:10]
    >>> true_dag = rawdata['y'][:10]
    >>> #np.asarray(rawdata['y'][:10])
    >>> anmNCPO = ANM_NCPO(alpha=0.05)
    >>> anmNCPO.learn(data=data)

    >>> # plot predict_dag and true_dag
    >>> GraphDAG(anmNCPO.causal_matrix, true_dag, show=False, save_name='result')
    >>> met = MetricsDAG(anmNCPO.causal_matrix, true_dag)
    >>> print(met.metrics)
    """

    def __init__(self, columns=None, alpha=0.05):
        super(ANM_NCPO, self).__init__()
        self.alpha = alpha

    def learn(self, data, columns=None,regressor=NCPOLR(),test_method=hsic_test, **kwargs):
        """Set up and run the ANM_NCPO algorithm.
        
        Parameters
        ----------
        data: numpy.ndarray or Tensor
            Training data.
        columns : Index or array-like
            Column labels to use for resulting tensor. Will default to
            RangeIndex (0, 1, 2, ..., n) if no column labels are provided.
        regressor: Class
            Nonlinear regression estimator, if not provided, it is NCPOLR.
            If user defined, must implement `estimate` method. such as :
                `regressor.estimate(x, y)`
        test_method: callable, default test_method
            independence test method, if not provided, it is HSIC.
            If user defined, must accept three arguments--x, y and keyword
            argument--alpha. such as :
                `test_method(x, y, alpha=0.05)`
        """

        self.regressor = regressor
        
        # create learning model and ground truth model
        data = Tensor(data, columns=columns)

        node_num = data.shape[0]
        self.causal_matrix = Tensor(np.zeros((node_num, node_num)))

        for i, j in combinations(range(node_num), 2):
            x = data[i, :, :]
            y = data[j, :, :]         
            xx = x.reshape((-1, 1))
            yy = y.reshape((-1, 1))

            flag = test_method(xx, yy, alpha=self.alpha)
            if flag == 1:
                continue
            # test x-->y
            flag = self.anmNCPO_estimate(x, y, regressor = regressor, test_method=test_method)
            if flag:
                self.causal_matrix[i, j] = 1
            # test y-->x
            flag = self.anmNCPO_estimate(y, x, regressor = regressor, test_method=test_method)
            if flag:
                self.causal_matrix[j, i] = 1

    def anmNCPO_estimate(self, x, y, regressor=NCPOLR(), test_method=hsic_test):
        """Compute the fitness score of the ANM_NCPOP Regression model in the x->y direction.

        Parameters
        ----------
        x: array
            Variable seen as cause
        y: array
            Variable seen as effect
        regressor: Class
            Nonlinear regression estimator, if not provided, it is NCPOP.
            If user defined, must implement `estimate` method. such as :
                `regressor.estimate(x, y)`
        test_method: callable, default test_method
            independence test method, if not provided, it is HSIC.
            If user defined, must accept three arguments--x, y and keyword
            argument--alpha. such as :
                `test_method(x, y, alpha=0.05)`
        Returns
        -------
        out: int, 0 or 1
            If 1, residuals n is independent of x, then accept x --> y
            If 0, residuals n is not independent of x, then reject x --> y
            
        Examples
        --------
        >>> import numpy as np
        >>> from castle.algorithms.ncpol._ncpol import ANM_NCPOP
        >>> rawdata = np.load('dataset/linear_gauss_6nodes_15edges.npz', allow_pickle=True)
        >>> data = rawdata['x'][:20]
        >>> true_dag = rawdata['y'][:20]
        >>> data = pd.DataFrame(data)
        >>> Y=np.asarray(data[0])
        >>> X=np.asarray(data[1])
        >>> anmNCPO = ANM_NCPO(alpha=0.05)
        """

        x = scale(x).reshape(-1)
        y = scale(y).reshape(-1)
        
        x_res = regressor.estimate(x, y)
        flag = test_method(np.asarray(x_res).reshape((-1, 1)), np.asarray(x).reshape((-1, 1)), alpha=self.alpha)
        print(flag)
        
        return flag


# Test2

# In[22]:


# from castle.algorithms.ncpol._ncpol import NCPOLR,ANM_NCPOP
from castle.common import GraphDAG
from castle.metrics import MetricsDAG
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

def read_graph(graph_file):
    with open(graph_file, "r") as file:
        lines = file.readlines()
    groundtruth = [tuple(line.strip().split()) for line in lines]
    return set(groundtruth)

def plot_graph(labels, graph_file, output_file):
    graph = read_graph(graph_file)
    adjacency = np.zeros((16, 16))

    for u, v in graph:
        uind = labels.index(u) # strip the _lag
        vind = labels.index(v)
        adjacency[uind, vind] = 1

    plt.figure()
    plt.imshow(adjacency, cmap='binary', interpolation='nearest', vmin=-1, vmax=1)
    plt.xticks(ticks=np.arange(len(adjacency)), labels=labels, rotation=80)
    plt.yticks(ticks=np.arange(len(adjacency)), labels=labels)
    plt.subplots_adjust(bottom=0.3)
    plt.savefig(output_file)

                
                
Test_data = np.load('Krebs_Cycle_16Nodes_43Edges_TS.npz', allow_pickle=True)
Raw_data = Test_data['x']
truedag = Test_data['y']
nodes = ["FUMARATE", "GTP", "H2O", "CIS-ACONITATE", "MALATE", "OXALOACETATE", "FAD", "SUCCINYL-COA", "NAD",
        "A-K-GLUTARATE", "GDP", "NADH", "CITRATE", "SUCCINATE", "ISOCITRATE", "ACETY-COA"]
Datasize = range(3,4,1)
Timeset = range(3,4,1)
f1_anm_ncpop = []
df = pd.DataFrame(columns=['Datasize','Timesets', 'Duration', 'fdr', 'tpr', 'fpr', 'shd', 'nnz', 'precision', 'recall', 'F1', 'gscore'])
for i in Datasize:
    for j in Timeset:
        data = Raw_data[:, :i, :j]
        t_start = time.time()
        # Test ANM-NCPOP
        anmNCPO = ANM_NCPO(alpha=0.05)
        anmNCPO.learn(data = data)
        # Save estimate causal_matrix
        save_name = 'result_'+ str(i) + 'Datasize_'+str(j) +'Timesets'
        Causal_Matrix = anmNCPO.causal_matrix
        pd.DataFrame(Causal_Matrix).to_csv(save_name+'_Causal_Matrix.csv',index=False)

        # Plot true_dag
        output = []
        with open(save_name+'_trueDAG_output.txt', 'w') as f:
            for i in range(len(truedag)):
                for j in range(len(truedag)):
                    if truedag[i][j] != 0:
                        f.write(f"{nodes[i]} {nodes[j]}\n")
        plot_graph(nodes, save_name+"_trueDAG_output.txt", save_name +"_TrueDAG.pdf")
        
        #  Plot predict_dag
        output = []
        with open(save_name+'_series_graph_output.txt', 'w') as f:
            for i in range(len(Causal_Matrix)):
                for j in range(len(Causal_Matrix)):
                    if Causal_Matrix[i][j] != 0:
                        f.write(f"{nodes[i]} {nodes[j]}\n")
        plot_graph(nodes, save_name+"_series_graph_output.txt", save_name+"_Prediction.pdf")
        # GraphDAG(anmNCPO.causal_matrix, truedag, show=False, save_name = save_name+'.png')

        # Save met.metrics
        met = MetricsDAG(Causal_Matrix, truedag)
        # if math.isnan(float(met.metrics['F1'])):
        #     f1_anm_ncpop = 0.2
        # else:
        #     f1_anm_ncpop = met.metrics['F1']
        dict1 = {'Datasize':i, 'Timesets':j, 'Duration':time.time()-t_start, 'F1_Score': f1_anm_ncpop}
        dict2 = met.metrics
        dict = {**dict1, **dict2}
        df = pd.concat([df, pd.DataFrame([dict])])
        df.to_csv('Sub_scores_Krebs_cycle.csv', index=False)
        print('ANM-NCPOP INFO: Krebs cycle is done!'+'F1 Score is '+ str(met.metrics['F1'])+'.')
        print('ANM-NCPOP INFO: Time Duration is '+ str(time.time()-t_start))
df.to_csv('Scores_Krebs_cycle.csv', index=False)


                


# In[ ]:


df_F1 = df['DataSize', 'Timesets', 'Duration', 'F1_Score']
df_F1.to_csv('Result_Krebs_cycle/F1_Krebs_cycle.csv',index=False)


# In[12]:


print(truedag)


# In[ ]:




