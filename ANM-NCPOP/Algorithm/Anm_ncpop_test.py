import os
import re
import csv
import math
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from castle.common import GraphDAG
from castle.metrics import MetricsDAG
from sklearn.preprocessing import scale
from itertools import combinations
from castle.common import BaseLearner, Tensor
from castle.common.independence_tests import hsic_test
from ncpol2sdpa import*



class Anm_ncpop_test(object):
    '''
    A class for simulating (causal) DAG, where the true DAG is a weighed/binary adjacency matrix based on ground truth.

    Parameters
    ------------------------------------------------------------------------------------------------
    File_PATH: str
            Read file path
    File_NAME: str
            Read data name
    File_PATH_Summary_Datails: str
            Save file path
    datasize: series
    
    Timesize: series
        
    Returns
    ------------------------------------------------------------------------------------------------
    Metrics DAG: np.matrix
            heatmap between estimate DAG matrix and true DAG
    Casaul Metrics: np.matrix
            estimate DAG matrix
    Summary scores table: pd.dataframe
           col_names = ['Datasize','Timesets', 'Duration', 'fdr', 'tpr', 'fpr', 'shd', 'nnz', 'precision', 'recall', 'F1', 'gscore'])
    Summary table: pd.dataframe
           col_names = ['DataSize', 'Timesets', 'F1_Score', 'Duration']

    Examples
    -------------------------------------------------------------------------------------------------
    >>> # filename = LinearSEM_GaussNoise
    >>> # data_name = LinearSEM_GaussNoise_6Nodes_15Edges_TS
    >>> # save_name = LinearSEM_GaussNoise_6Nodes_15Edges_TS_15Datasize_5Timesets

    >>> datasize = range(5, 40, 5)
    >>> Timesize = range(3, 6, 1)
    >>> File_PATH_Base = 'Test/Examples/Test_data/'
    >>> File_PATH_Summary_Datails = 'Test/Examples/Test_data/Summary'
    >>> File_PATH = File_PATH
    >>> Data_NAME = 'LinearSEM_GaussNoise_6Nodes_15Edges_TS.npz'
    >>> # Data_NAME = 'Krebs_Cycle_16Nodes_43Edges_TS.npz'
    >>> rt = Anm_ncpop_test(File_PATH_Base, Data_NAME, File_PATH_Summary_Datails, datasize, Timesize)
    >>> rt.Ancpop()

    '''
    def __init__(self, File_PATH = None, Datasize=range(5, 40, 5), Timeset= range(3, 6, 1)):
        self.File_PATH = File_PATH
        self.Datasize =  Datasize
        # self.Datasize_num = len(self.Datasize)
        self.Timeset = Timeset
        # self.Timesize_num = len(self.Timesize)
        self.filename = filename
        # re.split("_", re.split("/", self.File_PATH_Datasets)[-1])[0]
        

    def Ancpop(self):
        ################################################  Create Ground Tier Folders #############################################
        self.File_PATH_Base = self.File_PATH +'Result_'+ self.filename +'/'
        if not os.path.exists(self.File_PATH_Base):
            os.makedirs(self.File_PATH_Base)
        print('ANM-NCPOP INFO: Created Basement'+ ' File!')
        
        ################################################  Create First Tier Folders #############################################
        self.File_PATH_Summary = self.File_PATH_Base + 'Summary_'+ self.filename +'/'
        if not os.path.exists(self.File_PATH_Summary):
            os.makedirs(self.File_PATH_Summary)
        print('ANM-NCPOP INFO: Created Summary'+ ' File!')

        self.File_PATH_Datasets = self.File_PATH_Base + 'Datasets_'+ self.filename +'/'
        if not os.path.exists(self.File_PATH_Datasets):
            os.makedirs(self.File_PATH_Datasets)
            dt = Real_Data_Standardization(self.File_PATH, self.filename)
            dt.standardize_data()

        print('ANM-NCPOP INFO: Created Datasets' + ' File!')

        ################################################  Create Second Tier Folders #############################################
        self.File_PATH_Summary_Datails = self.File_PATH_Summary + 'Summary_Datails_'+self.filename +'/'
        self.File_PATH_MetricsDAG = self.File_PATH_Summary +'MetricsDAG_'+self.filename +'/'
        if not os.path.exists(self.File_PATH_Summary_Datails):
            os.makedirs(self.File_PATH_Summary_Datails)
        print('ANM-NCPOP INFO: Created Summary_Datails'+ ' File!')
        if not os.path.exists(self.File_PATH_MetricsDAG):
            os.makedirs(self.File_PATH_MetricsDAG)
        print('ANM-NCPOP INFO: Created MetricsDAG'+ ' File!')

        ################################################  Analyzing Data under Datasets ###############################
        tqdm=os.listdir(self.File_PATH_Summary_Datails)
        read_Dir=os.listdir(self.File_PATH_Datasets)
        while len(tqdm)!= len(read_Dir):
            for data_name in read_Dir:
                # print(file_f)
                filename = utils.saveName_transfer_to_filename(data_name)

                df_F1 = self.File_PATH_Summary_Datails + 'F1_'+ data_name +'.csv'
                if not os.path.exists(df_F1):
                    Rawdata = np.load(self.File_PATH_Datasets+data_name)
                    self.Ancpop_estimate(self, Rawdata, data_name)
                print('ANM-NCPOP INFO: Finished '+ data_name+'Analyzing!')
        print('ANM NCPOP INFO: Finished simulations!')

    @staticmethod
    def Ancpop_estimate(self, Rawdata, data_name):
        Raw_data = Rawdata['x']
        true_dag = Rawdata['y']
        duration_anm_ncpop = []
        f1_anm_ncpop = []
        df = pd.DataFrame(columns=['Datasize','Timesets', 'Duration', 'fdr', 'tpr', 'fpr', 'shd', 'nnz', 'precision', 'recall', 'F1', 'gscore'])
        for i in self.Datasize:
            for j in self.Timeset:
                data = Raw_data[:, :i, :j]
                t_start = time.time()
                # Test ANM-NCPOP
                anmNCPO = ANM_NCPO(alpha=0.05)
                anmNCPO.learn(data = data)
                # Save estimate causal_matrix
                save_name = data_name+'_' + str(i) + 'Datasize_'+str(j) +'Timesets'
                pd.DataFrame(anmNCPO.causal_matrix).to_csv(self.File_PATH_MetricsDAG + save_name+'.csv',index=False)

                # Plot predict_dag and true_dag
                GraphDAG(anmNCPO.causal_matrix, true_dag, show=False, save_name = self.File_PATH_MetricsDAG + save_name+'.png')

                # Save met.metrics
                met = MetricsDAG(anmNCPO.causal_matrix, true_dag)
                dict1 = {'Datasize':i, 'Timesets':j, 'Duration':time.time()-t_start}
                dict2 = met.metrics
                dict = {**dict1, **dict2}
                df = pd.concat([df, pd.DataFrame([dict])])
                if math.isnan(float(met.metrics['F1'])):
                    f1_anm_ncpop.append(0.2)
                else:
                    f1_anm_ncpop.append(met.metrics['F1'])
                print('ANM-NCPOP INFO: ' + save_name +' is done!'+'F1 Score is'+ str(met.metrics['F1'])+'.')
                print('ANM-NCPOP INFO: Time Duration is '+ str(time.time()-t_start))
                duration_anm_ncpop.append(time.time()-t_start)
        df.to_csv(self.File_PATH_MetricsDAG + 'Scores_'+data_name+'.csv', index=False)
        df_F1 = pd.DataFrame({"DataSize":self.Datasize, "Timesets":self.Timesets, 'F1_Score':f1_anm_ncpop, 'Duration':duration_anm_ncpop})
        df_F1.to_csv(self.File_PATH_Summary_Datails + 'F1_'+data_name+'.csv',index=False)
        return df_F1

class NCPOLR(object):
    """Estimator based on NCPOP Regressor

    References
    ----------
    Quan Zhou https://github.com/Quan-Zhou/Proper-Learning-of-LDS/blob/master/ncpop/functions.py

    Examples
    --------
    """

    def __init__(self, **kwargs):
        super(NCPOLR, self).__init__()


    def generate_operators(name, n_vars=1, hermitian=None, commutative=False):
        """Generates a number of commutative or noncommutative operators

        :param name: The prefix in the symbolic representation of the noncommuting
                    variables. This will be suffixed by a number from 0 to
                    n_vars-1 if n_vars > 1.
        :type name: str.
        :param n_vars: The number of variables.
        :type n_vars: int.
        :param hermitian: Optional parameter to request Hermitian variables .
        :type hermitian: bool.
        :param commutative: Optional parameter to request commutative variables.
                            Commutative variables are Hermitian by default.
        :type commutative: bool.

        :returns: list of :class:`sympy.physics.quantum.operator.Operator` or
                  :class:`sympy.physics.quantum.operator.HermitianOperator`
                  variables

        :Example:

        >>> generate_variables('y', 2, commutative=True)
        ￼[y0, y1]
        """

        variables = []
        for i in range(n_vars):
            if n_vars > 1:
                var_name = '%s%s' % (name, i)
            else:
                var_name = '%s' % name
            if hermitian is not None and hermitian:
                variables.append(HermitianOperator(var_name))
            else:
                variables.append(Operator(var_name))
            variables[-1].is_commutative = commutative
        return variables

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
        G = generate_operators("G", n_vars=1, hermitian=True, commutative=False)[0]
        f = generate_operators("f", n_vars=T, hermitian=True, commutative=False)
        n = generate_operators("m", n_vars=T, hermitian=True, commutative=False)
        p = generate_operators("p", n_vars=T, hermitian=True, commutative=False)

        # Objective
        obj = sum((Y[i]-f[i])**2 for i in range(T)) + 0.00005*sum(p[i] for i in range(T))

        # Constraints
        ine1 = [f[i] - G*X[i] - n[i] for i in range(T)]
        ine2 = [-f[i] + G*X[i] + n[i] for i in range(T)]
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

    def estimate2(self, X, Y):
        """Fit Estimator based on NCPOP Regressor model and predict y or produce residuals.
        The module converts a noncommutative optimization problem provided in SymPy
        format to an SDPA semidefinite programming problem.

        Parameters
        ----------
        Y: array
            Variable seen as effect

        Returns
        -------
        y_predict: array
            regression predict values of y or residuals
        """
        Y = np.transpose(Y)
        T = len(Y)-1
        level = 1


        # Decision Variables
        G = generate_operators("G", n_vars=1, hermitian=True, commutative=False)[0]
        Fdash = generate_operators("Fdash", n_vars=1, hermitian=True, commutative=False)[0]
        # m = generate_operators("m", n_vars=T+1, hermitian=True, commutative=False)
        q = generate_operators("q", n_vars=T, hermitian=True, commutative=False)
        p = generate_operators("p", n_vars=T, hermitian=True, commutative=False)
        f = generate_operators("f", n_vars=T, hermitian=True, commutative=False)

        # Objective
        obj = sum((Y[i]-f[i])**2 for i in range(T)) + 0.001*sum(p[i]**2 for i in range(T)) + 0.0005*sum(q[i]**2 for i in range(T))

        #c1*sum(p[i]**2 for i in range(T)) + c2*sum(q[i]**2 for i in range(T))

        # Constraints
        ine1 = [f[i] - Fdash*X[i+1] - p[i] for i in range(T)]
        ine2 = [-f[i] + Fdash*X[i+1] + p[i] for i in range(T)]
        ine3 = [X[i+1] - G*X[i] - q[i] for i in range(T)]
        ine4 = [-X[i+1] + G*X[i] + q[i] for i in range(T)]
        #ine5 = [(Y[i]-f[i])**2 for i in range(T)]
        ines = ine1+ine2+ine3+ine4 #+ine5

        # Solve the NCPO
        sdp = SdpRelaxation(variables = flatten([G,Fdash,f,p,q]),verbose = 1)
        sdp.get_relaxation(level, objective=obj, inequalities=ines)
        sdp.solve(solver='mosek')

        #sdp.solve(solver='sdpa', solverparameters={"executable":"sdpa_gmp","executable": "C:/Users/zhouq/Documents/sdpa7-windows/sdpa.exe"})
        print(sdp.primal, sdp.dual, sdp.status)

        if(sdp.status != 'infeasible'):
            print('ok.')
            est_noise = []
            for i in range(T):
                est_noise.append(sdp[p[i]])
            print(est_noise)
            return est_noise, X[1:]
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
    """

    def __init__(self, alpha=0.05):
        super(ANM_NCPO, self).__init__()
        self.alpha = alpha

    def learn(self, data, columns=None, regressor=NCPOLR(),test_method=hsic_test, **kwargs):
        """Set up and run the ANM_NCPOP algorithm.

        Parameters
        ----------
        data: numpy.ndarray or Tensor
            Training data.
        columns : Index or array-like
            Column labels to use for resulting tensor. Will default to
            RangeIndex (0, 1, 2, ..., n) if no column labels are provided.
        regressor: Class
            Nonlinear regression estimator, if not provided, it is NCPOLR.
            If user defined, must implement `estimate` self.method. such as :
                `regressor.estimate(x, y)`
        test_method: callable, default test_method
            independence test self.method, if not provided, it is HSIC.
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
            xx = x.reshape(-1,1)
            yy = y.reshape(-1,1)

            flag = test_method(xx, yy, alpha=self.alpha)
            if flag == 1:
                continue
            # test x-->y
            flag = self.ANMNCPO_fitness(x, y, regressor = regressor, test_method=test_method)
            if flag:
                self.causal_matrix[i, j] = 1
            # test y-->x
            flag = self.ANMNCPO_fitness(y, x, regressor = regressor, test_method=test_method)
            if flag:
                self.causal_matrix[j, i] = 1

    def ANMNCPO_fitness(self, x, y, regressor=NCPOLR(), test_method=hsic_test):
        """Compute the fitness score of the ANM_NCPOP Regression model in the x->y direction.

        Parameters
        ----------
        x: array
            Variable seen as cause
        y: array
            Variable seen as effect
        regressor: Class
            Nonlinear regression estimator, if not provided, it is NCPOP.
            If user defined, must implement `estimate` self.method. such as :
                `regressor.estimate(x, y)`
        test_method: callable, default test_method
            independence test self.method, if not provided, it is HSIC.
            If user defined, must accept three arguments--x, y and keyword
            argument--alpha. such as :
                `test_method(x, y, alpha=0.05)`
        Returns
        -------
        out: int, 0 or 1
            If 1, residuals n is independent of x, then accept x --> y
            If 0, residuals n is not independent of x, then reject x --> y

        """

        x = scale(x).reshape(-1)
        y = scale(y).reshape(-1)

        ncpop_res = regressor.estimate(x, y)
        print(x)
        print(y)

        flag = test_method(np.asarray(ncpop_res[0]).reshape((-1, 1)), np.asarray(ncpop_res[1]).reshape((-1, 1)), alpha=self.alpha)

        print(flag)

        return flag


