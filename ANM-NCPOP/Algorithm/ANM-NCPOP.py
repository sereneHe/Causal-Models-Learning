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


class Anm_ncpop_test(object):
    '''
    A class for simulating (causal) DAG, where the true DAG is a weighed/binary adjacency matrix based on ground truth.

    Parameters
    ------------------------------------------------------------------------------------------------
    File_PATH: str
        Save file path
    File_NAME: str
        Read data name
    Data_st: int
        Start number of samples for standard training time series
    Data_ed: int
        stop number of samples for standard training time series
    Datastep: int
        step number of samples for standard training time series
    Time_st: int
        Start number of timesets for standard training time series
    Time_ed: int
        stop number of timesets for standard training time series
    Timestep: int
        step number of timesets for standard training time series
        
    Returns
    ------------------------------------------------------------------------------------------------
    Metrics DAG: np.matrix
            estimate DAG matrix
    Summary table: pd.dataframe
           [DataSize, Timesets, F1_Score, Duration]

    Examples
    -------------------------------------------------------------------------------------------------
    >>> Data_st = 5
    >>> Data_ed = 40
    >>> Datastep = 5
    >>> Time_st = 3
    >>> Time_ed = 6
    >>> Timestep = 1
    >>> File_PATH = 'Test/Examples/Test_data/'
    >>> file_name = 'linearGauss_6_15_TS.npz'
    >>> # file_name = 'Krebs_Cycle_16_43_TS.npz'
    >>> rt = Anm_ncpop_test(File_PATH, file_name, Data_st, Data_ed, Datastep, Time_st, Time_ed, Timestep)
    >>> rt.Ancpop()

    '''
    def __init__(self, File_PATH,File_NAME, Data_st, Data_ed, Datastep, Time_st, Time_ed, Timestep):
        self.File_PATH = File_PATH
        self.File_NAME = File_NAME
        self.Data_st = Data_st
        self.Data_ed = Data_ed
        self.Datastep = Datastep
        self.Time_st = Time_st
        self.Time_ed = Time_ed
        self.Timestep = Timestep
        self.datasize = range(self.Data_st, self.Data_ed, self.Datastep)
        self.datasize_num = len(self.datasize)
        self.Timesize = range(self.Time_st, self.Time_ed, self.Timestep)
        self.Timesize_num = len(self.Timesize)
        self.sname = re.split("\.", self.File_NAME)[0]

    def Ancpop(self):
        ################################################  Test Data #############################################
        self.File_PATH_Heatmaps = self.File_PATH + 'Results_'+self.sname+ '/'
        self.Table_PATH_Summary = self.File_PATH_Heatmaps + 'Summary_'+self.sname+'.csv'
        if not os.path.exists(self.Table_PATH_Summary):
            print('INFO: Testing '+ self.sname+'!')
            self.File_PATH_Base = self.File_PATH + 'Details_'+self.sname+ '/'
            self.Ancpop_estimate(self)
        else:
            print('INFO: Finished '+ self.sname+'Sampling!')

    @staticmethod
    def Ancpop_estimate(self):
        ############################################## Create Files ###################################
        if not os.path.exists(self.File_PATH_Base):
            os.makedirs(self.File_PATH_Base)
        self.File_PATH_MetricsDAG = self.File_PATH_Base +'MetricsDAG/'
        if not os.path.exists(self.File_PATH_MetricsDAG):
            os.makedirs(self.File_PATH_MetricsDAG)
        if not os.path.exists(self.File_PATH_Heatmaps):
            os.makedirs(self.File_PATH_Heatmaps)

        duration_anm_ncpop = []
        f1_anm_ncpop = []
        df = pd.DataFrame(columns=['Datasize','Timesets', 'fdr', 'tpr', 'fpr', 'shd', 'nnz', 'precision', 'recall', 'F1', 'gscore'])
        for i in Datasize:
            for j in Timesets:
                data = Rawdata[:, :i, :j]
                t_start = time.time()
                # Test ANM-NCPOP
                anmNCPO = ANM_NCPO(alpha=0.05)
                anmNCPO.learn(data = data)
                # Save estimate causal_matrix
                pd.DataFrame(anmNCPO.causal_matrix).to_csv(self.File_PATH_MetricsDAG + self.sname+'_' + str(i) + 'Datasize_'+str(j) +'Timesets.csv',index=False)

                # Plot predict_dag and true_dag
                GraphDAG(anmNCPO.causal_matrix, true_dag, show=False, save_name = self.File_PATH_MetricsDAG + self.sname+'_' + str(i) + 'Datasize_'+str(j) +'Timesets.png')

                # Save met.metrics
                met = MetricsDAG(anmNCPO.causal_matrix, true_dag)
                dict1 = {'Datasize':i, 'Timesets':j}
                dict2 = met.metrics
                dict = {**dict1, **dict2}
                df = pd.concat([df, pd.DataFrame([dict])])
                if math.isnan(float(met.metrics['F1'])):
                    f1_anm_ncpop.append(0.2)
                else:
                    f1_anm_ncpop.append(met.metrics['F1'])
                print(self.sname+'_' + str(i) + 'Datasize_'+str(j) +'Timesets is done!'+'F1 Score is'+ str(met.metrics['F1'])+'.')
                print('Time Duration is '+ str(time.time()-t_start))
                duration_anm_ncpop.append(time.time()-t_start)
        df.to_csv(self.Table_PATH_Summary, index=False)
        df_F1 = pd.DataFrame({"DataSize":self.datasize, "Timesets":self.timesets, 'F1_Score':f1_anm_ncpop, 'Duration':duration_anm_ncpop})
        df_F1.to_csv(self.File_PATH_Heatmaps + 'F1_'+self.sname+'.csv',index=False)
        return df_F1
