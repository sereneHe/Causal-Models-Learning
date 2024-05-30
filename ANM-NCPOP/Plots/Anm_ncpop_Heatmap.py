import os
import re
import csv
import math
import time
import Anm_ncpop_test
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from BuiltinDataSet import DAG
import matplotlib.ticker as ticker
from castle.common import GraphDAG
from castle.metrics import MetricsDAG

class Ancpop_Simulation(object):
    '''
    A class for simulating random (causal) DAG based on synthetic datasets, where any DAG generator
    self.method would return the weighed/binary adjacency matrix of a DAG.

    Parameters
    -----------------------------------------------------------------------------------------------
    method: str, (linear or nonlinear), default='linear'
        Distribution for standard training time series.
    File_PATH: str
        Save file path
    sem_type: str
        gauss, exp, gumbel, uniform, logistic (linear);
        mlp, mim, gp, gp-add, quadratic (nonlinear).
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
    Summary table: pd.dataframe
            col_names = ['SEM','Noise','Nodes','Edges','DataSize','Timesets','F1_Score', 'Duration']

    Examples
    -------------------------------------------------------------------------------------------------
    >>> 
    >>> File_PATH = 'Test/Results'
    >>> Nodes = range(6, 15, 3)
    >>> Edges = range(10, 20, 5)
    >>> datasize = range(5, 40, 5)
    >>> Timesize = range(3, 6, 1)
    >>> st = Ancpop_Simulation(File_PATH, method, sem_type, Nodes, Edges, datasize, Timesize)
    >>> st.Ancpop_simulation_Test()
    '''

    def __init__(self, File_PATH_Summary_Datails, File_PATH_Heatmap, Datasize, Timesize):
        self.File_PATH_Summary_Datails = File_PATH_Summary_Datails
        self.File_PATH_Heatmap = File_PATH_Heatmap
        self.Datasize =  Datasize
        self.Datasize_num = len(self.datasize)
        self.Timesize = Timesize
        self.Timesize_num = len(self.Timesize)
        self.filename = re.split("_", re.split("/", self.File_PATH_Heatmap)[-1])[1]
        
    @staticmethod
    def extract_and_join(filename):
        parts = filename.split('_')
        name_without_extension = parts[:-1]
        parts_to_join = name_without_extension[:-3]
        # join with '_'
        result = '_'.join(parts_to_join)
        return result

    @staticmethod 
    def Ancpop_simulation_Test(self):
        ################################################  Create Ground Tier Folders #############################################
        self.File_PATH_Base = self.File_PATH +'Result_'+ self.filename +'/'
        if not os.path.exists(self.File_PATH_Base):
            os.makedirs(self.File_PATH_Base)
        print('ANM-NCPOP INFO: Created Basement'+ ' File!')

        ################################################  Create First Tier Folders #############################################
        self.File_PATH_Heatmap = self.File_PATH_Base + 'Heatmap_'+ self.filename + '/'
        if not os.path.exists(self.File_PATH_Heatmap):
            os.makedirs(self.File_PATH_Heatmap )
        print('ANM-NCPOP INFO: Created Heatmap_GetReady'+ ' File!')

        ################################################  Create Second Tier Folders under Summary ###############################
        self.File_PATH_Summary_Tables = self.File_PATH_Heatmap +'Summary_'+self.filename +'/'
        if not os.path.exists(self.File_PATH_Summary_Tables):
            os.makedirs(self.File_PATH_Summary_Tables)
        print('ANM-NCPOP INFO: Created Summary_Tables'+ ' File!')

        self.Summary_Results(self)

    @staticmethod
    def Summary_Results(self):
        ###############################################  Summarize results ###############################
        f1_anm_ncpop = pd.DataFrame()
        for summary_f in range(len(tqdm)):
            File_PATH = os.path.join(self.File_PATH_Summary_Datails,tqdm[summary_f])
            df = pd.read_csv(File_PATH)
            # file_name = 'F1_LinearSEM_GaussNoise_6Nodes_15Edges_TS.npz'
            s = re.split("\.", tqdm[summary_f])[0]
            # LinearSEM
            ss_SEM = re.split("_", s)[1]
            sem = re.split("SEM", ss_Noise)[0]
            df['SEM'] = sem
            # GaussNoise
            ss_Noise = re.split("_", s)[2]
            noise = re.split("Noise", ss_Noise)[0]
            df['Noise'] = noise
            # 6Nodes
            ss_Nodes = re.split("_", s)[3]
            nn = re.split("Nodes", ss_Nodes)[0]
            df['Nodes'] = nn
            # 15Edges
            ss_Edges = re.split("_", s)[4]
            ne = re.split("Edges", ss_Edges)[0]
            df['Edges'] = ne

            df_combined = []
            df_combined = pd.concat([df_combined, df], ignore_index=True, axis=1)

            f1_anm_ncpop_nan = df.loc[:,'F1_Score']
            if len([f1_anm_ncpop_nan == 0]) == len(f1_anm_ncpop_nan):
              f1_anm_ncpop = 0.2
            else:
              f1_anm_ncpop = round(np.nanmean(f1_anm_ncpop_nan), 3)

            f1_result = pd.DataFrame(df_combined, columns=['SEM','Noise','Nodes','Edges','DataSize','Timesets','F1_Score', 'Duration'])
        f1_result.to_csv(self.Table_PATH_Summary+'summary.csv',index=False)
        return f1_result

