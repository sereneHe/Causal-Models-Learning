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
    >>> datasize = range(5, 40, 5)
    >>> Timesize = range(3, 6, 1)
    >>> File_PATH_Base = 'Test/Examples/Test_data/'
    >>> File_PATH_Summary_Datails = 'Test/Examples/Test_data/Summary'
    >>> Data_NAME = 'LinearSEM_GaussNoise_6Nodes_15Edges_TS.npz'
    >>> # Data_NAME = 'Krebs_Cycle_16Nodes_43Edges_TS.npz'
    >>> rt = Anm_ncpop_test(File_PATH_Base, Data_NAME, File_PATH_Summary_Datails, datasize, Timesize)
    >>> rt.Ancpop()

    '''
    def __init__(self, File_PATH_Datasets=‘read/data/PATH’, Data_NAME='dataname', File_PATH_Summary_Datails=‘SAVE/TO/PATH’, Datasize=range(5, 40, 5), Timesize= range(3, 6, 1)):
        self.File_PATH_Datasets = File_PATH_Datasets
        self.File_PATH_Summary_Datails = File_PATH_Summary_Datails
        self.Data_NAME =  Data_NAME
        self.Datasize =  Datasize
        self.Datasize_num = len(self.Datasize)
        self.Timesize = Timesize
        self.Timesize_num = len(self.Timesize)
        self.filename = self.extract_and_join(self.Data_NAME)
        self.sname = re.split("\.", self.Data_NAME)[0]
        
        
    def extract_and_join(filename):
        parts = filename.split('_')
        name_without_extension = parts[:-1]
        parts_to_join = name_without_extension[:-3]
        
        # join with '_'
        result = '_'.join(parts_to_join)
        return result

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

        ################################################  Create Second Tier Folders #############################################
        self.File_PATH_Summary_Datails = self.File_PATH_Summary + 'Summary_Datails_'+self.filename +'/'
        self.File_PATH_MetricsDAG = self.File_PATH_Summary +'MetricsDAG_'+self.filename +'/'
        if not os.path.exists(self.File_PATH_Summary_Datails):
            os.makedirs(self.File_PATH_Summary_Datails)
        print('ANM-NCPOP INFO: Created Summary_Datails'+ ' File!')
        if not os.path.exists(self.File_PATH_MetricsDAG):
            os.makedirs(self.File_PATH_MetricsDAG)
        print('ANM-NCPOP INFO: Created MetricsDAG'+ ' File!')

        # Read data
        Rawdata = np.load(os.path.join(File_PATH_Datasets, self.Data_NAME))
        Raw_data = Rawdata['x']
        true_dag = Rawdata['y']
        
        self.Ancpop_estimate(self)
        print('ANM-NCPOP INFO: Finished '+ self.filename+'Analyzing!')

    @staticmethod
    def Ancpop_estimate(self):
        duration_anm_ncpop = []
        f1_anm_ncpop = []
        df = pd.DataFrame(columns=['Datasize','Timesets', 'Duration', 'fdr', 'tpr', 'fpr', 'shd', 'nnz', 'precision', 'recall', 'F1', 'gscore'])
        for i in self.Datasize:
            for j in self.Timesets:
                data = Raw_data[:, :i, :j]
                t_start = time.time()
                # Test ANM-NCPOP
                anmNCPO = ANM_NCPO(alpha=0.05)
                anmNCPO.learn(data = data)
                # Save estimate causal_matrix
                # save_result_name = LinearSEM_GaussNoise_6Nodes_15Edges_TS_15Datasize_5Timesets
                save_result_name = self.sname+'_' + str(i) + 'Datasize_'+str(j) +'Timesets'
                pd.DataFrame(anmNCPO.causal_matrix).to_csv(self.File_PATH_MetricsDAG + save_result_name+'.csv',index=False)

                # Plot predict_dag and true_dag
                GraphDAG(anmNCPO.causal_matrix, true_dag, show=False, save_name = self.File_PATH_MetricsDAG + save_result_name+'.png')

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
                print('ANM-NCPOP INFO: ' +save_result_name +' is done!'+'F1 Score is'+ str(met.metrics['F1'])+'.')
                print('ANM-NCPOP INFO: Time Duration is '+ str(time.time()-t_start))
                duration_anm_ncpop.append(time.time()-t_start)
        df.to_csv(self.File_PATH_MetricsDAG + 'Scores_'+self.sname+'.csv', index=False)
        df_F1 = pd.DataFrame({"DataSize":self.Datasize, "Timesets":self.Timesets, 'F1_Score':f1_anm_ncpop, 'Duration':duration_anm_ncpop})
        df_F1.to_csv(File_PATH_Summary_Datails + 'F1_'+self.sname+'.csv',index=False)
        return df_F1

