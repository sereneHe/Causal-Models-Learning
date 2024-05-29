# from Data_Standardization import*
from pickle import TRUE
import os
import re
import numpy as np
import pandas as pd
import tarfile
from itertools import combinations

class Real_Data_Standardization(object):
    '''
    A class for preparing data to simulate random (causal) DAG.

    Parameters
    ------------------------------------------------------------------------------------------------
    File_PATH: str
               Read file path
    File_NAME: str
               Read data name

    Returns
    ------------------------------------------------------------------------------------------------
    Raw_data: npz
            x：[d, n, T] sample time series 
            y: true_dag
    File_PATH_Datasets: 
            Route of saving test data

    Examples
    -------------------------------------------------------------------------------------------------
    >>> File_PATH = "Test/Datasets/Real_data/Telephone/"
    >>> file_name = 'Telephone'

    >>> File_PATH = "Test/Datasets/Real_data/Microwave/"
    >>> file_name = '25V_474N_Microwave'

    >>> Krebs_Cycle
    >>> File_PATH = "Test/Datasets/Synthetic_data/Kreb_Cycles/"
    >>> file_name = 'Krebs_Cycle'

    >>> dt = Real_Data_Standardization(File_PATH, file_name)
    >>> dt.Produce_Rawdata()
    '''

    def __init__(self, File_PATH, Data_NAME):
        self.File_PATH = File_PATH
        self.Data_NAME = Data_NAME

        filename = self.Data_NAME.capitalize()
        File_PATH_Base = self.File_PATH +'Results_'+ filename +'/'
        self.File_PATH_Datasets = File_PATH_Base + 'Datasets_'+ filename +'/'
        if not os.path.exists(self.File_PATH_Datasets):
            os.makedirs(self.File_PATH_Datasets)
        print('ANM-NCPOP INFO: Created Datasets_'+ method.capitalize()+ ' File!')

    def Produce_Rawdata(self):

        def readable_File(FilePATH):
            read_Dir=os.listdir(FilePATH)
            count = 0
            readable_F = []
            for f in read_Dir:
                file = os.path.join(FilePATH, f)
                if os.path.isdir(file):
                    count = count+1
                else:
                    readable_F.append(f)
            return count,readable_F

        self.Read_File = readable_File(self.File_PATH)[1]
        self.TS_path = self.File_PATH + self.Data_NAME + '_TS/'

        # Check empty files under riute
        if len(self.File_PATH) == 0:
            print('ANM-NCPOP INFO: No Data Under the Current Route!')
        else:
            File_NAME = []
            File_TYPE = []

            # Delete files and list readable Files
            for i in self.Read_File:
                File_NAME.append(re.split("\.", i)[0])
                File_TYPE.append(re.split("\.", i)[1])

            ###################################### Deal with Two Dimensions Causality Data ###################################
            '''if self.Data_NAME+'.npz' in self.Read_File:
                Tests_data = np.load(self.File_PATH + self.Data_NAME+'.npz', allow_pickle=True)
                Raw_data = Tests_data['x']
                true_dag = Tests_data['y']
                print('ANM-NCPOP INFO: Check for '+self.Data_NAME +'.npz'+ '!')'''

            if self.Data_NAME+'.tar.gz' in self.Read_File:
                # open file
                file = tarfile.open(self.File_PATH + self.Data_NAME + '.tar.gz')
                file_names = file.getnames()
                # extract files
                file.extractall(self.File_PATH)
                file.close()
                Raw_data = np.load(self.File_PATH+file_names[2])
                true_dag = pd.read_csv(self.File_PATH+file_names[3])

                # save numpy to npz file
                nn = true_dag.index
                ne = np.count_nonzero(true_dag)
                data_name = self.Data_NAME.capitalize()+'_'+str(nn)+'Nodes_'+str(ne)+'Edges_TS'
                np.savez(self.File_PATH_Datasets +data_name+'.npz', x=Raw_data , y=true_dag)
                print('ANM-NCPOP INFO: '+ data_name + ' IS DONE!')

            elif self.Data_NAME+'.csv' in self.Read_File:
                Raw_data = pd.read_csv(self.File_PATH+ self.Data_NAME+'.csv', header=0, index_col=False)
                true_dag = pd.read_csv(self.File_PATH+'true_graph.csv', header=0, index_col=0)
                
                # save numpy to npz file
                nn = true_dag.index
                ne = np.count_nonzero(true_dag)
                data_name = self.Data_NAME.capitalize()+'_'+str(nn)+'Nodes_'+str(ne)+'Edges_TS'
                np.savez(self.File_PATH_Datasets +data_name+'.npz', x=Raw_data , y=true_dag)
                print('ANM-NCPOP INFO: '+ data_name + ' IS DONE!')



            ################################ Deal with Multi-dimensions Causality Data ###################################
            elif os.path.exists(self.TS_path):
                read_Dir_TS=os.listdir(self.TS_path)
                true_graph = np.load(self.File_PATH+'true_graph.npz')

                labels = ["FUMARATE", "GTP", "H2O", "CIS-ACONITATE", "MALATE",
                "OXALOACETATE", "FAD", "SUCCINYL-COA", "NAD",
                          "A-K-GLUTARATE", "GDP", "NADH", "CITRATE", "SUCCINATE",
                "ISOCITRATE", "ACETY-COA"]
                true_dag = pd.DataFrame(true_graph['arr_0'],  index=labels, columns=labels)
                # print(true_dag)
                lds = pd.read_csv(self.TS_path+ read_Dir_TS[0], delimiter='\t', index_col=0, header=None)
                feature_name = np.array(lds.index)
                feature_num = len(feature_name)
                sample_num = len(read_Dir_TS)
                T_num = lds.shape[1]
                # if labels == feature_name:
                Raw_data = np.zeros((feature_num, sample_num, T_num))
                for ns in range(sample_num):
                    X = pd.read_csv(self.TS_path+ read_Dir_TS[ns], delimiter='\t', index_col=0, header=None)
                    X_trans = np.transpose(X)
                    for fn in range(feature_num):
                        Raw_data[fn, ns, :] = list(X_trans[feature_name[fn]])

                # save numpy to npz file
                nn = feature_num
                ne = np.count_nonzero(true_dag)
                data_name = self.Data_NAME.capitalize()+'_'+str(nn)+'Nodes_'+str(ne)+'Edges_TS'
                np.savez(self.File_PATH_Datasets +data_name+'.npz', x=Raw_data , y=true_dag)
                print('ANM-NCPOP INFO: '+ data_name + ' IS DONE!')

            else:
                print('INFO: Wrong DataType!')
