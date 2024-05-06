# from Data_Standardization import*
from pickle import TRUE
import os
import re
import numpy as np
import pandas as pd
import tarfile
from itertools import combinations

class ANMPOP_ReadData(object):
    '''
    A class for preparing data to simulate random (causal) DAG.

    Parameters
    ----------
    File_PATH: str
        Save file path
    File_NAME: str
        Read data name
    '''

    def __init__(self, File_PATH, Data_NAME):
        self.File_PATH = File_PATH
        self.Data_NAME = Data_NAME

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
            print('INFO: No Data Under the Current Route!')
        else:
            File_NAME = []
            File_TYPE = []

            # Delete files and list readable Files
            for i in self.Read_File:
                File_NAME.append(re.split("\.", i)[0])
                File_TYPE.append(re.split("\.", i)[1])

            ###################################### Deal with Two Dimensions Causality Data ###################################
            if self.Data_NAME+'.npz' in self.Read_File:
                Tests_data = np.load(self.File_PATH + self.Data_NAME+'.npz', allow_pickle=True)
                Raw_data = Tests_data['x']
                true_dag = Tests_data['y']
                print('INFO: Check for '+self.Data_NAME +'.npz'+ '!')

            elif self.Data_NAME+'.tar.gz' in self.Read_File:
                # open file
                file = tarfile.open(self.File_PATH + self.Data_NAME + '.tar.gz')

                # print file names
                file_names = file.getnames()
                print(file_names)

                # extract files
                file.extractall(self.File_PATH)

                # close file
                file.close()

                Raw_data = pd.read_csv(self.File_PATH+file_names[1])
                true_dag = np.load(self.File_PATH+file_names[2])
                
                # save numpy to npz file
                np.savez(self.File_PATH + self.Data_NAME+'.npz', x=Raw_data , y=true_dag)
                print('INFO: Check for '+self.Data_NAME +'.npz'+ '!')

            elif self.Data_NAME+'.csv' in self.Read_File:
                Raw_data = pd.read_csv(self.File_PATH+ self.Data_NAME+'.csv', header=0, index_col=0)
                true_dag = pd.read_csv(self.File_PATH+'true_graph.csv', header=0, index_col=0)

                # save numpy to npz file
                np.savez(self.File_PATH + self.Data_NAME+'.npz', x=Raw_data , y=true_dag)
                print('INFO: Check for '+self.Data_NAME +'.npz'+ '!')

            ################################ Deal with Multi-dimensions Causality Data ###################################
            elif os.path.exists(self.TS_path):
                read_Dir_TS=os.listdir(self.TS_path)
                Timeseries_List_path = self.File_PATH+'series_list.csv'
                Read_Timeseries = pd.read_csv(Timeseries_List_path)
                # print(len(Read_Timeseries), len(read_Dir_TS))
                if len(Read_Timeseries) >= len(read_Dir_TS):
                    print('INFO: Start Analyzing '+ self.Data_NAME + ' Time Series File!')
                    TS_List = read_Dir_TS
                else:
                    print('INFO: Start Analyzing '+ self.Data_NAME + ' Time Series List!')
                    TS_List = Read_Timeseries['Series_num']
                lds = pd.read_csv(self.TS_path+ TS_List[0], delimiter='\t', index_col=0, header=None) 
                # print(lds)
                n = len(TS_List)               
                T = lds.shape[1]
                d = lds.shape[0]
                # print(d, T, n)
                Raw_data = np.zeros((n, T, d))
                for ns in range(n):  
                    X = pd.read_csv(self.TS_path+ TS_List[ns], delimiter='\t', index_col=0, header=None) 
                    Raw_data[ns, :] = np.transpose(X)
                # save numpy to npz file
                matrix = np.zeros((d, d))
                np.fill_diagonal(matrix, 0)
                np.fill_diagonal(matrix[:, 1:], 1)
                np.savez(self.File_PATH + self.Data_NAME+'.npz', x=Raw_data , y=matrix)
                print('INFO: Check for '+self.Data_NAME +'.npz'+ '!')
            
            else:
                print('INFO: Wrong DataType!')
