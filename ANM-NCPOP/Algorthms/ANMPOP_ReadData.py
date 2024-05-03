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


    def readable_File(self):
        read_Dir=os.listdir(self.File_PATH)
        count = 0
        readable_File = []
        for f in read_Dir:
            file = os.path.join(self.File_PATH, f)
            if os.path.isdir(file):
                count = count+1
            else:
                readable_File.append(f)
        return count,readable_File

    @staticmethod
    def Produce_Rawdata(self):
        Readable_File = self.readable_File(self.File_PATH)[1]
        # num_readable_File = len(self.File_PATH) - self.readable_File(self.File_PATH)[0]
        self.TS_path = self.File_PATH + self.Data_NAME + '_TS/'

        # Check empty files under riute
        if len(self.File_PATH) == 0:
            print('INFO: No Data Under the Current Route!')
        else:
            File_NAME = []
            File_TYPE = []

            # Delete files and list readable Files
            for i in Readable_File:
                File_NAME.append(re.split("\.", i)[0])
                File_TYPE.append(re.split("\.", i)[1])

            ###################################### Deal with Two Dimensions Causality Data ###################################
            if self.Data_NAME+'.npz' in Readable_File:
                Tests_data = np.load(self.Data_NAME+'.npz', allow_pickle=True)
                Raw_data = Tests_data['x']
                true_dag = Tests_data['y']

            elif self.Data_NAME+'.tar.gz' in Readable_File:
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

            elif self.Data_NAME+'.csv' in Readable_File:
                Raw_data = pd.read_csv(self.File_PATH+ self.Data_NAME+'.csv', header=0, index_col=0)
                true_dag = pd.read_csv(self.File_PATH+'true_graph.csv', header=0, index_col=0)

                # 将两个 numpy 数组保存到 npz 文件中
                np.savez(self.Data_NAME+'.npz', x=Raw_data , y=true_dag)

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
                    '''
                    # for i, j in combinations(range(len(TS_List)), 2):
                    [i, j] = [1,2]
                    x = pd.read_csv(TS_path+TS_List[i], header=0, index_col=0)
                    y = pd.read_csv(TS_path+TS_List[j], header=0, index_col=0)
                    print(TS_List[i], TS_List[j])
                    '''
            else:
                print('INFO: Wrong DataType!')
