# from Data_Standardization import*
from pickle import TRUE
import os
import re
import numpy as np
import pandas as pd
import tarfile
from itertools import combinations
import urllib
import hashlib
from urllib.error import URLError
USER_AGENT = "gcastle/dataset"

def _check_exist(root, filename, files):
    path_exist = os.path.join(root, filename.split('.')[0])
    processed_folder_exists = os.path.exists(path_exist)
    if not processed_folder_exists:
        return False

    return all(
        _check_integrity(os.path.join(path_exist, file)) for file in files
    )

def _read_data(root, filename, files):
    path_exist = os.path.join(root, filename.split('.')[0])

    result = []
    for file in files:
        if file.split('.')[-1] == 'csv':
            file_path = os.path.join(path_exist, file)
            result.append(pd.read_csv(file_path))
        elif file.split('.')[-1] == 'npy':
            file_path = os.path.join(path_exist, file)
            result.append(np.load(file_path))

    if len(result) == 2:
        result.append(None)

    return result

def _check_integrity(fpath, md5=None):
    if not os.path.isfile(fpath):
        return False
    if md5 is None:
        return True

    md5f = hashlib.md5()
    with open(fpath, 'rb') as f:
        md5f.update(f.read())

    return md5 == md5f.hexdigest()


def _download(root, url, filename, md5):
    """Download the datasets if it doesn't exist already."""

    os.makedirs(root, exist_ok=True)

    # download files
    for mirror in url:
        filepath = "{}{}".format(mirror, filename)
        savegz = os.path.join(root, filename)
        try:
            print("Downloading {}".format(filepath))
            response = urllib.request.urlopen( \
                urllib.request.Request( \
                    filepath, headers={"User-Agent": USER_AGENT}))
            with open(savegz, "wb") as fh:
                fh.write(response.read())

            tar = tarfile.open(savegz)
            names = tar.getnames()
            for name in names:
                tar.extract(name, path=root)
            tar.close()
        except URLError as error:
            print("Failed to download (trying next):\n{}".format(error))
            continue
        break
    else:
        raise RuntimeError("Error downloading {}".format(filename))

    # check integrity of downloaded file
    if not _check_integrity(savegz, md5):
        raise RuntimeError("File not found or corrupted.")

def load_dataset(name='IID_Test', root=None, download=False):
    """
    A function for loading some well-known datasets.

    Parameters
    ----------
    name: class, default='IID_Test'
        Dataset name, independent and identically distributed (IID),
        Topological Hawkes Process (THP) and real datasets.
    root: str
        Root directory in which the dataset will be saved.
    download: bool
        If true, downloads the dataset from the internet and
        puts it in root directory. If dataset is already downloaded, it is not
        downloaded again.

    Return
    ------
    out: tuple
        true_graph_matrix: numpy.matrix
            adjacency matrix for the target causal graph.
        topology_matrix: numpy.matrix
            adjacency matrix for the topology.
        data: pandas.core.frame.DataFrame
            standard trainning dataset.
    """

    if name not in DataSetRegistry.meta.keys():
        raise ValueError('The dataset {} has not been registered, you can use'
                         ' ''castle.datasets.__builtin_dataset__'' to get registered '
                         'dataset list'.format(name))
    loader = DataSetRegistry.meta.get(name)()
    loader.load(root, download)
    return loader.data, loader.true_graph_matrix, loader.topology_matrix


class BuiltinDataSet(object):

    def __init__(self):
        self._data = None
        self._true_graph_matrix = None
        self._topology_matrix = None

    def load(self, *args, **kwargs):
        raise NotImplementedError

    @property
    def data(self):
        return self._data

    @property
    def true_graph_matrix(self):
        return self._true_graph_matrix

    @property
    def topology_matrix(self):
        return self._topology_matrix

class RealDataSet(BuiltinDataSet):

    def __init__(self):
        super().__init__()
        self.url = None
        self.tar_file = None
        self.md5 = None
        self.file_list = None

    def load(self, root=None, download=False):

        if root is None:
            root = './'

        if _check_exist(root, self.tar_file, self.file_list):
            self._data, self._true_graph_matrix, self._topology_matrix = \
                _read_data(root, self.tar_file, self.file_list)
            return

        if download:
            _download(root, self.url, self.tar_file, self.md5)

        if not _check_exist(root, self.tar_file, self.file_list):
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it.')

        self._data, self._true_graph_matrix, self._topology_matrix = \
            _read_data(root, self.tar_file, self.file_list)


class V18_N55_Wireless(RealDataSet):
    """
    A function for loading the real dataset: V18_N55_Wireless
    url: https://raw.githubusercontent.com/gcastle-hub/dataset/master/alarm/18V_55N_Wireless.tar.gz
    """

    def __init__(self):
        super().__init__()
        self.url = ['https://raw.githubusercontent.com/gcastle-hub/dataset/master/alarm/']
        self.tar_file = "18V_55N_Wireless.tar.gz"
        self.md5 = "36ee135b86c8dbe09668d9284c23575b"
        self.file_list = ['Alarm.csv', 'DAG.npy']


class V24_N439_Microwave(RealDataSet):
    """
    A function for loading the real dataset: V24_N439_Microwave
    url: https://raw.githubusercontent.com/gcastle-hub/dataset/master/alarm/24V_439N_Microwave.tar.gz
    """

    def __init__(self):
        super().__init__()
        self.url = ['https://raw.githubusercontent.com/gcastle-hub/dataset/master/alarm/']
        self.tar_file = "24V_439N_Microwave.tar.gz"
        self.md5 = "b4c8b32d34c04a86aa93c7259f7d086c"
        self.file_list = ['Alarm.csv', 'DAG.npy', 'Topology.npy']


class V25_N474_Microwave(RealDataSet):
    """
    A function for loading the real dataset: V25_N474_Microwave
    url: https://raw.githubusercontent.com/gcastle-hub/dataset/master/alarm/25V_474N_Microwave.tar.gz
    """

    def __init__(self):
        super().__init__()
        self.url = ['https://raw.githubusercontent.com/gcastle-hub/dataset/master/alarm/']
        self.tar_file = "25V_474N_Microwave.tar.gz"
        self.md5 = "51f43ed622d4b44ef6daf8fabf81e162"
        self.file_list = ['Alarm.csv', 'DAG.npy', 'Topology.npy']


class DataSetRegistry(object):
    '''
    A class for resgistering the datasets, in which each dataset
    can be loaded by 'load_dataset' api.
    '''

    meta = {'V18_N55_Wireless': V18_N55_Wireless,
            'V24_N439_Microwave': V24_N439_Microwave,
            'V25_N474_Microwave': V25_N474_Microwave}



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
    Input data: npz
            Raw_data：[d, n, T] sample time series
            true_dag: true_causal_matrix
    File_PATH_Datasets:
            Route of saving test data

    Examples
    -------------------------------------------------------------------------------------------------
    >>> File_PATH = "../Test/Examples/Test_data/"
    >>> file_name = 'Telephone'
    >>> dt = Real_Data_Standardization(File_PATH, file_name)
    >>> dt.standardize_data()

    >>> File_PATH = "../Test/Datasets/Synthetic datasets/Krebs_Cycle/"
    >>> file_name = 'Krebs_Cycle'
    >>> dt = Real_Data_Standardization(File_PATH, file_name)
    >>> dt.standardize_data()

    >>> File_PATH = "../Test/Datasets/Real_data/Microwave/"
    >>> file_name = 'V24_N439_Microwave'
    >>> dt = Real_Data_Standardization(File_PATH, file_name)
    >>> dt.standardize_data()
    '''

    def __init__(self, File_PATH='Kreb_Cycles/', filename='Krebs_Cycle'):
        self.File_PATH = File_PATH
        self.filename = filename

    def standardize_data(self):
        ################################################  Create Ground Tier Folders #############################################
        self.File_PATH_Base = self.File_PATH +'Result_'+ self.filename +'/'

        ################################################  Create First Tier Folders #############################################
        self.File_PATH_Datasets = self.File_PATH_Base + 'Datasets_'+ self.filename +'/'
        if not os.path.exists(self.File_PATH_Datasets):
            os.makedirs(self.File_PATH_Datasets)
        print('ANM-NCPOP INFO: Created Datasets' + ' File!')

        Raw_data = Real_Data_Standardization.Produce_Rawdata(self)[0]
        true_dag = Real_Data_Standardization.Produce_Rawdata(self)[1]

        # save numpy to npz file
        nn = len(true_dag)
        ne = np.count_nonzero(true_dag)
        data_name = self.filename  +'_'+str(nn)+'Nodes_'+str(ne)+'Edges_TS'
        if self.filename in ['IID_Test','THP_Test','V18_N55_Wireless', 'V24_N439_Microwave', 'V25_N474_Microwave']:
            topology_matrix_devices = Real_Data_Standardization.Produce_Rawdata(self)[2]
            np.savez(self.File_PATH_Datasets + data_name +'.npz', x=Raw_data , y=true_dag , z=topology_matrix_devices)
        else:
            np.savez(self.File_PATH_Datasets + data_name +'.npz', x=Raw_data , y=true_dag)
        print('ANM-NCPOP INFO: Finished '+ data_name+' dataset standardization!')

    @staticmethod
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

        # Website data
        if self.filename in ['IID_Test','THP_Test','V18_N55_Wireless', 'V24_N439_Microwave', 'V25_N474_Microwave']:
            Raw_data, true_dag, topology_matrix_devices  = load_dataset(self.filename, download=True)
            return Raw_data, true_dag, topology_matrix_devices

        else:
            # Check empty files under riute
            if len(self.Read_File ) == 0:
                raise ValueError('No Data Under the Current Route!')
            else:
                self.File_PATH_TS = self.File_PATH +self.filename +'_TS/'
                File_NAME = []
                File_TYPE = []
                # Delete files and list readable Files
                for i in self.Read_File:
                    File_NAME.append(re.split("\.", i)[0])
                    File_TYPE.append(re.split("\.", i)[1])

                ###################################### Deal with Two Dimensions Causality Data ###################################
                if self.filename+'.npz' in self.Read_File:
                    Test_data = np.load(self.File_PATH + self.filename+'.npz', allow_pickle=True)
                    Raw_data = Test_data['x']
                    true_dag = Test_data['y']
                    return Raw_data, true_dag

                elif self.filename+'.tar.gz' in self.Read_File:
                    # open file
                    file = tarfile.open(self.File_PATH + self.filename + '.tar.gz')
                    file_names = file.getnames()
                    # extract files
                    file.extractall(self.File_PATH)
                    file.close()
                    Raw_data = np.load(self.File_PATH+file_names[2])
                    true_dag = pd.read_csv(self.File_PATH+file_names[3])
                    return Raw_data, true_dag

                elif self.filename+'.csv' in self.Read_File:
                    Raw_data = pd.read_csv(self.File_PATH+ self.filename+'.csv', header=0, index_col=False)
                    true_dag = pd.read_csv(self.File_PATH+'true_graph.csv', header=0, index_col=0)
                    return Raw_data, true_dag

                ################################ Deal with Multi-dimensions Causality Data ###################################
                elif os.path.exists(self.File_PATH_TS):
                    read_Dir_TS=os.listdir(self.File_PATH_TS)
                    true_graph = np.load(self.File_PATH+'true_graph.npz')

                    # labels = ["FUMARATE", "GTP", "H2O", "CIS-ACONITATE", "MALATE",
                    # "OXALOACETATE", "FAD", "SUCCINYL-COA", "NAD",
                    #           "A-K-GLUTARATE", "GDP", "NADH", "CITRATE", "SUCCINATE",
                    # "ISOCITRATE", "ACETY-COA"]
                    #true_dag = pd.DataFrame(true_graph['arr_0'],  index=labels, columns=labels)
                    true_dag = pd.DataFrame(true_graph['arr_0'])

                    # print(true_dag)
                    lds = pd.read_csv(self.File_PATH_TS+ read_Dir_TS[0], delimiter='\t', index_col=0, header=None)
                    feature_name = np.array(lds.index)
                    feature_num = len(feature_name)
                    sample_num = len(read_Dir_TS)
                    T_num = lds.shape[1]
                    # if labels == feature_name:
                    Raw_data = np.zeros((feature_num, sample_num, T_num))
                    for ns in range(sample_num):
                        X = pd.read_csv(self.File_PATH_TS+ read_Dir_TS[ns], delimiter='\t', index_col=0, header=None)
                        X_trans = np.transpose(X)
                        for fn in range(feature_num):
                            Raw_data[fn, ns, :] = list(X_trans[feature_name[fn]])
                    return Raw_data, true_dag

                else:
                    raise ValueError('Unknown input data type.')
