import BuiltinDataSet
import NCPOLR
import popcorn

method = 'nonlinear'
File_PATH_Base = '/content/drive/MyDrive/Colab Notebooks/Causality_NotesTest/'+'Result_'+method.capitalize()+'/'
# gauss, exp, gumbel, uniform, logistic (linear);
# mlp, mim, gp, gp-add, quadratic (nonlinear).
sem_type = 'mlp'
nodes = range(6,14,3)
edges = range(10,21,5)
start = 5
stop = 40
step = 5
rt = ANM_NCPOP(method, File_PATH_Base, sem_type, nodes, edges, start, stop, step)
rt.Popcorn()
