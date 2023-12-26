# Causal Model Learning with Non-Commutative Polynomial Optimization

Welcome to the repository for causal discovery deliverable 3.2! This codebase encompasses notebooks and test results associated with each figure in the deliverable.

## Causal Learning Process
1. **Generate Data:** Use the `Ancpop_Simulation` class in `Ancpop_Simulation.py` to create artificial true causal graphs and observation data.
2. **Learn Structure:** Uncover the causal structure beneath the observation data.
3. **Visualize Comparison:** Generate heat maps to compare estimated and true graphs.
4. **Calculate Metrics:** Assess the performance metrics.
5. **Demonstrate in Heatmap:** Illustrate results in a heatmap for multiple datasets.

## Getting Started
1. **Synthetic Data:** Utilize the `Ancpop_Simulation` class to generate custom datasets.
2. **Real-world Data:** Employ the `Ancpop_Real` class in `Ancpop_Real.py` to test real data from the `./Real Data` folder. Alternatively, use generated data in `./Synthetic data`.
3. **Experiments:** Experiment scripts for proposed methods and baselines are available in Jupyter notebooks named "Methods_withdevice.ipynb".
4. **Results:** Access all results and figures mentioned in the paper from the `./result` folder. Repeat the results using "ANCPOP_Simulation_test.ipynb".

## Running Notebooks
- **ANCPOP_Simulation_test.ipynb:** Execute this notebook for all experiments on synthetic and real-world data using the ANM_NCPOP approach. Upload datasets and the notebook to [Colab](https://colab.research.google.com/) for seamless execution.

## Installation
To run experiments locally, ensure you have the following dependencies installed:
- Python (>= 3.6, <= 3.9)
- tqdm (>= 4.48.2)
- NumPy (>= 1.19.1)
- Pandas (>= 0.22.0)
- SciPy (>= 1.7.3)
- scikit-learn (>= 0.21.1)
- Matplotlib (>= 2.1.2)
- NetworkX (>= 2.5)
- PyTorch (>= 1.9.0)
- ncpol2sdpa 1.12.2 [Documentation](https://ncpol2sdpa.readthedocs.io/en/stable/index.html)
- MOSEK (>= 9.3) [MOSEK](https://www.mosek.com/)
- gcastle (>= 1.0.3)

### PIP Installation
```bash
# Execute the following commands to run the notebook directly in Colab. Ensure your MOSEK license file is in one of these locations:
#
# /content/mosek.lic   or   /root/mosek/mosek.lic
#
# inside this notebook's internal filesystem.
# Install MOSEK and ncpol2sdpa if not already installed
pip install mosek 
pip install ncpol2sdpa
pip install gcastle==1.0.3





# Learning Causal Models as a Problem in Non-Commutative Polynomial Optimization
This is the source code for the causal discovery deliveriable 3.2. It includes the notebooks and test results for each figure in the deliveriable.


## The procedure for causal learning
- Generate the artificial true causal graph and observation data based on the SCM.
- Learn the Causal Structure beneath the observation data.
- Visualize the comparison of estimated/true graphs using a heat map.
- Calculate Metrics.
- Demonstrate in a heatmap for multiple data.


## Get Started
1.Synthetic data: class Ancpop_Simulation in class Ancpop_Simulation.py can be used to generate your own dataset
2.Real-world data: class Ancpop_Real in Ancpop_Real.py can be employed to test real data in the folder./Real Data. As an alternative choice, you can also utilize generated data in the folder ./Synthetic data.
3.Experiments: We provide the experiment scripts of both proposed methods and baselines. You can access them through jupyter notebooks with the name "Methods_withdevice.ipynb".
4. Results: All results and figures mentioned in the paper are under the folder ./result. You can utilize "ANCPOP_Simulation_test.ipynb" to repeat the results.


## Run Notebooks
ANCPOP_Simulation_test.ipynb: This notebook contains scripts for all experiments on both synthetic and real-world data using ANM_NCPOP approach. You can upload datasets and the notebook to Colab https://colab.research.google.com/ and run it. 


## Installation
If one would like to run experiments on local:
### Dependencies
Requires:
- python (>= 3.6, <=3.9)
- tqdm (>= 4.48.2)
- numpy (>= 1.19.1)
- pandas (>= 0.22.0)
- scipy (>= 1.7.3)
- scikit-learn (>= 0.21.1)
- matplotlib (>=2.1.2)
- networkx (>= 2.5)
- torch (>= 1.9.0)
- ncpol2sdpa  1.12.2 https://ncpol2sdpa.readthedocs.io/en/stable/index.html
- MOSEK (>= 9.3)  https://www.mosek.com/
- gcastle (>= 1.0.3)


### PIP installation
If you use Mosek as a solver, a license is required. After applying for a license from Mosek, you can put "mosek.lic" to colab Files.
```bash
# To execute the notebook directly in colab make sure your MOSEK license file is in one the locations
#
# /content/mosek.lic   or   /root/mosek/mosek.lic
#
# inside this notebook's internal filesystem.
# Install MOSEK and ncpol2sdpa if not already installed
pip install mosek 
pip install ncpol2sdpa
pip install gcastle==1.0.3
```
