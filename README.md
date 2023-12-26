## The procedure for causal learning
### Generate the artificial true causal graph and observation data based on the SCM.
### Learn the Causal Structure beneath the observation data.
### Visualize the comparison of estimated/true graphs using a heat map.
### Calculate Metrics.
### Demonstrate in a heatmap.

## Installation

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
- ncpol2sdpa
- MOSEK (>= 9.3)
- gcastle (>= 1.0.3)


### PIP installation
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
