# Multi-Objective Personalized Ranking for Recommendation
This repository contains the source codes and datasets of the paper _Multi-objective Personalized Ranking for Recommendation_ submitted at SIGIR 2024.


### Data
In the folder `data`, you can find the data used in our work (`Amazon Baby`, `Facebook Books`, `MovieLens1M`). We provide the split version of the data.

### Run the models

In the following, we explain how to run the models within the paper.
- BPRMF, LightGCN, MultiFR, and MPR can be executed through the `main.py` script. Specifically, you should refer to the configuration files contained into the folder `config_files`. You may train the model by running the following code:
  ```
  $ python3 -u main.py --config [CONFIGURATION_FILE_NAME]
  ```
- CPFAIR can be executed through the `main_cpfair.py` script. It requires the matrix of scores (user, item) predicted by a backbone model (in this paper BPRMF and LightGCN). For this reason, this code is prepared in order to load such matrix saved with `.npz` [extension](https://numpy.org/doc/stable/reference/generated/numpy.savez_compressed.html) from a folder called `arrays`.  
