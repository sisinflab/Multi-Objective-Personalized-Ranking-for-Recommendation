# Multi-Objective Personalized Ranking for Recommendation
This repository contains the source codes and datasets of the paper _Multi-objective Personalized Ranking for Recommendation_ submitted at SIGIR 2024.

### Requirements
We implemented and tested the models in Python `3.8.10`, with `PyTorch==2.0.1` and CUDA `11.7`. The `LightGCN` moodel require `PyTorch Geometric`. Then, the requirements listed in the `req_MPR.txt` file refer to these versions. You may create the virtual environment with the requirements file as follows:

```
$ python3 -m venv venv
$ source venv/bin/activate
$ pip install --upgrade pip
$ pip install -r req_MPR.txt
```

### Data
In the folder `data`, you can find the data used in our work (`Amazon Baby`, `Facebook Books`, `MovieLens1M`). We provide the split version of the data.

### Run the models
In the following, we explain how to run the models within the paper.
- BPRMF, LightGCN, MultiFR, and MPR can be executed through the `main.py` script. Specifically, you should refer to the configuration files contained into the folder `config_files`. You may train the model by running the following code:
  ```
  $ python3 -u main.py --config [CONFIGURATION_FILE_NAME]
  ```
- CPFAIR can be executed through the `main_cpfair.py` script. It requires the matrix of scores (user, item) predicted by a backbone model (in this paper BPRMF and LightGCN). For this reason, this code is prepared in order to load such matrix saved with `.npz` [extension](https://numpy.org/doc/stable/reference/generated/numpy.savez_compressed.html) from a folder called `arrays`.  You may train the model by running the following code:
  ```
  $ python3 -u main_cpfair.py
  ```

### Evaluation
To evaluate the models, we relied on the public and open-source framework Elliot. Here, given a pre-obtained recommendation list by running the models as explained above, you can compute the metrics discussed within the paper. Due to GitHub size limitations, we load a compressed version of Elliot on this repository (`elliot_MPR.zip`). Please, refer to Elliot official [documentation](https://elliot.readthedocs.io/en/latest/) for further details on how compute the metrics. We add an example of configuration file in the `config_files` folder within the compressed file.
