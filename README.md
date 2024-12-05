# Implicit Regularization of AdaDelta

The code in this repository can be used to run the experiments from the paper:

[**Implicit Regularization of AdaDelta**](https://openreview.net/pdf?id=nm40lbbwoR) <br>
*Matthias Englert, Ranko Lazic, Avi Semler*


## Usage

### MLP Experiments

To run the MLP experiments, use the following command:
```bash
python mlp_exp.py
```
This generates a number of csv files in the current folder, which contain the results of the experiments. Specifically, it generates the following files:

- `data.csv`: Contains the generated training data.
- `avg_accuracy.csv`: Contains the average accuracy for each epoch for each optimizer.
- `avg_loss.csv`: Contains the average loss for each epoch for each optimizer.
- `margins.csv`: Contains the margin for each epoch for optimizer.
- `reciprocal_AdaDelta.csv`: Contains the final reciprocal square root of the adapter for AdaDelta for each run.
- `reciprocal_AdaDeltaN.csv`: Contains the final reciprocal square root of the adapter for AdaDeltaN for each run.
- `reciprocal_AdaDeltaNS.csv`: Contains the final reciprocal square root of the adapter for AdaDeltaNS for each run.
- `reciprocal_AdaDeltaS.csv`: Contains the final reciprocal square root of the adapter for AdaDeltaS for each run.

### MNIST and CIFAR Experiments

To run the MNIST and CIFAR experiments, use the following command:
```bash
python mnist_and_cifar_exp.py --experiment [mnist|cifar] --mode [coarse|fine] --runs [number_of_runs] --device [cpu|cuda]
```

This creates a folder `runs` in the current directory and saves the results of the experiments in this folder.

To then collate the results of all completed experiments, use the following command:
```bash
python collate_results.py
```
This generates a number of csv files in the `aggregated_data` folder, which contain the collated results of all experiments.


