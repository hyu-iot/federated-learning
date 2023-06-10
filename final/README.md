# Scalable Federated Learning Simulation Using Flower

## About

This framework is for scalable federated learning simulation using the Flower framework. User can test various CNN models using many federated learning aggregation models. The user may select the options using `config.json`. Visualization is also able selecting models and strategies to compare.

## Installation

To install, all that needs to be done is clone this repository to the desired directory.

## Dependencies

`pytorch`, `matplotlib`, `pandas` is essential.

## Simulation

To start a simulation, run [`run.py`](run.py) from the repository's root directory:

```shell
python run.py
  --config=config.json
  --log=INFO
```

* `run.py` flags

  * `--config` (`-c`): path to the configuration file to be used.
  * `--log` (`-l`): level of logging info to be written to console, defaults to `INFO`.

## Results
The simulation will create results in the `./result` directory. It will create a directory using the timestamp when the simulation started. Inside the directory, you will have 3 results. 
* Copy of the used `config.json`
* `csv` file of the results of each model.
    * It will contain the `loss`, `accuracy`, `f1-score`, `precision`, and `recall` of each `model` and `strategy`.
* `png` of the visualized graph.

## Visualization
The visualization options will create a graph in a `png` file. The created graph shows the selected `metric` as the y-axis, and the rounds as the x-axis. You could check out how the `metric` changes as the rounds progressed.


# Configuration 

This framework uses configuration files to manage the parameters of a simulation. These files, typically named `config.json`, are of JSON format and have specific keys for setting parameters.

Use the provided example [`config.json`](final/configs/config.json.template) for reference.

## Parameters 

Configuration parameters are divided into the following nested sections within a `config.` file:

* `clients`: Nested options pertaining to the clients.
  * `total`: `[positive integer]`, Number of total clients to be managed by server.
  * `per_round`: `[positive integer]`, Number of clients participating in each round.
  * `validation_ratio`: `[float between 0 and 1]`, Rate of the validation set from the train set. For instance, 0.1 means 10% of the train set will be used as the validation set.


* `data`: Nested options for how data is managed
  * `dataset`: `["cifar10", "cifar100"]`, Dataset used to train and test.
  * `batch_size`: `[10]`, Size of data batch for each gradient update on a client.
  * `remain_ratio`: `[float between 0 and 1]`,TODO: need to update.
  * `random_seed`: `[positive integer]`, Seed for the random_split. Select when you want the seed to be fixed.
* `model`: Nested options for the model structure to train.
  * `[model_name:"densenet121", "mobilenet_v2", "googlenet", "resnet18", "resnet34", "resnet50", "vgg11", "vgg13", "vgg16", "vgg19"]`, Select model structure to train. The model is loaded by `torchvision.models`.
    *  `pretrained_path`: `[string]`, path of directory of the pretrained model to be used.
    *  `args`: Nested class of additional arguments used for the model.
       *  `` //TODO: need to check.
* `strategy`: `["Centralized", "FedAvg", "FedAvgM", "FedOpt", "FedProx"]`, Strategies to compare. Single or multiple strategies may be selected.

* `federated learning`: Specifications of federated learning task.
  * `rounds`: `[positive integer]`, Number of rounds of federated learning to simulate.
  * `epochs`: `[positive integer]`, Number of epochs each client performs each round.
  * `learning_rate `: `[float between 0 and 1]`, Learning rate for training model.
  * `weight_decay`: `[float between 0 and 1]`, Weight decay reate for training model.
  * `fraction_fit`: `[float between 0 and 1]`, Sampling rate of available clients for training. 1.0 means 100% of clients participates training.
  * `fraction_evaluate`: `[float between 0 and 1]`, Sampling rate of available clients for evaluation. 1.0 means 100% of clients participates evaluation.
  * `min_fit_clients`: `[positive integer]`, Minimum number of clients to sample during training. For instance if it is 10, it means to never sample less than 10 clients for training.
  * `min_evaluate_clients`: `[positive integer]`, Minimum number of clients to sample during evaluation. For instance if it is 10, it means to never sample less than 10 clients for evaluation.
  * `min_available_clients`: `[positive integer]`, Minimum number of clients to start simulation. For instance if it is 10, it waits until 10 clients are available.

* `paths`: Nested options for paths used by the framework.
  * `data`: `[string]`, Path to root data directory. Defaults to `./data`.
  * `model`: `[string]`, Path to root model directory. Defaults to `./models`.

* `visualization`: Nested options for visualization options.
  * `model`: `["model.model_name"]`, Models to compare. It must only contain the model names from `model`'s `model_name` option. Single or multiple models are available.
  * `strategy`: `["strategy"]`, Strategies to compare. It must only contains strategies from `strategy` option. Single or multiple strategies are available.
  * `metrics`: `["loss", "accuracy", "f1-score", "precision", "recall"]`,  Metrics to visualize. Single or multiple metrics are available.
