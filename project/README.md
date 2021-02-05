# All you can creep

## Project requirements

### conda, miniconda, anaconda
Create a new conda environment for the project:  
```conda env create -f environment.yml```   


Activate the conda environment for the project:   
```conda activate autonome```   


You can update your conda environment with:   
```conda env update -f environment.yml```   

### Unity environment
- We use the unity version 2019.4.16f1. 
- Open the worm dynamic scene, delete all worms, but the first and build it for your operating system withing the folder `Unity/` and name `worm_single_environment`.

### MLFlow
- We use mlflow to track params, measures, create graphs and upload model artifacts.
- It is preinstalled with the conda environment.
- If you have your own remote server, please set the following environment variables:
  - `MLFLOW_TRACKING_URI`
  - `MLFLOW_TRACKING_USERNAME`
  - `MLFLOW_TRACKING_PASSWORD`

## Running the project

### Single worm training
You can use `python3 -m src.worm.train` to start a training run. You'll need to use specify as first positional argument the algorithm you intent to use.

Use `-h` to receive further information.

If you want to run multiple worms in parallel on the same machine. Please choose a different `-w` respectively. The unity environments require a socket and would fail otherwise if no other port was chosen.

### Tuning worm training
We implemented gridsearch and an evolutionary algorithm to explore the hyperparameter space. Please specify the algorithm with `-v` and the amount of runs to be run in parallel with `-p`. If you intend to run multiple tuning algorithms on the same system, You have to offset `-w` with at least the chosen level of parallism.


