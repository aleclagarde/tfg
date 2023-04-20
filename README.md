# Energy efficiency measurement in optimization and inference of ML models

**Author: Alec Lagarde Teixidó**

**Director: Silverio Martínez-Fernández**

**Co-director: Matias Martinez**

The research of this TFG consists of understanding how existing ML optimization frameworks impact the energy efficiency 
and accuracy of ML models. We will study the following aspects of ML models deployment:

1. Energy consumption measurement after applying model optimization (e.g., quantization,
pruning)
2. Impact regarding the optimization framework (PyTorch and Tensorflow)
3. Context-aware evaluation of the energy efficiency for diverse ML models.

This project was developed from January to June 2023 as a Bachelor's Degree thesis for the Data Science and Engineering 
Degree in the Universitat Politècnica de Catalunya.

----------

## Environment
To replicate this projects environment, simply run ``conda create --name tfg python=3.10``. This will create the 
environment (named **tfg**) with Python 3.10.

Once the environment is created, you can install all necessary packages with ``pip install -r requirements.txt``.
It is necessary to install the packages with pip instead of conda because conda does not work for some packages.
Then, run ``conda activate tfg`` to activate the environment.

----------

## Structure and usage

It is structured in 3 directories:

1. **src**: performs the optimization and inference of the 9 models.
2. **results**: stores the testing results.
3. **reports**: analyzes the results.

Each of these directories, except **results** contains a python script that has to be executed using the environment 
created with ``environment.yml``.
