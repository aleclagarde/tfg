# Energy efficiency measurement and optimization of ML models deployment in cloud providers

**Author: Alec Lagarde Teixidó**

**Director: Silverio Martínez-Fernández**

**Co-director: Matias Martinez**

The research of this TFG consists of understanding how existing ML inference cloud providers optimize calculations for 
energy reduction. We will study the following aspects of ML models deployment: 

1. Energy consumption measurement after 
applying model optimization (e.g., quantization, pruning); 

2. Context-aware evaluation of the energy efficiency for 
diverse cloud providers (e.g., AWS, Azure).

This project was developed from January to June 2023 as a Bachelor's Degree thesis for the Data Science and Engineering 
Degree in the Universitat Politècnica de Catalunya.

----------

## Environment
To replicate this projects environment, simply run ``conda env create -f environment.yml``. This will create the 
environment (named **tfg**) and install all necessary libraries.

Once the environment is created, run ``conda activate tfg`` to activate the environment.

----------

## Structure and usage

It is structured in 4 directories:

1. **src**: performs the actual deployment.
2. **testing**: performs the inference.
3. **results**: stores the testing results.
4. **reports**: analyzes the results.

Each of these directories, except **results** contains a python script that has to be executed using the environment 
created with ``environment.yml``.

To do this, a new main script was created in the demo directory to execute everything at once.
``main.py`` executes all the scripts and accepts the ``--prune_pct`` flag, which is the pruning coefficient, that is the
percentage of weights to be pruned.