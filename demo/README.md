# Demo of one ML deployment

This is a demo of one Machine Learning deployment using Daniel's TFG replication package as a main source.

This demo deploys the T5 translation model to azure 4 times with different pruning percentages (0, 10, 20 and 30).

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