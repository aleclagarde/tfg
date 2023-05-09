"""
Optimization script
===================

.. module:: optimize
   :platform: Linux
   :synopsis: Optimization script.

.. module_author:: Alec Lagarde Teixidó <aleclagarde@gmail.com>

This script performs the whole process of optimization. That is, loading the base models, saving the baselines and
perform pruning and quantization for both frameworks (PyTorch and TensorFlow).
"""

# Operating system
import os

# DataFrame modification
import pandas as pd

# Auxiliary functions
from optimize_utils import prune_torch, prune_tf, quantize_torch, quantize_tf, add_measurements, tflite_to_keras
from get_model_objects import get_model_objects


models = ['bert', 'gpt2', 't5', 'resnet', 'vit', 'convnext', 'codeparrot', 'codeberta', 'codegpt']
new_measurements_table = True

number_of_measurements = 30
pruning_cf = 0.2

if new_measurements_table:
    df = pd.DataFrame(columns=['timestamp', 'project_name', 'run_id', 'duration', 'emissions', 'emissions_rate',
                               'cpu_power', 'gpu_power', 'ram_power', 'cpu_energy', 'gpu_energy', 'ram_energy',
                               'energy_consumed', 'country_name', 'country_iso_code', 'region', 'cloud_provider',
                               'cloud_region', 'os', 'python_version', 'cpu_count', 'cpu_model', 'gpu_count',
                               'gpu_model', 'longitude', 'latitude', 'ram_total_size', 'tracking_mode', 'on_cloud',
                               'domain', 'model', 'framework', 'strategy', 'iteration'])
else:
    df = pd.read_csv('optimization_results.csv')


# Loop over the models and the coefficients and prune each model
for model_name in models:
    model_dict = get_model_objects(model_name=model_name)
    # Initialize models
    model_torch = model_dict["constructor_torch"].from_pretrained(model_dict["full_name"])
    # Codeparrot is an exception
    if model_name == 'codeparrot':
        model_tf = model_dict["constructor_tf"].from_pretrained(model_dict["full_name"], from_pt=True)
    else:
        model_tf = model_dict["constructor_tf"].from_pretrained(model_dict["full_name"])
    # Save baseline
    model_torch.save_pretrained(f"saved/{model_name}-torch-baseline")
    model_tf.save_pretrained(f"saved/{model_name}-tf-baseline")

    # PRUNING
    print("#############################################################################################")
    print("#############################################################################################")
    print(f"Pruning {model_name}")
    print("#############################################################################################")
    print("#############################################################################################")
    # PyTorch
    for i in range(number_of_measurements):
        print("#############################################################################################")
        print(f"Torch pruning: {model_name} with coefficient {pruning_cf}. Iteration: {i+1}")
        print("#############################################################################################")
        model_torch = model_dict["constructor_torch"].from_pretrained(f"saved/{model_name}-torch-baseline")
        prune_torch(model=model_torch, model_name=model_name, cf=pruning_cf, cv=model_name in ['resnet', 'vit', 'convnext'])
    df = add_measurements(dataframe=df, number_of_measurements=number_of_measurements, model_name=model_name,
                          framework='torch', strategy='pruning')

    # Tensorflow
    for i in range(number_of_measurements):
        print("#############################################################################################")
        print(f"TF pruning: {model_name} with coefficient {pruning_cf}. Iteration: {i+1}")
        print("#############################################################################################")
        model_tf = model_dict["constructor_tf"].from_pretrained(f"saved/{model_name}-tf-baseline")
        prune_tf(model=model_tf, model_name=model_name, cf=pruning_cf)
    df = add_measurements(dataframe=df, number_of_measurements=number_of_measurements, model_name=model_name,
                          framework='tf', strategy='pruning')

    # QUANTIZATION
    print("#############################################################################################")
    print("#############################################################################################")
    print(f"Quantization for {model_name}")
    print("#############################################################################################")
    print("#############################################################################################")
    # PyTorch
    for i in range(number_of_measurements):
        print("#############################################################################################")
        print(f"Torch quantization: {model_name}. Iteration: {i+1}")
        print("#############################################################################################")
        model_torch = model_dict["constructor_torch"].from_pretrained(f"saved/{model_name}-torch-baseline")
        quantize_torch(model=model_torch, model_name=model_name)
    df = add_measurements(dataframe=df, number_of_measurements=number_of_measurements, model_name=model_name,
                          framework='torch', strategy='quantization')

    # Tensorflow
    for i in range(number_of_measurements):
        print("#############################################################################################")
        print(f"TF quantization: {model_name}. Iteration: {i+1}")
        print("#############################################################################################")
        model_tf = model_dict["constructor_tf"].from_pretrained(f"saved/{model_name}-tf-baseline")
        tf_quantized_model = quantize_tf(model=model_tf)
    tflite_to_keras(model=tf_quantized_model, model_name=model_name)
    df = add_measurements(dataframe=df, number_of_measurements=number_of_measurements, model_name=model_name,
                          framework='tf', strategy='quantization')

df.to_csv('optimization_results.csv')

# Remove the emissions file
os.remove('emissions.csv')
