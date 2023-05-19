"""
Optimization utils
==================

.. module:: optimize_utils
   :platform: Linux
   :synopsis: Optimization auxiliary functions.

.. module_author:: Alec Lagarde Teixidó <aleclagarde@gmail.com>

This script contains auxiliary functions for the optimization part of the project, such as the functions that perform
the optimizations, a function needed to convert back from TFLite model to TensorFlow model, and one to add the
measurements to the results table.

.. autosummary::
   :toctree: generated/

   prune_torch
   prune_tf
   quantize_torch
   quantize_tf
   convert_tflite_to_tf
   add_measurements
"""

# PyTorch libraries
import torch
from torch.nn.utils import prune

# TensorFlow libraries
import tensorflow as tf
import tensorflow_model_optimization as tfmot

# DataFrame modification
import pandas as pd

# Emissions measuring
from codecarbon import track_emissions

import shutil
import subprocess


@track_emissions
def prune_torch(model, model_name: str, cf: float, cv: bool):
    """
    This function prunes a pretrained PyTorch model from the transformers library given the pruning coefficient and
    saves the pruned model in disk.

    :param model: PyTorch model from the transformers' library.
    :param model_name: Short model name (eg. 'gpt2').
    :param cf: Pruning coefficient.
    :param cv: Whether the model is from the Computer Vision domain.
    """
    if cv:
        to_prune = model.classifier[-1]
        prune.l1_unstructured(to_prune, name='weight', amount=cf)
    else:
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear) and "weight" in name:
                prune.l1_unstructured(module, name='weight', amount=cf)

    # Save the pruned model to disk
    model.save_pretrained(f"saved/{model_name}-torch-pruned")


@track_emissions
def prune_tf(model, model_name: str, cf: float):
    """
    This function prunes a pretrained TensorFlow model from the transformers library given the pruning coefficient and
    saves the pruned model in disk.

    :param model: TensorFlow model from the transformers' library.
    :param model_name: Short model name (eg. 'gpt2').
    :param cf: Pruning coefficient.
    """
    # Define the pruning parameters
    pruning_params = {
        'pruning_schedule': tfmot.sparsity.keras.ConstantSparsity(target_sparsity=cf, begin_step=0, end_step=-1,
                                                                  frequency=100),
        'block_size': (1, 1)
    }

    # Loop through the model's submodules and prune its parameters
    for submodule in model.submodules:
        if 'mlp' in submodule.name or 'dense' in submodule.name or 'convolution' in submodule.name:
            # Create a custom PrunableLayer for the unsupported layer
            class PrunableLayer(tf.keras.layers.Layer, tfmot.sparsity.keras.PrunableLayer):
                def __init__(self, layer):
                    super(PrunableLayer, self).__init__()
                    self.layer = layer

                def get_prunable_weights(self):
                    return self.layer.trainable_weights

            # Prune the custom PrunableLayer
            prunable_layer = PrunableLayer(submodule)
            tfmot.sparsity.keras.prune_low_magnitude(prunable_layer, **pruning_params)

    # `prune_low_magnitude` requires a recompile.
    model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    # Save the pruned model to disk
    model.save_pretrained(f"saved/{model_name}-tf-pruned")


@track_emissions
def quantize_torch(model, model_name: str):
    """
    This function quantize a pretrained PyTorch model from the transformers library and saves the quantized model in
    disk.

    :param model: PyTorch model from the transformers' library.
    :param model_name: Short model name (eg. 'gpt2').
    """
    # Quantize the model
    quantized_model = torch.quantization.quantize_dynamic(
        model,
        {torch.nn.Linear},
        dtype=torch.qint8
    )

    # Save the quantized model
    torch.save(quantized_model, f"saved/{model_name}-torch-quantized.pth")


@track_emissions
def quantize_tf(model_name: str, long_model_name: str):
    """
    This function quantizes a pretrained TensorFlow model from the transformers library using Optimum
    and returns the quantized model in Keras format.

    :param model_name: Short model name (eg. 'gpt2').
    :param long_model_name: Full model name (eg. microsoft/resnet-50)
    """
    onnx_model_path = f"saved/{model_name}-onnx"
    quantized_model_path = f"saved/{model_name}-tf-quantized"

    # Construct the command to run
    command = ["optimum-cli", "export", "onnx", "--model", long_model_name, "--framework", "tf", "--optimize", "O2",
               onnx_model_path]

    # Run the command
    subprocess.run(command)

    # Construct the command to run
    command = ["optimum-cli", "onnxruntime", "quantize", "--onnx_model", onnx_model_path, "--avx512", "-o",
               quantized_model_path]

    # Run the command
    subprocess.run(command)

    # Delete auxiliary onnx model
    shutil.rmtree(onnx_model_path)


def add_measurements(dataframe: pd.DataFrame, number_of_measurements: int, model_name: str, framework: str,
                     strategy: str) -> pd.DataFrame:
    """
    This function takes a Pandas DataFrame and adds *number_of_measurements* rows with the string *information* in the
    'information' column, along with the measurement number in the 'iteration' column. This is used to add emissions
    measurements to the results table.

    :param dataframe: Pandas DataFrame to add rows.
    :param number_of_measurements: Number of measurements (rows) to add.
    :param model_name: String to store in the new rows' model column. It is also used to get the domain.
    :param framework: String to store in the new rows' framework column.
    :param strategy: String to store in the new rows' strategy column.
    :return: Pandas DataFrame with the added measurements.
    """
    new_measurements = pd.read_csv(filepath_or_buffer='emissions.csv').tail(n=number_of_measurements)

    if model_name in ['gpt2', 'opt', 'gptj']:
        domain = 'NLP'
    elif model_name in ['resnet', 'vit', 'regnet']:
        domain = 'CV'
    else:
        domain = 'CG'

    new_measurements['domain'] = domain
    new_measurements['model'] = model_name
    new_measurements['framework'] = framework
    new_measurements['strategy'] = strategy
    new_measurements['iteration'] = [x for x in range(1, number_of_measurements+1)]
    return pd.concat([dataframe, new_measurements], axis=0)
