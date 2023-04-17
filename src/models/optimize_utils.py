import torch
from torch.nn.utils import prune
import tensorflow as tf
import tensorflow_model_optimization as tfmot
import pandas as pd
from codecarbon import track_emissions


@track_emissions
def prune_torch(model, model_name: str, cf: float):
    """
    This function prunes a pretrained PyTorch model from the transformers library given the pruning coefficient and
    saves the pruned model in disk.

    :param model: PyTorch model from the transformers' library.
    :param model_name: Short model name (eg. 't5').
    :param cf: Pruning coefficient.
    """
    for name, param in model.named_parameters():
        if "embedding" in name:
            torch.nn.utils.prune.random_unstructured(param, name='weight', amount=cf)

    # Save the pruned model to disk
    model.save_pretrained(f"saved/{model_name}-torch-pruned")


@track_emissions
def prune_tf(model, model_name: str, cf: float):
    """
    This function prunes a pretrained PyTorch model from the transformers library given the pruning coefficient and
    saves the pruned model in disk.

    :param model: TensorFlow model from the transformers' library.
    :param model_name: Short model name (eg. 't5').
    :param cf: Pruning coefficient.
    """
    # Get a list of all the modules in the model
    submodules = model.submodules  # Access the submodules tuple as an attribute
    modules = list(submodules)  # Convert the submodules tuple to a list
    # Loop through each module and prune its parameters
    for module in modules:
        if isinstance(module, tf.keras.models.Sequential):
            # Prune the model with the given coefficient
            tfmot.sparsity.keras.prune_low_magnitude(model, sparsity=cf)

    # `prune_low_magnitude` requires a recompile.
    model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    # Save the pruned model to disk
    model.save_pretrained(f"saved/{model_name}-tf-pruned")


@track_emissions
def quantize_torch(model, model_name):
    # Quantize the model
    quantized_model = torch.quantization.quantize_dynamic(
        model,
        {torch.nn.Linear},
        dtype=torch.qint8
    )

    # Save the quantized model's JIT script module and state dictionary together using torch.jit.save
    torch.save(quantized_model, f"saved/{model_name}-torch-quantized.pth")


@track_emissions
def quantize_tf(model):
    # Convert the model to TensorFlow Lite format
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS
    ]

    # Optimize the model
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    return tflite_model


def convert_tflite_to_tf(model, model_name):
    # Load the TensorFlow Lite model into an interpreter
    interpreter = tf.lite.Interpreter(model_content=model)
    interpreter.allocate_tensors()

    # Extract the input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Create a new Sequential model
    tf_model = tf.keras.models.Sequential()

    # Iterate through the input details and add input layers to the model
    for input_detail in input_details:
        shape = input_detail['shape']
        dtype = input_detail['dtype']
        input_layer = tf.keras.layers.Input(shape=shape[1:], dtype=dtype)
        tf_model.add(input_layer)

    # Iterate through the output details and add output layers to the model
    for output_detail in output_details:
        shape = output_detail['shape']
        dtype = output_detail['dtype']
        output_layer = tf.keras.layers.Input(shape=shape[1:], dtype=dtype)
        tf_model.add(output_layer)

    # Save the TensorFlow model to disk
    tf.keras.models.save_model(tf_model, f"saved/{model_name}-tf-quantized")


def add_measurements(df, number_of_measurements, information):
    new_measurements = pd.read_csv('emissions.csv').tail(number_of_measurements)
    new_measurements['information'] = information
    return pd.concat([df, new_measurements], axis=0)
