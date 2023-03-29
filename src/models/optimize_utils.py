import transformers
import torch
from torch.nn.utils import prune
import tensorflow as tf
import tensorflow_model_optimization as tfmot
from codecarbon import track_emissions


@track_emissions
def prune_torch(model, model_name: str, modules, cf: float):
    for module in modules:
        # Loop through each module and prune its parameters
        if isinstance(module, torch.nn.Linear) or \
                isinstance(module, transformers.pytorch_utils.Conv1D):
            torch.nn.utils.prune.random_unstructured(module, name='weight', amount=cf)

    # Save the pruned model to disk
    model.save_pretrained(f"{model_name}-torch-pruned-{cf}")


@track_emissions
def prune_tf(model, model_name: str, modules, cf: float):
    # Loop through each module and prune its parameters
    for module in modules:
        if isinstance(module, tf.keras.models.Sequential):
            # Prune the model with the given coefficient
            tfmot.sparsity.keras.prune_low_magnitude(model, sparsity=cf)

    # `prune_low_magnitude` requires a recompile.
    model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    # Save the pruned model to disk
    model.save_pretrained(f"{model_name}-tf-pruned-{cf}")


@track_emissions
def quantize_torch(model, model_name):
    # Quantize the model
    quantized_model = torch.quantization.quantize_dynamic(
        model,
        {torch.nn.Linear},
        dtype=torch.qint8
    )

    # Save the pruned model to disk
    quantized_model.save_pretrained(f"{model_name}-torch-quantized")


@track_emissions
def quantize_tf(model, model_name):
    # Convert the model to TensorFlow Lite format
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    # Quantize the model
    quantized_model = tf.lite.Interpreter(model_content=tflite_model)
    quantized_model.allocate_tensors()
    quantized_model.quantize()

    # Save the pruned model to disk
    quantized_model.save_pretrained(f"{model_name}-tf-quantized")
