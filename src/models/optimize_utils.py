import transformers
import torch
from torch.nn.utils import prune
import tensorflow as tf
import tensorflow_model_optimization as tfmot
from codecarbon import track_emissions
import pickle


@track_emissions
def prune_torch(model, model_name: str, cf: float):
    # Get a list of all the modules in the model
    modules = list(model.modules())
    for module in modules:
        # Loop through each module and prune its parameters
        if isinstance(module, torch.nn.Linear) or \
                isinstance(module, transformers.pytorch_utils.Conv1D):
            torch.nn.utils.prune.random_unstructured(module, name='weight', amount=cf)

    # Save the pruned model to disk
    model.save_pretrained(f"saved/{model_name}-torch-pruned-{cf}")


@track_emissions
def prune_tf(model, model_name: str, cf: float):
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
    model.save_pretrained(f"saved/{model_name}-tf-pruned-{cf}")


@track_emissions
def quantize_torch(model, model_name):
    # Quantize the model
    quantized_model = torch.quantization.quantize_dynamic(
        model,
        {torch.nn.Linear},
        dtype=torch.qint8
    )

    # Save the state dictionary of the quantized model to disk
    # Can't use save_pretrained with quantized models, use torch.jit.load() to load the quantized model
    torch.save(quantized_model.state_dict(), f"saved/{model_name}-torch-quantized.pth")


"""
To load back the model:

state_dict = torch.load(f"{model_name}-torch-quantized.pth")
constructor.load_state_dict(state_dict)
"""


@track_emissions
def quantize_tf(model, model_name):
    # Convert the model to TensorFlow Lite format
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    # converter.experimental_new_converter = True
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS
    ]

    # Optimize the model
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    # Save the state dictionary of the quantized model to disk
    with open(f"saved/{model_name}-tf-quantized.pkl", "wb") as f:
        pickle.dump(tflite_model, f)


"""
To load back the model:

with open(f"{model_name}-tf-quantized.pkl", "rb") as f:
    state_dict = pickle.load(f)
quantized_model = tf.lite.Interpreter(model_content=state_dict)
quantized_model.allocate_tensors()
"""
