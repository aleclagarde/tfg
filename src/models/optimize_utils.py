import transformers
import torch
import tensorflow as tf
from codecarbon import track_emissions


@track_emissions
def prune_torch(model, model_name, pruning_method, modules, cf):
    for module in modules:
        # Loop through each module and prune its parameters
        if isinstance(module, torch.nn.modules.linear.Linear) or \
                isinstance(module, transformers.pytorch_utils.Conv1D):
            pruning_method(module, name='weight', amount=cf)

    # Save the pruned model to disk
    model.save_pretrained(f"{model_name}-torch-pruned-{cf}")


@track_emissions
def prune_tf(model, model_name, pruning_method, modules, cf):
    # Loop through each module and prune its parameters
    for module in modules:
        if isinstance(module, tf.keras.models.Sequential):
            # Prune the model with the given coefficient
            pruning_method(model, sparsity=cf)

    # `prune_low_magnitude` requires a recompile.
    model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    # Save the pruned model to disk
    model.save_pretrained(f"{model_name}-tf-pruned-{cf}")
