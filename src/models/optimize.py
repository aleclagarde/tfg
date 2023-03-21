from transformers import (
    BertTokenizer,
    BertModel,
    GPT2Tokenizer,
    GPT2LMHeadModel,
    T5Tokenizer,
    T5ForConditionalGeneration,
    ViTFeatureExtractor,
    ViTForImageClassification,
    OwlViTProcessor,
    OwlViTForObjectDetection,
    MaskFormerForInstanceSegmentation,
    MaskFormerFeatureExtractor,
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
)
import torch
from torch.nn.utils import prune
import tensorflow as tf
import tensorflow_model_optimization as tfmot

# from optimize_utils import prune_torch

# Define the models and their associated tokenizers
models = [
    {
        "name": "bert",
        "tokenizer": BertTokenizer.from_pretrained("bert-base-uncased"),
        "model": BertModel.from_pretrained("bert-base-uncased"),
    },
    {
        "name": "gpt2",
        "tokenizer": GPT2Tokenizer.from_pretrained("gpt2"),
        "model": GPT2LMHeadModel.from_pretrained("gpt2"),
    },
    {
        "name": "t5",
        "tokenizer": T5Tokenizer.from_pretrained("t5-small"),
        "model": T5ForConditionalGeneration.from_pretrained("t5-small"),
    },
    {
        "name": "vit",
        "tokenizer": ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224"),
        "model": ViTForImageClassification.from_pretrained("google/vit-base-patch16-224"),
    },
    {
        "name": "owlvit",
        "tokenizer": OwlViTProcessor.from_pretrained("google/owlvit-base-patch32"),
        "model": OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32"),
    },
    {
        "name": "maskformer",
        "tokenizer": MaskFormerFeatureExtractor.from_pretrained("facebook/maskformer-swin-small-coco"),
        "model": MaskFormerForInstanceSegmentation.from_pretrained("facebook/maskformer-swin-small-coco"),
    },
    {
        "name": "codeparrot",
        "tokenizer": AutoTokenizer.from_pretrained("codeparrot/codeparrot-small"),
        "model": AutoModelForCausalLM.from_pretrained("codeparrot/codeparrot-small"),
    },
    {
        "name": "codeberta",
        "tokenizer": AutoTokenizer.from_pretrained("huggingface/CodeBERTa-small-v1"),
        "model": AutoModelForMaskedLM.from_pretrained("huggingface/CodeBERTa-small-v1"),
    },
    {
        "name": "codegen",
        "tokenizer": AutoTokenizer.from_pretrained("Salesforce/codegen-350M-mono"),
        "model": AutoModelForCausalLM.from_pretrained("Salesforce/codegen-350M-mono"),
    },
]


pruning_cf = [0.2, 0.5, 0.8]
pruning_methods = [
    ("torch", torch.nn.utils.prune.random_unstructured),
    ("tf", tfmot.sparsity.keras.prune_low_magnitude),
]

# Loop over the models and the coefficients and prune each model
for model_dict in models:
    model_name = model_dict["name"]
    model = model_dict["model"]
    tokenizer = model_dict["tokenizer"]
    for method_name, method_func in pruning_methods:
        for cf in pruning_cf:
            # Instantiate the pruning method
            if method_name == "torch":
                pruning_method = method_func
                # Get a list of all the modules in the model
                modules = list(model.modules())

                # Loop through each module and prune its parameters if they exist
                for module in modules:
                    if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
                        pruning_method(module, name='weight', amount=cf)

                # model = prune_torch(model_name, model, pruning_method, cf)
            elif method_name == "tf":
                continue
                pruning_method = method_func
                # Prune the model with the given coefficient
                model = pruning_method(
                    model,
                    sparsity=cf,
                )

                # `prune_low_magnitude` requires a recompile.
                model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                              metrics=['accuracy'])
            # Save the pruned model to disk
            model.save_pretrained(f"{model_name}-{method_name}-pruned-{cf}")

    # Optionally, you can reload the original model from disk
    # model_dict["model"] = model.from_pretrained(f"{model_name}-{method_name}-pruned-{pruning_cf[0]}")
    # model_dict["tokenizer"] = tokenizer.from_pretrained(f"{model_name}-{method_name}-pruned-{pruning_cf[0]}")

"""
def prune_t5(prune_pct):
    # Prune the model
    if prune_pct > 0:
        for name, param in model.named_parameters():
            if "embedding" in name:
                l1_unstructured(param, name='weight', amount=prune_pct)
    return model
"""
