"""
Get model objects
=================

.. module:: get_model_objects
   :platform: Linux
   :synopsis: Get model objects script.

.. module_author:: Alec Lagarde Teixidó <aleclagarde@gmail.com>

This script contains the initialization of the objects of each model and a function that returns a dictionary for a
selected model.

.. autosummary::
   :toctree: generated/

   get_model_objects
"""

# Transformers import, it imports the models and tokenizers for all 9 models
from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
    TFGPT2LMHeadModel,
    OPTForCausalLM,
    TFOPTForCausalLM,
    AutoTokenizer,
    AutoImageProcessor,
    ResNetForImageClassification,
    TFResNetForImageClassification,
    AutoFeatureExtractor,
    RegNetForImageClassification,
    TFRegNetForImageClassification,
    AutoModelForCausalLM,
    TFAutoModelForCausalLM,
)

# Define the models constructors and their associated tokenizers
models = {
    "gpt2": {
        "full_name": "gpt2",
        "constructor_tokenizer": GPT2Tokenizer,
        "constructor_torch": GPT2LMHeadModel,
        "constructor_tf": TFGPT2LMHeadModel,
    },
    "opt": {
        "full_name": "facebook/opt-125m",
        "constructor_tokenizer": AutoTokenizer,
        "constructor_torch": OPTForCausalLM,
        "constructor_tf": TFOPTForCausalLM,
    },
    "resnet": {
        "full_name": "microsoft/resnet-50",
        "constructor_tokenizer": AutoImageProcessor,
        "constructor_torch": ResNetForImageClassification,
        "constructor_tf": TFResNetForImageClassification,
    },
    "regnet": {
        "full_name": "facebook/regnet-y-008",
        "constructor_tokenizer": AutoFeatureExtractor,
        "constructor_torch": RegNetForImageClassification,
        "constructor_tf": TFRegNetForImageClassification,
    },
    "codeparrot": {
        "full_name": "codeparrot/codeparrot-small",
        "constructor_tokenizer": AutoTokenizer,
        "constructor_torch": AutoModelForCausalLM,
        "constructor_tf": TFAutoModelForCausalLM,
    },
    "codegpt": {
        "full_name": "microsoft/CodeGPT-small-py",
        "constructor_tokenizer": GPT2Tokenizer,
        "constructor_torch": GPT2LMHeadModel,
        "constructor_tf": TFGPT2LMHeadModel,
    }
}


def get_model_objects(model_name: str) -> dict:
    """
    Function that returns the model dictionary with its information and objects given the model name.

    :param model_name: Model short name.
    :return: Model dictionary.
    """
    return models[model_name]
