# Transformers import, it imports the models and tokenizers for all 9 models
from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
    TFGPT2LMHeadModel,
    OPTForCausalLM,
    TFOPTForCausalLM,
    XLNetTokenizer,
    XLNetModel,
    TFXLNetModel,
    AutoImageProcessor,
    ResNetForImageClassification,
    TFResNetForImageClassification,
    ViTFeatureExtractor,
    ViTForImageClassification,
    TFViTForImageClassification,
    ConvNextFeatureExtractor,
    ConvNextForImageClassification,
    TFConvNextForImageClassification,
    AutoTokenizer,
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
    "xlnet": {
        "full_name": "xlnet-base-cased",
        "constructor_tokenizer": XLNetTokenizer,
        "constructor_torch": XLNetModel,
        "constructor_tf": TFXLNetModel,
    },
    "resnet": {
        "full_name": "microsoft/resnet-50",
        "constructor_tokenizer": AutoImageProcessor,
        "constructor_torch": ResNetForImageClassification,
        "constructor_tf": TFResNetForImageClassification,
    },
    "vit": {
        "full_name": "google/vit-base-patch16-224",
        "constructor_tokenizer": ViTFeatureExtractor,
        "constructor_torch": ViTForImageClassification,
        "constructor_tf": TFViTForImageClassification,
    },
    "convnext": {
        "full_name": "facebook/convnext-large-224",
        "constructor_tokenizer": ConvNextFeatureExtractor,
        "constructor_torch": ConvNextForImageClassification,
        "constructor_tf": TFConvNextForImageClassification,
    },
    "codeparrot": {
        "full_name": "codeparrot/codeparrot-small",
        "constructor_tokenizer": AutoTokenizer,
        "constructor_torch": AutoModelForCausalLM,
        "constructor_tf": TFAutoModelForCausalLM,
    },
    "codeberta": {
        "full_name": "huggingface/CodeBERTa-small-v1",
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
