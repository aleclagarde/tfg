# Transformers import, it imports the models and tokenizers for all 9 models
from transformers import (
    BertTokenizer,
    BertModel,
    TFBertModel,
    GPT2Tokenizer,
    GPT2Model,
    TFGPT2Model,
    T5Tokenizer,
    T5ForConditionalGeneration,
    TFT5ForConditionalGeneration,
    ViTFeatureExtractor,
    ViTForImageClassification,
    TFViTForImageClassification,
    CLIPProcessor,
    CLIPModel,
    TFCLIPModel,
    SegformerFeatureExtractor,
    SegformerForSemanticSegmentation,
    TFSegformerForSemanticSegmentation,
    AutoTokenizer,
    AutoModelForCausalLM,
    TFAutoModelForCausalLM,
    AutoModelForMaskedLM,
    TFAutoModelForMaskedLM,
)

# Define the models constructors and their associated tokenizers
models = {
    "bert": {
        "full_name": "bert-base-uncased",
        "constructor_tokenizer": BertTokenizer,
        "constructor_torch": BertModel,
        "constructor_tf": TFBertModel,
    },
    "gpt2": {
        "full_name": "gpt2",
        "constructor_tokenizer": GPT2Tokenizer,
        "constructor_torch": GPT2Model,
        "constructor_tf": TFGPT2Model,
    },
    "t5": {
        "full_name": "t5-base",
        "constructor_tokenizer": T5Tokenizer,
        "constructor_torch": T5ForConditionalGeneration,
        "constructor_tf": TFT5ForConditionalGeneration,
    },
    "vit": {
        "full_name": "google/vit-base-patch16-224",
        "constructor_tokenizer": ViTFeatureExtractor,
        "constructor_torch": ViTForImageClassification,
        "constructor_tf": TFViTForImageClassification,
    },
    "clip": {
        "full_name": "openai/clip-vit-large-patch14",
        "constructor_tokenizer": CLIPProcessor,
        "constructor_torch": CLIPModel,
        "constructor_tf": TFCLIPModel,
    },
    "segformer": {
        "full_name": "nvidia/segformer-b0-finetuned-ade-512-512",
        "constructor_tokenizer": SegformerFeatureExtractor,
        "constructor_torch": SegformerForSemanticSegmentation,
        "constructor_tf": TFSegformerForSemanticSegmentation,
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
        "constructor_torch": AutoModelForMaskedLM,
        "constructor_tf": TFAutoModelForMaskedLM,
    },
    "codegpt": {
        "full_name": "microsoft/CodeGPT-small-py",
        "constructor_tokenizer": GPT2Tokenizer,
        "constructor_torch": GPT2Model,
        "constructor_tf": TFGPT2Model,
    }
}


def get_model_objects(model_name: str) -> dict:
    """
    Function that returns the model dictionary with its information and objects given the model name.

    :param model_name: Model short name.
    :return: Model dictionary.
    """
    return models[model_name]
