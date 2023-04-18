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

# Operating system
import os

# DataFrame modification
import pandas as pd

# Auxiliary functions
from optimize_utils import prune_torch, prune_tf, quantize_torch, quantize_tf, add_measurements, convert_tflite_to_tf


# Define the models constructors and their associated tokenizers
models = [
    {
        "name": "bert",
        "full_name": "bert-base-uncased",
        "constructor_tokenizer": BertTokenizer,
        "constructor_torch": BertModel,
        "constructor_tf": TFBertModel,
    },
    {
        "name": "gpt2",
        "full_name": "gpt2",
        "constructor_tokenizer": GPT2Tokenizer,
        "constructor_torch": GPT2Model,
        "constructor_tf": TFGPT2Model,
    },
    {
        "name": "t5",
        "full_name": "t5-base",
        "constructor_tokenizer": T5Tokenizer,
        "constructor_torch": T5ForConditionalGeneration,
        "constructor_tf": TFT5ForConditionalGeneration,
    },
    {
        "name": "vit",
        "full_name": "google/vit-base-patch16-224",
        "constructor_tokenizer": ViTFeatureExtractor,
        "constructor_torch": ViTForImageClassification,
        "constructor_tf": TFViTForImageClassification,
    },
    {
        "name": "clip",
        "full_name": "openai/clip-vit-large-patch14",
        "constructor_tokenizer": CLIPProcessor,
        "constructor_torch": CLIPModel,
        "constructor_tf": TFCLIPModel,
    },
    {
        "name": "segformer",
        "full_name": "nvidia/segformer-b0-finetuned-ade-512-512",
        "constructor_tokenizer": SegformerFeatureExtractor,
        "constructor_torch": SegformerForSemanticSegmentation,
        "constructor_tf": TFSegformerForSemanticSegmentation,
    },
    {
        "name": "codeparrot",
        "full_name": "codeparrot/codeparrot-small",
        "constructor_tokenizer": AutoTokenizer,
        "constructor_torch": AutoModelForCausalLM,
        "constructor_tf": TFAutoModelForCausalLM,
    },
    {
        "name": "codeberta",
        "full_name": "huggingface/CodeBERTa-small-v1",
        "constructor_tokenizer": AutoTokenizer,
        "constructor_torch": AutoModelForMaskedLM,
        "constructor_tf": TFAutoModelForMaskedLM,
    },
    {
        "name": "codegpt",
        "full_name": "microsoft/CodeGPT-small-py",
        "constructor_tokenizer": GPT2Tokenizer,
        "constructor_torch": GPT2Model,
        "constructor_tf": TFGPT2Model,
    },
]

number_of_measurements = 30
pruning_cf = 0.5

df = pd.DataFrame(columns=['timestamp', 'project_name', 'run_id', 'duration', 'emissions', 'emissions_rate',
                           'cpu_power', 'gpu_power', 'ram_power', 'cpu_energy', 'gpu_energy', 'ram_energy',
                           'energy_consumed', 'country_name', 'country_iso_code', 'region', 'cloud_provider',
                           'cloud_region', 'os', 'python_version', 'cpu_count', 'cpu_model', 'gpu_count', 'gpu_model',
                           'longitude', 'latitude', 'ram_total_size', 'tracking_mode', 'on_cloud', 'domain', 'model',
                           'framework', 'strategy', 'iteration'])


# Loop over the models and the coefficients and prune each model
for model_dict in models:
    # Get model name
    model_name = model_dict["name"]
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
        prune_torch(model_torch, model_name, pruning_cf)
    df = add_measurements(dataframe=df, number_of_measurements=number_of_measurements, model_name=model_name,
                          framework='torch', strategy='pruning')

    # Tensorflow
    for i in range(number_of_measurements):
        print("#############################################################################################")
        print(f"TF pruning: {model_name} with coefficient {pruning_cf}. Iteration: {i+1}")
        print("#############################################################################################")
        model_tf = model_dict["constructor_tf"].from_pretrained(f"saved/{model_name}-tf-baseline")
        prune_tf(model_tf, model_name, pruning_cf)
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
        quantize_torch(model_torch, model_name)
    df = add_measurements(dataframe=df, number_of_measurements=number_of_measurements, model_name=model_name,
                          framework='torch', strategy='quantization')

    # Tensorflow
    for i in range(number_of_measurements):
        print("#############################################################################################")
        print(f"TF quantization: {model_name}. Iteration: {i+1}")
        print("#############################################################################################")
        model_tf = model_dict["constructor_tf"].from_pretrained(f"saved/{model_name}-tf-baseline")
        tf_quantized_model = quantize_tf(model_tf)
    convert_tflite_to_tf(tf_quantized_model, model_name)
    df = add_measurements(dataframe=df, number_of_measurements=number_of_measurements, model_name=model_name,
                          framework='tf', strategy='quantization')

df.to_csv('optimization_results.csv')

# Remove the emissions file
os.remove('emissions.csv')
