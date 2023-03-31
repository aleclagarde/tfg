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
from optimize_utils import prune_torch, prune_tf, quantize_torch, quantize_tf

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


number_of_measurements = 1
pruning_cf = [0.2, 0.5, 0.8]

# Loop over the models and the coefficients and prune each model
for model_dict in models:
    # Get model name
    model_name = model_dict["name"]
    if model_name != 't5':
        continue
    # Initialize models
    model_torch = model_dict["constructor_torch"].from_pretrained(model_dict["full_name"])
    # Codeparrot is an exception
    if model_name == 'codeparrot':
        model_tf = model_dict["constructor_tf"].from_pretrained(model_dict["full_name"], from_pt=True)
    else:
        model_tf = model_dict["constructor_tf"].from_pretrained(model_dict["full_name"])
    # Initialize tokenizer
    tokenizer = model_dict["constructor_tokenizer"].from_pretrained(model_dict["full_name"])
    # Save baseline
    model_torch.save_pretrained(f"{model_name}-torch-baseline")
    model_tf.save_pretrained(f"{model_name}-tf-baseline")

    # PRUNING
    print(f"Pruning {model_name}")
    for cf in pruning_cf:
        # Loop to get emissions measurements
        for i in range(number_of_measurements):
            print(f"Torch pruning: {model_name} with coefficient {cf}. Iteration: {i}")
            model_torch = model_dict["constructor_torch"].from_pretrained(f"{model_name}-torch-baseline")
            prune_torch(model_torch, model_name, cf)
        # Loop to get emissions measurements
        for i in range(number_of_measurements):
            print(f"TF pruning: {model_name} with coefficient {cf}. Iteration: {i}")
            if model_name == 'codeparrot':
                model_tf = model_dict["constructor_tf"].from_pretrained(f"{model_name}-tf-baseline", from_pt=True)
            else:
                model_tf = model_dict["constructor_tf"].from_pretrained(f"{model_name}-tf-baseline")
            prune_tf(model_tf, model_name, cf)

    # QUANTIZATION
    print(f"Quantization for {model_name}")
    # Loop to get emissions measurements
    for i in range(number_of_measurements):
        print(f"Torch quantization: {model_name}. Iteration: {i}")
        model_torch = model_dict["constructor_torch"].from_pretrained(f"{model_name}-torch-baseline")
        quantize_torch(model_torch, model_name)

    # Loop to get emissions measurements
    for i in range(number_of_measurements):
        print(f"TF quantization: {model_name}. Iteration: {i}")
        if model_name == 'codeparrot':
            model_tf = model_dict["constructor_tf"].from_pretrained(f"{model_name}-tf-baseline", from_pt=True)
        else:
            model_tf = model_dict["constructor_tf"].from_pretrained(f"{model_name}-tf-baseline")
        quantize_tf(model_tf, model_name)

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
