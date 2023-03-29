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

# Define the models and their associated tokenizers
models = [
    {
        "name": "bert",
        "tokenizer": BertTokenizer.from_pretrained("bert-base-uncased"),
        "model_torch": BertModel.from_pretrained("bert-base-uncased"),
        "model_tf": TFBertModel.from_pretrained("bert-base-uncased"),
    },
    {
        "name": "gpt2",
        "tokenizer": GPT2Tokenizer.from_pretrained("gpt2"),
        "model_torch": GPT2Model.from_pretrained("gpt2"),
        "model_tf": TFGPT2Model.from_pretrained("gpt2"),
    },
    {
        "name": "t5",
        "tokenizer": T5Tokenizer.from_pretrained("t5-base"),
        "model_torch": T5ForConditionalGeneration.from_pretrained("t5-base"),
        "model_tf": TFT5ForConditionalGeneration.from_pretrained("t5-base"),
    },
    {
        "name": "vit",
        "tokenizer": ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224"),
        "model_torch": ViTForImageClassification.from_pretrained("google/vit-base-patch16-224"),
        "model_tf": TFViTForImageClassification.from_pretrained("google/vit-base-patch16-224"),
    },
    {
        "name": "clip",
        "tokenizer": CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14"),
        "model_torch": CLIPModel.from_pretrained("openai/clip-vit-large-patch14"),
        "model_tf": TFCLIPModel.from_pretrained("openai/clip-vit-large-patch14"),
    },
    {
        "name": "segformer",
        "tokenizer": SegformerFeatureExtractor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512"),
        "model_torch": SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512"),
        "model_tf": TFSegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512"),
    },
    {
        "name": "codeparrot",
        "tokenizer": AutoTokenizer.from_pretrained("codeparrot/codeparrot-small"),
        "model_torch": AutoModelForCausalLM.from_pretrained("codeparrot/codeparrot-small"),
        "model_tf": TFAutoModelForCausalLM.from_pretrained("codeparrot/codeparrot-small", from_pt=True),
    },
    {
        "name": "codeberta",
        "tokenizer": AutoTokenizer.from_pretrained("huggingface/CodeBERTa-small-v1"),
        "model_torch": AutoModelForMaskedLM.from_pretrained("huggingface/CodeBERTa-small-v1"),
        "model_tf": TFAutoModelForMaskedLM.from_pretrained("huggingface/CodeBERTa-small-v1"),
    },
    {
        "name": "codegpt",
        "tokenizer": GPT2Tokenizer.from_pretrained("microsoft/CodeGPT-small-py"),
        "model_torch": GPT2Model.from_pretrained("microsoft/CodeGPT-small-py"),
        "model_tf": TFGPT2Model.from_pretrained("microsoft/CodeGPT-small-py"),
    },
]


pruning_cf = [0.2, 0.5, 0.8]

# Loop over the models and the coefficients and prune each model
for model_dict in models:
    model_name = model_dict["name"]
    model_torch = model_dict["model_torch"]
    model_tf = model_dict["model_tf"]
    tokenizer = model_dict["tokenizer"]
    # Save baseline
    model_torch.save_pretrained(f"{model_name}-torch-baseline")
    model_tf.save_pretrained(f"{model_name}-tf-baseline")

    # PRUNING
    for cf in pruning_cf:
        # Loop to get 30 emissions measurements
        for i in range(30):
            prune_torch(model_torch, model_name, cf)
        # Loop to get 30 emissions measurements
        for i in range(30):
            prune_tf(model_tf, model_name, cf)

    # QUANTIZATION
    for i in range(30):
        quantize_torch(model_torch, model_name)

    for i in range(30):
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
