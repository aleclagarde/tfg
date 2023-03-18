from transformers import AutoTokenizer, AutoModelForCausalLM
from codecarbon import track_emissions

tokenizer = AutoTokenizer.from_pretrained("huggingface/CodeBERTa-small-v1")
model = AutoModelForCausalLM.from_pretrained("huggingface/CodeBERTa-small-v1")


@track_emissions
def infer_codeberta(text):
    text = "def hello_world():"
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model(**inputs)

    return tokenizer.decode(outputs[0], skip_special_tokens=True)
