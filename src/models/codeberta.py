from transformers import AutoTokenizer, AutoModelForCausalLM
from codecarbon import track_emissions

tokenizer = AutoTokenizer.from_pretrained("huggingface/CodeBERTa-small-v1")
model = AutoModelForCausalLM.from_pretrained("huggingface/CodeBERTa-small-v1")


@track_emissions
def infer_codeberta(text):
    input_ids = tokenizer.encode(text, return_tensors="pt")
    outputs = model.generate(input_ids, do_sample=True)

    return tokenizer.decode(outputs[0], skip_special_tokens=True)
