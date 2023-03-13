from transformers import GPT2Tokenizer, GPT2LMHeadModel
from codecarbon import track_emissions

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")


@track_emissions
def infer_gpt2(text):
    input_ids = tokenizer.encode(text, return_tensors="pt")
    outputs = model.generate(input_ids, do_sample=True)

    return tokenizer.decode(outputs[0], skip_special_tokens=True)
