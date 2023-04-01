from transformers import GPT2Tokenizer, GPT2LMHeadModel
from codecarbon import track_emissions

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")


@track_emissions
def infer_gpt2(text):
    encoded_input = tokenizer(text, return_tensors='pt')
    output = model(**encoded_input)

    return output
