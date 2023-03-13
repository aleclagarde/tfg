from transformers import AutoTokenizer, AutoModelWithLMHead
from codecarbon import track_emissions

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
model = AutoModelWithLMHead.from_pretrained("EleutherAI/gpt-neo-125M")


@track_emissions
def infer_codeparrot(text):
    input_ids = tokenizer.encode(text, return_tensors="pt")
    outputs = model.generate(input_ids, do_sample=True)

    return tokenizer.decode(outputs[0], skip_special_tokens=True)
