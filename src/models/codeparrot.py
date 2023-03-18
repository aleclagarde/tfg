from transformers import AutoTokenizer, AutoModelWithLMHead
from codecarbon import track_emissions

tokenizer = AutoTokenizer.from_pretrained("codeparrot/codeparrot-small")
model = AutoModelWithLMHead.from_pretrained("codeparrot/codeparrot-small")


@track_emissions
def infer_codeparrot(text):
    text = "def hello_world():"
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model(**inputs)

    return tokenizer.decode(outputs[0], skip_special_tokens=True)
