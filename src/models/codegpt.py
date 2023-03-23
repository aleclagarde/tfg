from transformers import GPT2Tokenizer, GPT2Model
from codecarbon import track_emissions

tokenizer = GPT2Tokenizer.from_pretrained("microsoft/CodeGPT-small-py")
model = GPT2Model.from_pretrained("microsoft/CodeGPT-small-py")


@track_emissions
def infer_codegen(text):
    text = "def hello_world():"
    input_ids = tokenizer(text, return_tensors="pt").input_ids

    generated_ids = model.generate(input_ids, max_length=128)
    print(tokenizer.decode(generated_ids[0], skip_special_tokens=True))
