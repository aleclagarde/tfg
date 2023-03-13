from transformers import BertTokenizer, BertForSequenceClassification
from codecarbon import track_emissions

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased")


@track_emissions
def infer_bert(text):
    input_ids = tokenizer(text, return_tensors="pt").input_ids
    outputs = model(input_ids)

    return outputs[0].argmax().item()
