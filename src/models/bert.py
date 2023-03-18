import torch
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics import accuracy_score
from codecarbon import track_emissions

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased")

texts = ["This movie was fantastic!",
         "I didn't enjoy this book at all.",
         "The food at this restaurant was amazing.",
         "I had a terrible experience at this hotel.",
         "The service at this store was excellent.",
         "I would highly recommend this product to anyone.",
         "This concert was the best I've ever been to!",
         "The customer service at this company was terrible.",
         "I thought this book was very well written.",
         "This product did not meet my expectations.",
         "The acting in this movie was superb.",
         "I would not recommend this restaurant to anyone.",
         "This hotel was a great place to stay.",
         "The service at this restaurant was very slow.",
         "This product was a complete waste of money.",
         "I absolutely loved this book!",
         "I had a wonderful experience at this hotel.",
         "The customer service at this store was exceptional.",
         "I would definitely buy this product again.",
         "This play was absolutely incredible!",
         "I had a terrible time at this restaurant.",
         "This hotel was not worth the price.",
         "The service at this company was very unprofessional.",
         "I found this book to be very boring.",
         "This product was exactly what I was looking for.",
         "The acting in this movie was terrible.",
         "I would never go back to this restaurant again.",
         "This hotel was not very clean.",
         "The customer service at this company was excellent.",
         "I thought this book was just okay.",
         "This product was not as advertised.",
         "I really enjoyed this concert!",
         "The food at this restaurant was terrible.",
         "This hotel was a disaster.",
         "The service at this store was very friendly.",
         "I regret buying this product.",
         "This play was very disappointing.",
         "I had a great time at this restaurant.",
         "This hotel was fantastic!",
         "The customer service at this company was terrible.",
         "I could not put this book down!",
         "This product exceeded my expectations.",
         "The acting in this movie was amazing.",
         "I would never recommend this restaurant to anyone.",
         "This hotel was a bit overpriced.",
         "The service at this company was very helpful.",
         "I really didn't like this book.",
         "This product was a total scam.",
         "I had an amazing time at this concert!",
         "The food at this restaurant was mediocre."]

labels = [1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1,
          0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0]


@track_emissions
def infer_bert(text):
    # encode text and labels for a validation set
    encoded_inputs = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
    labels_t = torch.tensor(labels)

    # make predictions on the validation set
    with torch.no_grad():
        outputs = model(**encoded_inputs, labels=labels_t)
        predictions = torch.argmax(outputs.logits, dim=1)

    # compute accuracy
    accuracy = accuracy_score(labels_t.numpy(), predictions.numpy())
    print(f"Accuracy: {accuracy}")
