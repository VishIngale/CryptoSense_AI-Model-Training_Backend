from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

# Load FinBERT model
MODEL_NAME = "ProsusAI/finbert"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)


def predict_sentiment(text: str):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

    outputs = model(**inputs)
    probabilities = F.softmax(outputs.logits, dim=1)

    confidence, predicted_class = torch.max(probabilities, dim=1)

    labels = ["negative", "neutral", "positive"]
    sentiment = labels[predicted_class.item()]

    return {
        "sentiment": sentiment,
        "confidence": round(confidence.item() * 100, 2)
    }
