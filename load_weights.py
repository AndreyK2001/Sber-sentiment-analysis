from transformers import AutoModelForSequenceClassification
import torch

# Загружаем модель и токенайзер
model_name = "lxyuan/distilbert-base-multilingual-cased-sentiments-student"
model = AutoModelForSequenceClassification.from_pretrained(model_name)

torch.save(model.state_dict(), "/sentiment_analysis/model/pytorch_model.pth")
