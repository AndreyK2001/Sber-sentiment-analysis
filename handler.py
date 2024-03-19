import torch
import logging
from ts.torch_handler.base_handler import BaseHandler
from transformers import AutoModelForSequenceClassification, AutoTokenizer

logger = logging.getLogger(__name__)


class SentimentAnalysisHandler(BaseHandler):
    def __init__(self):
        self.initialized = False

    def initialize(self, ctx):
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "lxyuan/distilbert-base-multilingual-cased-sentiments-student"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            "lxyuan/distilbert-base-multilingual-cased-sentiments-student"
        )
        self.model.eval()
        self.initialized = True

    def preprocess(self, data):
        # Обработка списка входных данных
        texts = [
            item.get("data") if "data" in item else item.get("text", "")
            for item in data
        ]
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
            add_special_tokens=True,
        )
        return inputs

    def inference(self, inputs):
        # Выполнение предсказания
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs

    def postprocess(self, inference_output):
        # Преобразование выходных данных модели в формат ответа
        logits = inference_output.logits
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        labels = [
            self.model.config.id2label[i]
            for i in range(len(self.model.config.id2label))
        ]

        responses = []
        for prob in probabilities:
            response = [
                {"label": label, "score": score.item()}
                for label, score in zip(labels, prob)
            ]
            responses.append(response)
        return responses
