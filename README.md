# Анализ тональности

Репозиторий реализует предсказание тональности текста при помощи модели ```lxyuan/distilbert-base-multilingual-cased-sentiments-student``` из huggingface. Так как сама модель - трансформет, то на вход можно подавать текст на (практически) любом языке и ожидать (сколько-нибудь) адекватный результат.

В репозитории находится серверная часть, реализующая логику предсказания и пользовательский интерфейс, который получает от пользователя текст и выдаёт вероятности, что текст относится к каждому из классов __positive__, __neutral__, __negative__.

Запуск протестирован на процессоре M1. 

## Структура

```
├── Dockerfile
├── LICENSE
├── README.md
├── config.properties
├── docker-compose.yml
├── handler.py - описание пайплайна предсказания
├── load_weights.py - сохранение весов модели 
├── prometheus.yml
├── requirements.txt
└── sentiment.html
```

## Запуск

Запуск серверной части:

```
docker-compose -f docker-compose.yml up
```

Запуск интерфейса:
```
sentiment.html
```