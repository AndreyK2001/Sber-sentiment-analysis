FROM pytorch/torchserve:latest

ENV PROJECT_ROOT /sentiment_analysis
WORKDIR $PROJECT_ROOT

COPY requirements.txt  $PROJECT_ROOT

# install dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# copy config for torchserve
COPY config.properties handler.py load_weights.py $PROJECT_ROOT

RUN mkdir -p $PROJECT_ROOT/model
RUN python load_weights.py

RUN mkdir -p $PROJECT_ROOT/model-store

# archive model from model_dir -> model.mar
RUN torch-model-archiver --model-name sentiment_analysis  \
    --version 1.0  \
    --model-file $PROJECT_ROOT/model/pytorch_model.pth  \
    --handler $PROJECT_ROOT/handler.py  \
    --export-path model-store