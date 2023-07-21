#!/bin/bash

torch-model-archiver --model-name text2sql \
  --version 1.0 \
  --extra-files ds-config.json -r requirements.txt \
  --config-file model-config.yaml \
  --handler text2sql_handler.py \
  --export-path /date/bruce/workspace/RESDSQL-Coding/model-server/model-store \
  --force

torch-model-archiver --model-name text2sql --version 1.0 --extra-files ds-config.json -r requirements.txt --config-file model-config.yaml --handler text2sql_handler.py --export-path /date/bruce/workspace/RESDSQL-Coding/model-server/model-store --force

torchserve --start --ncs --ts-config config.properties

torchserve --start --ncs --model-store model_store --ts-config ./config.properties

torchserve --start --ncs --model-store model-store --models opt.tar.gz --ts-config ./config.properties



docker build -t resdsql-v1 .

docker run  --rm --user "$(id -u):$(id -g)" -it --gpus '"device=0,1,2,3"' -p 3100:3100 -p 3101:3101 \
  -v $(pwd)/model-server:/home/model-server \
  -v $(pwd)/logs:/home/model-server/logs \
  pytorch/torchserve:latest-gpu

torch-model-archiver --model-name opt --version 1.0 --handler custom_handler.py --extra-files ds-config.json -r requirements.txt --config-file opt/model-config.yaml  --archive-format tgz