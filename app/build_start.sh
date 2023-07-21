#!/bin/bash

source /date/bruce/anaconda3/etc/profile.d/conda.sh
conda activate torchserve
python --version
git pull
echo "conda activate torchserve"

cd /date/bruce/workspace/RESDSQL-Coding/classifier
torch-model-archiver --model-name classifier --version 1.0 --handler classifier_handler.py --extra-files ./classifier_model.py,./memory_dataset.py,./ds-config.json --config-file model-config.yaml --handler classifier_handler.py --export-path /date/bruce/workspace/RESDSQL-Coding/model-server/model-store --force
echo "classifier model archiver done"

cd /date/bruce/workspace/RESDSQL-Coding/app
torch-model-archiver --model-name text2sql --version 1.0 --extra-files text2sql_requests_generator.py,ds-config.json -r requirements.txt --config-file model-config.yaml --handler text2sql_handler.py --export-path /date/bruce/workspace/RESDSQL-Coding/model-server/model-store --force
echo "text2sql model archiver done"

echo "stopping torchserve"
torchserve --stop

echo "starting torchserve"
torchserve --start --ncs --ts-config ./config.properties
