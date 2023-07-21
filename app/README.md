# RESDSQL Inference with TorchServe 

## Introduction

This directory contains all necessary files for building and launching RESDSQL models with TorchServe. There are two models that need to be launched to serve text2sql task requests.

## Environment Preparation

### Conda Environment

If you do not have a Conda environment ready, please prepare it with the following commands:
```
conda create -n torchserve python=3.9
conda activate torchserve
```

### TorchServe Environment

After creating the Conda environment, install necessary packages for TorchServe:
```
sudo apt update
sudo apt install openjdk-17-jdk
conda activate torchserve
git clone git@github.com:pytorch/serve.git
cd serve
python ./ts_scripts/install_dependencies.py --cuda={your cuda version eg. cu117}
pip install torchserve-nightly torch-model-archiver-nightly torch-workflow-archiver-nightly
export PATH={your anaconda path}/envs/torchserve/bin:$PATH
```

### Python Packages for RESDSQL

Install Python packages for RESDSQL by following the guides at https://github.com/RUCKBReasoning/RESDSQL

## Launching Models 

After preparing the environment, launch models with TorchServe:
```
bash ./build_start.sh
```

## Using the Text2SQL API
* Request URL: http://192.168.3.43:3100/predictions/text2sql

* Sample json data:
```
  [{
    "question": "How many singers do we have ?",
    "db_id": "concert_singer",
    "db_schema": [
      {
        "table_name_original": "stadium",
        "table_name": "stadium",
        "column_names": [
          "stadium id",
          "location",
          "name",
          "capacity",
          "highest",
          "lowest",
          "average"
        ],
        "column_names_original": [
          "stadium_id",
          "location",
          "name",
          "capacity",
          "highest",
          "lowest",
          "average"
        ],
        "column_types": [
          "number",
          "text",
          "text",
          "number",
          "number",
          "number",
          "number"
        ]
      },
      {
        "table_name_original": "singer",
        "table_name": "singer",
        "column_names": [
          "singer id",
          "name",
          "country",
          "song name",
          "song release year",
          "age",
          "is male"
        ],
        "column_names_original": [
          "singer_id",
          "name",
          "country",
          "song_name",
          "song_release_year",
          "age",
          "is_male"
        ],
        "column_types": [
          "number",
          "text",
          "text",
          "text",
          "text",
          "number",
          "others"
        ]
      },
      {
        "table_name_original": "concert",
        "table_name": "concert",
        "column_names": [
          "concert id",
          "concert name",
          "theme",
          "stadium id",
          "year"
        ],
        "column_names_original": [
          "concert_id",
          "concert_name",
          "theme",
          "stadium_id",
          "year"
        ],
        "column_types": [
          "number",
          "text",
          "text",
          "text",
          "text"
        ]
      },
      {
        "table_name_original": "singer_in_concert",
        "table_name": "singer in concert",
        "column_names": [
          "concert id",
          "singer id"
        ],
        "column_names_original": [
          "concert_id",
          "singer_id"
        ],
        "column_types": [
          "number",
          "text"
        ]
      }
    ],
    "pk": [
      {
        "table_name_original": "stadium",
        "column_name_original": "stadium_id"
      },
      {
        "table_name_original": "singer",
        "column_name_original": "singer_id"
      },
      {
        "table_name_original": "concert",
        "column_name_original": "concert_id"
      },
      {
        "table_name_original": "singer_in_concert",
        "column_name_original": "concert_id"
      }
    ],
    "fk": [
      {
        "source_table_name_original": "concert",
        "source_column_name_original": "stadium_id",
        "target_table_name_original": "stadium",
        "target_column_name_original": "stadium_id"
      },
      {
        "source_table_name_original": "singer_in_concert",
        "source_column_name_original": "singer_id",
        "target_table_name_original": "singer",
        "target_column_name_original": "singer_id"
      },
      {
        "source_table_name_original": "singer_in_concert",
        "source_column_name_original": "concert_id",
        "target_table_name_original": "concert",
        "target_column_name_original": "concert_id"
      }
    ]
  }]
```