import json
from unittest import TestCase
from ..app import requests_prepare


class TestRequestsPrepare(TestCase):

    def test_count_dev(self):
        json_str = """
            [
          {
            "question": "How many singers do we have?",
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
          }
        ]
            """
        data = json.loads(json_str)
        print(data)
        print(type(data))
        print(requests_prepare.prepare_requests(data))
