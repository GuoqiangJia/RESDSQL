import json
import numpy as np


def lista_contains_listb(lista, listb):
    for b in listb:
        if b not in lista:
            return 0

    return 1


def prepare_input_and_output(use_contents, add_fk_info, ranked_data):
    question = ranked_data["question"]

    schema_sequence = ""
    for table_id in range(len(ranked_data["db_schema"])):
        table_name_original = ranked_data["db_schema"][table_id]["table_name_original"]
        # add table name
        schema_sequence += " | " + table_name_original + " : "

        column_info_list = []
        for column_id in range(len(ranked_data["db_schema"][table_id]["column_names_original"])):
            # extract column name
            column_name_original = ranked_data["db_schema"][table_id]["column_names_original"][column_id]

            column_info = table_name_original + "." + column_name_original
            if use_contents:
                db_contents = ranked_data["db_schema"][table_id]["db_contents"][column_id]
                if len(db_contents) != 0:
                    column_contents = " , ".join(db_contents)
                    column_info = column_info + " ( " + column_contents + " ) "

            column_info_list.append(column_info)

        # add column names
        schema_sequence += " , ".join(column_info_list)

    if add_fk_info:
        for fk in ranked_data["fk"]:
            schema_sequence += " | " + fk["source_table_name_original"] + "." + fk["source_column_name_original"] + \
                               " = " + fk["target_table_name_original"] + "." + fk["target_column_name_original"]

    # remove additional spaces in the schema sequence
    while "  " in schema_sequence:
        schema_sequence = schema_sequence.replace("  ", " ")

    # input_sequence = question + schema sequence
    input_sequence = question + schema_sequence
    return input_sequence, " | "


def generate_eval_ranked_dataset(test_with_probs, use_contents, add_fk_info, topk_table_num, topk_column_num):
    dataset = test_with_probs

    output_dataset = []
    ranked_data = dict()
    ranked_data["question"] = dataset["question"]
    ranked_data["db_id"] = dataset["db_id"]
    ranked_data["db_schema"] = []

    table_pred_probs = list(map(lambda x: round(x, 4), dataset["table_pred_probs"]))
    # find ids of tables that have top-k probability
    topk_table_ids = np.argsort(-np.array(table_pred_probs), kind="stable")[:topk_table_num].tolist()

    # record top-k1 tables and top-k2 columns for each table
    for table_id in topk_table_ids:
        new_table_info = dict()
        new_table_info["table_name_original"] = dataset["db_schema"][table_id]["table_name_original"]
        column_pred_probs = list(map(lambda x: round(x, 2), dataset["column_pred_probs"][table_id]))
        topk_column_ids = np.argsort(-np.array(column_pred_probs), kind="stable")[:topk_column_num].tolist()

        new_table_info["column_names_original"] = [dataset["db_schema"][table_id]["column_names_original"][column_id]
                                                   for column_id in topk_column_ids]
        if use_contents:
            new_table_info["db_contents"] = [dataset["db_schema"][table_id]["db_contents"][column_id] for column_id in
                                             topk_column_ids]

        ranked_data["db_schema"].append(new_table_info)

    # record foreign keys among selected tables
    table_names_original = [table["table_name_original"] for table in dataset["db_schema"]]
    needed_fks = []
    for fk in dataset["fk"]:
        source_table_id = table_names_original.index(fk["source_table_name_original"])
        target_table_id = table_names_original.index(fk["target_table_name_original"])
        if source_table_id in topk_table_ids and target_table_id in topk_table_ids:
            needed_fks.append(fk)
    ranked_data["fk"] = needed_fks

    input_sequence, output_sequence = prepare_input_and_output(use_contents, add_fk_info, ranked_data)

    # record table_name_original.column_name_original for subsequent correction function during inference
    tc_original = []
    for table in ranked_data["db_schema"]:
        for column_name_original in table["column_names_original"] + ["*"]:
            tc_original.append(table["table_name_original"] + "." + column_name_original)

    output_dataset.append(
        {
            "db_id": dataset["db_id"],
            "input_sequence": input_sequence,
            "output_sequence": output_sequence,
            "tc_original": tc_original
        }
    )

    return json.dumps(output_dataset, indent=2, ensure_ascii=False)
