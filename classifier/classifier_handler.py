import copy
import logging
from abc import ABC

import torch
import transformers
from tokenizers import AddedToken
from transformers import RobertaTokenizerFast, XLMRobertaTokenizerFast

from ts.context import Context
from ts.torch_handler.distributed.base_deepspeed_handler import BaseDeepSpeedHandler
from torch.utils.data import DataLoader

from memory_dataset import ColumnAndTableClassifierDataset
from classifier_model import MyClassifier

logger = logging.getLogger(__name__)
logger.info("Transformers version %s", transformers.__version__)


class ClassifierHandler(BaseDeepSpeedHandler, ABC):
    """
    Transformers handler class for sequence, token classification and question answering.
    """

    def __init__(self):
        super(ClassifierHandler, self).__init__()
        self.max_length = None
        self.max_new_tokens = None
        self.tokenizer = None
        self.initialized = False
        self.use_contents = False
        self.add_fk_info = True

    def initialize(self, ctx: Context):
        """In this initialize function, the HF large model is loaded and
        partitioned using DeepSpeed.
        Args:
            ctx (context): It is a JSON Object containing information
            pertaining to the model artefacts parameters.
        """
        super().initialize(ctx)
        self.max_length = int(ctx.model_yaml_config["handler"]["max_length"])
        self.max_new_tokens = int(ctx.model_yaml_config["handler"]["max_new_tokens"])
        model_path = ctx.model_yaml_config["handler"]["model_path"]
        seed = int(ctx.model_yaml_config["handler"]["manual_seed"])
        self.use_contents = bool(ctx.model_yaml_config["handler"]["use_contents"])
        self.add_fk_info = bool(ctx.model_yaml_config["handler"]["add_fk_info"])
        torch.manual_seed(seed)

        logger.info("Model %s loading tokenizer", ctx.model_name)

        tokenizer_class = XLMRobertaTokenizerFast if "xlm" in model_path else RobertaTokenizerFast
        self.tokenizer = tokenizer_class.from_pretrained(
            model_path,
            add_prefix_space=True
        )
        self.tokenizer.add_tokens(AddedToken("[FK]"))

        self.model = MyClassifier(
            model_name_or_path=model_path,
            vocab_size=len(self.tokenizer),
            mode="test"
        )

        self.model.load_state_dict(torch.load(model_path + "/dense_classifier.pt", map_location=torch.device('cpu')))

        if torch.cuda.is_available():
            self.model = self.model.cuda()

        # ds_engine = get_ds_engine(self.model, ctx)
        # self.model = ds_engine.module
        logger.info("Model %s loaded successfully", ctx.model_name)
        self.initialized = True

    def preprocess(self, requests):
        input_texts = [data.get("data") or data.get("body") for data in requests]
        logger.info(type(input_texts[0]))
        logger.info(input_texts[0])
        return input_texts[0]

    def prepare_batch_inputs_and_labels(self, batch):
        batch_size = len(batch)

        batch_questions = [data[0] for data in batch]

        batch_table_names = [data[1] for data in batch]
        batch_column_infos = [data[2] for data in batch]

        batch_input_tokens, batch_column_info_ids, batch_table_name_ids, batch_column_number_in_each_table = [], [], [], []
        for batch_id in range(batch_size):
            input_tokens = [batch_questions[batch_id]]
            table_names_in_one_db = batch_table_names[batch_id]
            column_infos_in_one_db = batch_column_infos[batch_id]

            batch_column_number_in_each_table.append(
                [len(column_infos_in_one_table) for column_infos_in_one_table in column_infos_in_one_db])

            column_info_ids, table_name_ids = [], []

            for table_id, table_name in enumerate(table_names_in_one_db):
                input_tokens.append("|")
                input_tokens.append(table_name)
                table_name_ids.append(len(input_tokens) - 1)
                input_tokens.append(":")

                for column_info in column_infos_in_one_db[table_id]:
                    input_tokens.append(column_info)
                    column_info_ids.append(len(input_tokens) - 1)
                    input_tokens.append(",")

                input_tokens = input_tokens[:-1]

            batch_input_tokens.append(input_tokens)
            batch_column_info_ids.append(column_info_ids)
            batch_table_name_ids.append(table_name_ids)

        # notice: the trunction operation will discard some tables and columns that exceed the max length
        tokenized_inputs = self.tokenizer(
            batch_input_tokens,
            return_tensors="pt",
            is_split_into_words=True,
            padding="max_length",
            max_length=512,
            truncation=True
        )

        batch_aligned_question_ids, batch_aligned_column_info_ids, batch_aligned_table_name_ids = [], [], []

        # align batch_question_ids, batch_column_info_ids, and batch_table_name_ids after tokenizing
        for batch_id in range(batch_size):
            word_ids = tokenized_inputs.word_ids(batch_index=batch_id)

            aligned_question_ids, aligned_table_name_ids, aligned_column_info_ids = [], [], []

            # align question tokens
            for token_id, word_id in enumerate(word_ids):
                if word_id == 0:
                    aligned_question_ids.append(token_id)

            # align table names
            for t_id, table_name_id in enumerate(batch_table_name_ids[batch_id]):
                temp_list = []
                for token_id, word_id in enumerate(word_ids):
                    if table_name_id == word_id:
                        temp_list.append(token_id)
                # if the tokenizer doesn't discard current table name
                if len(temp_list) != 0:
                    aligned_table_name_ids.append(temp_list)

            # align column names
            for c_id, column_id in enumerate(batch_column_info_ids[batch_id]):
                temp_list = []
                for token_id, word_id in enumerate(word_ids):
                    if column_id == word_id:
                        temp_list.append(token_id)
                # if the tokenizer doesn't discard current column name
                if len(temp_list) != 0:
                    aligned_column_info_ids.append(temp_list)

            batch_aligned_question_ids.append(aligned_question_ids)
            batch_aligned_table_name_ids.append(aligned_table_name_ids)
            batch_aligned_column_info_ids.append(aligned_column_info_ids)

        encoder_input_ids = tokenized_inputs["input_ids"]
        encoder_input_attention_mask = tokenized_inputs["attention_mask"]

        # print("\n".join(tokenizer.batch_decode(encoder_input_ids, skip_special_tokens = True)))

        if torch.cuda.is_available():
            encoder_input_ids = encoder_input_ids.cuda()
            encoder_input_attention_mask = encoder_input_attention_mask.cuda()

        return encoder_input_ids, encoder_input_attention_mask, \
               batch_aligned_question_ids, batch_aligned_column_info_ids, \
               batch_aligned_table_name_ids, batch_column_number_in_each_table

    def inference(self, input_batch):
        table_pred_probs_for_auc, column_pred_probs_for_auc = [], []
        returned_table_pred_probs, returned_column_pred_probs = [], []
        original_input_batch = copy.deepcopy(input_batch)
        dataset = ColumnAndTableClassifierDataset(
            dataset=input_batch,
            use_contents=self.use_contents,
            add_fk_info=self.add_fk_info
        )

        dataloder = DataLoader(
            dataset,
            batch_size=32,
            shuffle=False,
            collate_fn=lambda x: x
        )
        for batch in dataloder:
            encoder_input_ids, encoder_input_attention_mask, batch_aligned_question_ids, \
            batch_aligned_column_info_ids, batch_aligned_table_name_ids, \
            batch_column_number_in_each_table = self.prepare_batch_inputs_and_labels(batch)

            with torch.no_grad():
                model_outputs = self.model(
                    encoder_input_ids,
                    encoder_input_attention_mask,
                    batch_aligned_question_ids,
                    batch_aligned_column_info_ids,
                    batch_aligned_table_name_ids,
                    batch_column_number_in_each_table
                )

            for batch_id, table_logits in enumerate(model_outputs["batch_table_name_cls_logits"]):
                table_pred_probs = torch.nn.functional.softmax(table_logits, dim=1)
                returned_table_pred_probs.append(table_pred_probs[:, 1].cpu().tolist())

                table_pred_probs_for_auc.extend(table_pred_probs[:, 1].cpu().tolist())

            for batch_id, column_logits in enumerate(model_outputs["batch_column_info_cls_logits"]):
                column_number_in_each_table = batch_column_number_in_each_table[batch_id]
                column_pred_probs = torch.nn.functional.softmax(column_logits, dim=1)
                returned_column_pred_probs.append([column_pred_probs[:, 1].cpu().tolist()[
                                                   sum(column_number_in_each_table[:table_id]):sum(
                                                       column_number_in_each_table[:table_id + 1])] \
                                                   for table_id in range(len(column_number_in_each_table))])

                column_pred_probs_for_auc.extend(column_pred_probs[:, 1].cpu().tolist())

        logger.info(original_input_batch)
        original_input_batch[0]["table_pred_probs"] = returned_table_pred_probs[0]
        original_input_batch[0]["column_pred_probs"] = returned_column_pred_probs[0]
        return original_input_batch

    def postprocess(self, inference_output):
        """Post Process Function converts the predicted response into Torchserve readable format.
        Args:
            inference_output (list): It contains the predicted response of the input text.
        Returns:
            (list): Returns a list of the Predictions and Explanations.
        """
        return inference_output
