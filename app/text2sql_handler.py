import json
import logging
from abc import ABC

import requests
import torch
import transformers
from tokenizers import AddedToken
from transformers import T5TokenizerFast, MT5ForConditionalGeneration, T5ForConditionalGeneration

from ts.context import Context
from ts.handler_utils.distributed.deepspeed import get_ds_engine
from ts.torch_handler.distributed.base_deepspeed_handler import BaseDeepSpeedHandler
from text2sql_requests_generator import generate_eval_ranked_dataset

logger = logging.getLogger(__name__)
logger.info("Transformers version %s", transformers.__version__)


class Text2SqlHandler(BaseDeepSpeedHandler, ABC):
    """
    Transformers handler class for sequence, token classification and question answering.
    """

    def __init__(self):
        super(Text2SqlHandler, self).__init__()
        self.max_length = None
        self.max_new_tokens = None
        self.tokenizer = None
        self.initialized = False
        self.num_beams = None
        self.num_return_sequences = None
        self.use_contents = False
        self.add_fk_info = True
        self.topk_table_num = None
        self.topk_column_num = None
        self.classifier_url = None

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
        model_name = ctx.model_yaml_config["handler"]["model_name"]
        model_path = ctx.model_yaml_config["handler"]["model_path"]
        seed = int(ctx.model_yaml_config["handler"]["manual_seed"])
        self.num_beams = int(ctx.model_yaml_config["handler"]["num_beams"])
        self.num_return_sequences = int(ctx.model_yaml_config["handler"]["num_return_sequences"])
        self.use_contents = bool(ctx.model_yaml_config["handler"]["use_contents"])
        self.add_fk_info = bool(ctx.model_yaml_config["handler"]["add_fk_info"])
        self.topk_table_num = int(ctx.model_yaml_config["handler"]["topk_table_num"])
        self.topk_column_num = int(ctx.model_yaml_config["handler"]["topk_column_num"])
        self.classifier_url = ctx.model_yaml_config["handler"]["classifier_url"]
        torch.manual_seed(seed)

        logger.info("Model %s loading tokenizer", ctx.model_name)
        logger.info("Model %s loading tokenizer", model_path)

        self.tokenizer = T5TokenizerFast.from_pretrained(
            model_path,
            add_prefix_space=True
        )
        if isinstance(self.tokenizer, T5TokenizerFast):
            self.tokenizer.add_tokens([AddedToken(" <="), AddedToken(" <")])

        model_class = MT5ForConditionalGeneration if "mt5" in model_name else T5ForConditionalGeneration
        with torch.device("meta"):
            self.model = model_class.from_pretrained(model_path, torch_dtype=torch.float16)
        self.model = self.model.eval()

        ds_engine = get_ds_engine(self.model, ctx)
        self.model = ds_engine.module
        logger.info("Model %s loaded successfully", ctx.model_name)
        self.initialized = True

    def post_classifier(self, input_text):
        headers = {'Content-type': 'application/json'}
        response = requests.post(self.classifier_url, data=json.dumps(input_text), headers=headers)
        return response.text

    def preprocess(self, requests):
        """
        Basic text preprocessing, based on the user's choice of application mode.
        Args:
            requests (list): A list of dictionaries with a "data" or "body" field, each
                            containing the input text to be processed.
        Returns:
            tuple: A tuple with two tensors: the batch of input ids and the batch of
                attention masks.
        """
        input_texts = [data.get("data") or data.get("body") for data in requests]
        input_ids_batch, attention_mask_batch = [], []
        input_ids_batch = []
        logger.info(self.device)
        for input_text in input_texts:
            logger.info('current device: ' + str(self.device))

            classified_input = self.post_classifier(input_text)
            logger.info('classified output: ' + classified_input)
            text2sql_request = generate_eval_ranked_dataset(json.loads(classified_input), self.use_contents,
                                                            self.add_fk_info, self.topk_table_num, self.topk_column_num)
            logger.info('generated output: ' + text2sql_request)
            input_ids, attention_mask = self.encode_input_text(text2sql_request)
            input_ids_batch.append(input_ids)
            attention_mask_batch.append(attention_mask)

        input_ids_batch = torch.cat(input_ids_batch, dim=0).to(self.device)
        attention_mask_batch = torch.cat(attention_mask_batch, dim=0).to(self.device)
        return input_ids_batch, attention_mask_batch

    def encode_input_text(self, input_text):
        """
        Encodes a single input text using the tokenizer.
        Args:
            input_text (str): The input text to be encoded.
        Returns:
            tuple: A tuple with two tensors: the encoded input ids and the attention mask.
        """

        if isinstance(input_text, (bytes, bytearray)):
            input_text = input_text.decode("utf-8")
        logger.info("Received text: '%s'", input_text)
        inputs = self.tokenizer.encode_plus(
            input_text,
            max_length=self.max_length,
            padding=True,
            add_special_tokens=True,
            return_tensors="pt",
            truncation=True,
        )
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        return input_ids, attention_mask

    def inference(self, input_batch):
        """
        Predicts the class (or classes) of the received text using the serialized transformers
        checkpoint.
        Args:
            input_batch (tuple): A tuple with two tensors: the batch of input ids and the batch
                                of attention masks, as returned by the preprocess function.
        Returns:
            list: A list of strings with the predicted values for each input text in the batch.
        """

        input_ids_batch, attention_mask_batch = input_batch
        if torch.cuda.is_available():
            input_ids_batch = input_ids_batch.cuda()
            attention_mask_batch = attention_mask_batch.cuda()

        outputs = self.model.generate(
            input_ids_batch,
            attention_mask=attention_mask_batch,
            max_length=self.max_new_tokens,
        )

        inferences = self.tokenizer.batch_decode(
            outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        logger.info("Generated text: %s", inferences)

        return [s.split('|')[1].strip() for s in inferences if len(s.split('|')) > 1]

    def postprocess(self, inference_output):
        """Post Process Function converts the predicted response into Torchserve readable format.
        Args:
            inference_output (list): It contains the predicted response of the input text.
        Returns:
            (list): Returns a list of the Predictions and Explanations.
        """
        return inference_output
