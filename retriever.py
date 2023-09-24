import os 
import numpy as np
import torch
from cfg import RetrieverCFG
from threading import Thread
from torch import Tensor
from transformers import AutoModel 
from transformers import AutoTokenizer

class Retriever:
    def __init__(self):
        dir(Retriever)
        self.tokenizer = self.load_tokenizer()
        self.retiever = self.load_retriever()
        self._question = None
        self._input_text = None

    @property
    def input_text(self):
        return self._input_text
    
    @input_text.setter
    def input_text(self, value):
        self._input_text =value

    @input_text.getter
    def input_text(self):
        return self._input_text

    @input_text.deleter
    def input_text(self):
        self._input_text = None

    def average_pool(last_hidden_states: Tensor,
                    attention_mask: Tensor) -> Tensor:
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    def load_tokenizer(self):
        return AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path = RetrieverCFG.model_name, 
            local_files_only=True
            )

    def load_retriever(self):
        return AutoModel.from_pretrained(
            pretrained_model_name_or_path = RetrieverCFG.model_name,
            local_files_only = True
        )

    def embedd_normalize(self, embeddings: torch.Tensor):
        return F.normalize(embeddings)

    def check_embed(self, embed1, embed2):
        return torch.all(embed1 == embed2)

    def proces_thread(self, question: str):
        if self.parallell:
            process = threading.Thread(target = self.process_once, name = 'process_once')
            process.daemon = True
            process.start()

    def process_once(self):
        batch_dict = self.tokenizer(input_text, max_length = RetrieverCFG.max_length, 
            padding = RetrieverCFG.padding, 
            truncation = RetrieverCFG.truncation, 
            return_tensors = RetrieverCFG.return_tensors)

        outputs = self.retriever(**batch_dict)
        embeddings = self.average_pool(
            last_hidden_states = outputs.last_hidden_state,
            attention_mask = batch_dict['attention_mask']
        )
        embeddings = self.embedd_normalize(embeddings)
        return embeddings

    def batch_processing(self):
        ...
