import torch
import numpy as np 
import os 
import re 
from retriever import Retriever
from transformers import MarianMTModel, MarianTokenizer
from typing import List, Dict, Union
from typing import Any, TypeVar
from cfg import RetrieverCFG, TranslatorCFG

class Translator:
    def __init__(self):
        self.config = TranslatorCFG
        self._input_text = None
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.tokenizer = self.load_tokenizer()
        self.retriever = self.load_retriever()

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

    def load_tokenizer(self) -> Any:
        return MarianTokenizer.from_pretrained(
            pretrained_model_name_or_path = TranslatorCFG.model_name, 
            local_files_only = TranslatorCFG.local_files_only 
        )

    def load_retriever(self) -> Any:
        return MarianMTModel.from_pretrained(
            pretrained_model_name_or_path = TranslatorCFG.model_name, 
            local_files_only = TranslatorCFG.local_files_only
        ).to(self.device).eval()

    def tokenize(self, input_text: Union[str, List]):
        return self.tokenizer(input_text, 
            padding = TranslatorCFG.padding, 
            truncation = TranslatorCFG.truncation, 
            max_length = TranslatorCFG.max_length, 
            return_tensors = TranslatorCFG.return_tensors ) 

    def translate(self, tokenized: torch.Tensor) -> Union[torch.Tensor, List]:
        return self.retriever.generate(tokenizer) 

    def process_once(self):
        ...

    def batch_chunk(self, input_base: List) -> torch.Tensor:
        """
        Return sliced input based, base on chunk_size 
        provided in RetrieverCFG.chunk_size argument 
        
        Args:
            input_base: List    
                Consist knowledge base, stored in dictionary. 
                This is a chunked one.
            We just want to extending to chunk_size, to calculate more embeddings at once.
        Returns:
            List[List]
                List of lists which consist of nested lists of lists.
        """
        emb_len = len(input_base)
        chunk_size = TranslatorCFG.chunk_size
        n_ixes = int(emb_len/TranslatorCFG.chunk_size)
        return [input_base[(ix*chunk_size):(ix+1)*chunk_size] if ix != (n_ixes+1) else input[ix:] for ix in range(n_ixes+1)] 

    class Processor():
      def __init__(self):
        self.retriever = Retriever()
        self.translator = Translator()
