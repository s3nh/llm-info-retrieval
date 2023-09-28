import ctranslate2
import torch
import numpy as np 
import os 
import re 
from retriever import Retriever
from transformers import MarianMTModel, MarianTokenizer
from typing import List, Dict, Union
from typing import Any, TypeVar
from tqdm import tqdm
from cfg import RetrieverCFG, TranslatorCFG

class Translator:
    def __init__(self):
        self.config = TranslatorCFG
        self._input_text = None
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.tokenizer = self.load_tokenizer()
        self.translator = self.load_translator()

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
            pretrained_model_name_or_path = TranslatorCFG.tokenizer_name, 
            local_files_only = TranslatorCFG.local_files_only 
        )

    def load_hf_model(self):
        """
        Load transformer based MarianMT model
        """
        return MarianMTModel.from_pretrained(
                pretrained_model_name_or_path = TranslatorCFG.model_name, 
                local_files_only = TranslatorCFG.local_files_only
            ).to(self.device).eval()

    def load_ct2_model(self):
        """
        Load ct2 based MarianMT model.
        """
        return ctranslate2.Translator(TranslatorCFG.model_name, device="cuda" if torch.cuda.is_available() else "cpu") 

    def load_translator(self) -> Any:
        """
        Load translator model based on arch file.
        """
        model_files: List = os.listdir(TranslatorCFG.model_name)
        if TranslatorCFG.hf_model_file in model_files:
            return self.load_hf_model()
        elif TranslatorCFG.ct2_model_file in model_files:
            return self.load_ct2_model()
        
    def tokenize(self, input_text: Union[str, List]):
        return self.tokenizer(input_text, 
            padding = TranslatorCFG.padding, 
            truncation = TranslatorCFG.truncation, 
            max_length = TranslatorCFG.max_length, 
            return_tensors = TranslatorCFG.return_tensors ) 

    def translate(self, tokenized: torch.Tensor) -> Union[torch.Tensor, List]:
        return self.translator.generate(tokenizer) 

    def process_once(self) -> List:
        output: List = []
        if isinstance(self.translator, ctranslate2._ext.Translator):
            source = [self.tokenizer.convert_ids_to_tokens(self.tokenizer.encode(el)) for el in self.input_text]
            results = self.translator.translate_batch([source][0])
            target = [results[el].hypotheses[0] if len(results) > 0 else results[0].hypotheses[0] for el in range(len(results))]
            return [self.tokenizer.decode(self.tokenizer.convert_tokens_to_ids(el) ) for el in target]
        else:
            raise NotImplementedError("Assuming that we are working on ctranslate2")

    def process_batch(self, input_base: List[List]) -> List:
        output: List = []
        for chunk in tqdm(input_base):
            self.input_text = chunk
            translated = self.process_once()
            output.append(translated)
        return [el for sublist in output for el in sublist]


    #Batch chunk is used repeatedly so it could move to utils.
    #Nothing unique for specific methods.
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
    
    #TODO: Rethink if needed
    class Processor():
      def __init__(self):
        self.translator = Retriever()
        self.translator = Translator()
