import ctranslate2
import os 
import numpy as np
import threading
import torch
import torch.nn.functional as F
from typing import List, Dict, Union
from typing import Any, TypeVar
from cfg import RetrieverCFG
from threading import Thread
from torch import Tensor
from transformers import AutoModel 
from transformers import AutoTokenizer
#Retriever load
#2.57 s ± 266 ms per loop (mean ± std. dev. of 7 runs, 3 loops each)
#4.17 s ± 883 ms per loop (mean ± std. dev. of 7 runs, 3 loops each)
#Tokenizer load
#19.8 ms ± 4.61 ms per loop (mean ± std. dev. of 7 runs, 3 loops each)
#20.6 ms ± 1.47 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)
#22.2 ms ± 2.15 ms per loop (mean ± std. dev. of 7 runs, 50 loops each)

#Retriever inference 
# Inference tests
# Gte-large 
# 1.2 s ± 149 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)
# 840 ms ± 55.9 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)

class Retriever:
    def __init__(self):
        self._question = None
        self._input_text = None
        self.parallel: bool = True
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.tokenizer = self.load_tokenizer()
        self.retriever = self.load_retriever()
        torch.set_default_dtype(torch.half)

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

    def average_pool(self, last_hidden_states: Tensor,
                    attention_mask: Tensor) -> Tensor:
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    def load_tokenizer(self):
        return AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path = RetrieverCFG.model_name, 
            local_files_only=True
            )

    def load_retriever(self):
        retriever = AutoModel.from_pretrained(
                pretrained_model_name_or_path = RetrieverCFG.model_name,
                local_files_only = True
            )
        if self.device == torch.device('cuda') and RetrieverCFG.model_half:
            return  retriever.to(self.device).half().eval()
        else:
            return retriever.to(self.device).eval()

    def load_translator(self):
        ...

    def embedd_normalize(self, embeddings: torch.Tensor):
        return F.normalize(embeddings)

    def check_embed(self, embed1, embed2):
        return torch.all(embed1 == embed2)

    def process_thread(self):
        if self.parallel:
            process = threading.Thread(target = self.process_once, name = 'process_once')
            process.daemon = True
            process.start()

    def process_translator(self):
        ...        

    def process_once(self):
        """
        Process self.input_text in few steps, based on arguments 
        stored in RetrieverCFG.
        Few steps happened here. 
        """
        batch_dict = self.tokenizer(self.input_text, max_length = RetrieverCFG.max_length, 
            padding = RetrieverCFG.padding, 
            truncation = RetrieverCFG.truncation, 
            return_tensors = RetrieverCFG.return_tensors).to(self.device)

        with torch.no_grad():
            outputs = self.retriever(**batch_dict)
        embeddings = self.average_pool(
            last_hidden_states = outputs.last_hidden_state.to(self.device),
            attention_mask = batch_dict['attention_mask'].to(self.device)
        )
        #embeddings = self.embedd_normalize(embeddings)
        return embeddings

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
        chunk_size = RetrieverCFG.chunk_size
        n_ixes = int(emb_len/RetrieverCFG.chunk_size)
        return [input_base[(ix*chunk_size):(ix+1)*chunk_size] if ix != (n_ixes+1) else input[ix:] for ix in range(n_ixes+1)] 

    def batch_processing(self, base: List[List], chunk: bool = False) -> torch.Tensor:
        """
        Batch processing based on previously chunk data. 
        If chunk = True, chunk data using self.batch_chunk 
        using args stored in Retriever.CFG
        """
        output = torch.Tensor()
        if not chunk:
            for chunk in base:
                chunk = [re.sub(r'[^\w\s]', '', str(el) + RetrieverCFG.pad_token * (RetrieverCFG.max_length - len(str(el).split()))) for el in chunk]
                self.input_text = chunk
                embeddings = self.process_once().cpu()
                torch.cuda.empty_cache()
                output = torch.cat((output, embeddings))
        return output
