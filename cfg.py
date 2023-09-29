import numpy as np

class TranslatorCFG:
    model_name: str = '../assets/ct2-opus-mt-pl-en'
    tokenizer_name: str = '../assets/opus-mt-pl-en'
    local_files_only: bool = True
    chunk_size: int = 12
    max_length: int = 512
    return_tensors: str = 'pt'
    hf_model_file: str = 'pytorch_model.bin'
    ct2_model_file: str = 'model.bin'

class RetrieverCFG:
    model_name: str = '../assets/bge-large-v1'
    local_files_only: bool = True
    max_length: int = 512
    padding: bool = True
    truncation: bool = True
    return_tensors: str = 'pt'
    chunk_size: int = 16
    pad_token: str = "PAD "
    model_half: bool = False 
