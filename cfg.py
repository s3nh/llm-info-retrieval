import numpy as np

class RetrieverCFG:
    model_name: str = ''
    local_files_only: bool = True
    max_length: int = 512
    padding: bool = True
    truncation: bool = True
    return_tensors: str = 'pt'
    chunk_size: int = 16
