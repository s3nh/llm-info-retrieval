import pathlib
import pickle
import torch
import torch.functional as F
from pathlib import Path
from typing import List, Dict, Union, Any
from cfg import RetrieverCFG
# Fastest cosine similarity calc at the moment. 
# %%timeit -n 10
#similarity_calc(question_embedd, torch.rand((15000, 1024)))
#114 ms ± 1.21 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)

def similarity_calc(embeda: torch.Tensor, embedb: torch.Tensor, normalize: bool = False) -> torch.Tensor:
    """
    If normalize set to True, then torch.functional.normalize is used 
    """
    assert embeda.shape[-1] == embedb.shape[-1], 'Embeddings does not have equal shape'
    if normalize:
        embeda = F.normalize(embeda)
        embedb = F.normalize(embedb)
    return embeda @ embedb.T

def get_most_similar(similarities: torch.Tensor) -> torch.Tensor:
    """
    Return argmax index from tensor  of similarities 
    """
    return torch.argmax(similarities, dim = 1)

def path_exists(path: Union[str, pathlib.Path]) -> bool:
    if isinstance(path, str):
        path = pathlib.Path(path)
    return pathlib.Path(path).exists()

def load_pickle(path: Union[str, pathlib.Path]) -> Any:
    if path_exists(path):
        with open(path, 'rb') as outfile:
            output = pickle.load(outfile)
        return output
    else:
        raise ValueError("Provided path does not exist")

def load_tensors(path: Union[str, pathlib.Path]) -> torch.Tensor:
    if path_exists(path):
        return torch.load(path, map_location= torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    else:
        raise ValueError("Provided path does not exist") 


#TODO: remove
def padding_token(input):
    return self.tokenizer(
        self.input_text,
        return_tensors="pt",
        padding=True,
        truncation = True,
        max_length=512.
    )


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

def save_pickle(object, outpath) -> None:
    with open(outpath, 'wb') as outfile:
        pickle.dump(object, outfile)

def translate_base(input, unit_name, outdict: Dict):
    tr = Translator()
    all_chunk = tr.batch_chunk(input)
    output = tr.process_batch(input_base = all_chunk)
    outdict[unit_name] = output
