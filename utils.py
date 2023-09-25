import torch
import torch.functional as F

def similarity_calc(embeda: torch.Tensor, embedb: torch.Tensor, normalize: bool = False) -> torch.Tensor:
    #Return similarity metrics between 2 embeddings
    assert embeda.shape[-1] == embedb.shape[-1], 'Embeddings does not have equal shape'
    if normalize:
        embeda = F.normalize(embeda)
        embedb = F.normalize(embedb)
    return embeda @ embedb.T
