import torch
import torch.functional as F

def similarity_calc(embeda: torch.Tensor, embedb: torch.Tensor, normalize: bool = False) -> torch.Tensor:
    #Return similarity metrics between 2 embeddings
    assert embeda.shape[-1] == embedb.shape[-1], 'Embeddings does not have equal shape'
    if normalize:
        embeda = F.normalize(embeda)
        embedb = F.normalize(embedb)
    return embeda @ embedb.T


###
x  = []
for chunk in tqdm(all_sentence_scawk_chunk):
    #print(chunk)
    chunk_tokenized = tr.tokenizer(chunk, return_tensors=TranslatorCFG.return_tensors, padding=True,  truncation=True)
    #print(chunk_tokenized)
    #translated  = tr.retriever.generate(**chunk_tokenized.to(torch.device('cuda')))
    #x.append([tr.tokenizer.decode(el, skip_special_tokens=True) for el in translated])
    output = tr.retriever.generate(**chunk_tokenized.to(torch.device('cuda')))
    trans_chunk = [tr.tokenizer.decode(el, skip_special_tokens=True) for el in output]
    #x= torch.cat((x, output))
    x.append(trans_chunk)
###
