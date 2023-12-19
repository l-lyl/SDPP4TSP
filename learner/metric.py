import numpy as np
import torch
import json


def mae(predict, truth):
    return torch.mean(torch.abs(predict - truth)).item()


def mse(predict, truth):
    return torch.mean((predict - truth) ** 2).item()


def rmse(predict, truth):
    return torch.sqrt(torch.mean((predict - truth) ** 2)).item()


def phr(predict, truth):
    pass


def recall(predict, truth, top_k=5, coex_embs=None):
    """
    Args:
        predict (Tensor): shape (batch_size, items_total)
        truth (Tensor): shape (batch_size, items_total)
        top_k (int):
    Returns:
        output (float)
    """
    if coex_embs == None:
        _, predict_indices = predict.topk(k=top_k)
    else:
        '''
        #first topk based on scores, and then select based on co-ex
        _, predict_indices = predict.topk(k=top_k+int(top_k/5))
        dummy =predict_indices.unsqueeze(2).expand(predict_indices.size(0), predict_indices.size(1), coex_embs.size(2))
        indexed_embs = torch.gather(coex_embs, 1, dummy)
        co_scores = torch.bmm(indexed_embs, torch.transpose(indexed_embs, 1,2)).mean(2)
        _, scor_indices = co_scores.topk(k=top_k)
        
        res = []
        for i in range(predict_indices.shape[0]):
            res.append(predict_indices[i][scor_indices[i]])
        res = torch.stack(res, 0)
        predict_indices = res
        '''
        #first remove less likely being co-ex, and then topk, remove last b
        predict_min = torch.min(predict)
        co_scores = torch.bmm(coex_embs, torch.transpose(coex_embs, 1,2)).mean(2)
        _, co_indices = co_scores.topk(k=int(predict.shape[1]*0.2), largest=False) # *b
        predict.scatter_(1, co_indices, predict_min.item())
        _, predict_indices = predict.topk(k=top_k)

    predict, truth = predict.new_zeros(predict.shape).scatter_(1, predict_indices, 1).long(), truth.long()
    tp, t = ((predict == truth) & (truth == 1)).sum(-1), truth.sum(-1)
    return (tp.float() / (t.float() + 1e-7)).mean().item()

def precision(predict, truth, top_k=5, coex_embs=None):
    """
    Args:
        predict (Tensor): shape (batch_size, items_total)
        truth (Tensor): shape (batch_size, items_total)
        top_k (int):
    Returns:
        output (float)
    """
    _, predict_indices = predict.topk(k=top_k)
    predict, truth = predict.new_zeros(predict.shape).scatter_(1, predict_indices, 1).long(), truth.long()
    tp, p = ((predict == truth) & (truth == 1)).sum(-1), predict.sum(-1)
    return (tp.float() / (p.float() + 1e-7)).mean().item()


def f1_score(predict: torch.Tensor, truth: torch.Tensor, top_k=5, coex_embs=None):
    """
    Args:
        predict (Tensor): shape (batch_size, items_total)
        truth (Tensor): shape (batch_size, items_total)
        top_k (int):
    Returns:
        output (float)
    """
    _, predict_indices = predict.topk(k=top_k)
    predict, truth = predict.new_zeros(predict.shape).scatter_(1, predict_indices, 1).long(), truth.long()
    tp, t, p = ((predict == truth) & (truth == 1)).sum(-1), truth.sum(-1), predict.sum(-1)
    precision, recall = tp.float() / (p.float() + 1e-7), tp.float() / (t.float() + 1e-7)
    return (2 * precision * recall / (precision + recall + 1e-7)).mean().item()


def dcg(type, predict, truth, top_k, coex_embs=None):
    """
    Args:
        predict: (batch_size, items_total)
        truth: (batch_size, items_total)
        top_k:

    Returns:

    """
    if type == 0 and coex_embs != None:
        '''
        _, predict_indices = predict.topk(k=top_k+int(top_k/5))
        dummy =predict_indices.unsqueeze(2).expand(predict_indices.size(0), predict_indices.size(1), coex_embs.size(2))
        indexed_embs = torch.gather(coex_embs, 1, dummy)
        co_scores = torch.bmm(indexed_embs, torch.transpose(indexed_embs, 1,2)).mean(2)
        _, scor_indices = co_scores.topk(k=top_k)
        
        res = []
        for i in range(predict_indices.shape[0]):
            res.append(predict_indices[i][scor_indices[i]])
        res = torch.stack(res, 0)
        predict_indices = res
        '''
        predict_min = torch.min(predict)
        co_scores = torch.bmm(coex_embs, torch.transpose(coex_embs, 1,2)).mean(2)
        _, co_indices = co_scores.topk(k=int(predict.shape[1]*0.2), largest=False)  ## remove last 0.02 items
        predict.scatter_(1, co_indices, predict_min.item())
        _, predict_indices = predict.topk(k=top_k)
    else:
        _, predict_indices = predict.topk(k=top_k)
    gain = truth.gather(-1, predict_indices)  # (batch_size, top_k)
    return (gain.float() / torch.log2(torch.arange(top_k, device=predict.device).float() + 2)).sum(-1)  # (batch_size,)

def get_cate_map(cate_path):
    with open(cate_path, 'r') as file:
        data_dict = json.load(file)
    iidcate_map = {}   #cate:iids 
    cates = set()
    for iid, cs in data_dict.items():
        iidcate_map[int(iid)] = cs
        cates.add(cs)
    return iidcate_map, len(cates)

def cc(cate_path, predict, truth, top_k, coex_embs=None):

    cate_map, cate_num = get_cate_map(cate_path)
    _, predict_indices = predict.topk(k=top_k)
    
    cc_scores = 0
    for bi in predict_indices:
        cate_list = []
        for i in bi:
            cates = cate_map[i.item()]
            cate_list.append(cates)
        
        cate_c = set()
        for i in range(top_k):
            cate_c.add(cate_list[i])
        cc_scores += len(cate_c) / cate_num 
    return cc_scores / predict.shape[0]

def ndcg(predict, truth, top_k, coex_embs=None):
    """
    Args:
        predict: (batch_size, items_total)
        truth: (batch_size, items_total)
        top_k:

    Returns:

    """
    dcg_score = dcg(0, predict, truth, top_k, coex_embs=coex_embs)
    idcg_score = dcg(1, truth, truth, top_k, coex_embs=coex_embs)
    return (dcg_score / idcg_score).mean().item()


if __name__ == '__main__':
    predict = torch.tensor([
        [0.9, 0.6, 0.1, 0.5, 0.4],
        [0.2, 0.3, 0.6, 0.4, 0.8],
    ])

    truth = torch.tensor([
        [1, 0, 1, 1, 0],
        [0, 1, 1, 0, 0]
    ])

    print(recall(predict, truth, top_k=3))  
    print(precision(predict, truth, top_k=3))  
    print(f1_score(predict, truth, top_k=3)) 
    print(ndcg(predict, truth, top_k=3))
