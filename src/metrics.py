# src/metrics.py
import torch

def calculate_metrics_generic(retrieved_items, gold_items_set, k_list):
    """
    Passage든 Song이든 '예측 리스트'와 '정답 셋'을 주면 지표를 계산해주는 범용 함수
    """
    metrics = {}
    
    # 1. Recall@K
    for k in k_list:
        top_k_items = set(retrieved_items[:k])
        # 교집합이 있으면 Hit
        hit = 1 if not top_k_items.isdisjoint(gold_items_set) else 0
        metrics[f'R@{k}'] = hit
            
    # 2. MRR
    mrr = 0.0
    for rank, item in enumerate(retrieved_items, 1):
        if item in gold_items_set:
            mrr = 1.0 / rank
            break
            
    metrics['MRR'] = mrr
    return metrics

def calculate_retrieval_metrics(scores, labels, k_list=[1, 5, 10, 50, 100]):
    """
    Args:
        scores (torch.Tensor): (Query 수, Passage 수) 크기의 유사도 행렬
        labels (torch.Tensor): (Query 수,) 크기의 정답 인덱스 (각 쿼리의 정답 Passage 위치)
        k_list (list): 계산할 Top-K 목록
    Returns:
        dict: {'R@1': ..., 'R@5': ..., 'MRR': ...}
    """
    metrics = {}
    
    # 1. 점수 기준으로 내림차순 정렬하여 랭킹(Indices) 구함
    # topk를 쓰면 전체 정렬보다 빠름 (최대 K까지만 보면 되므로)
    max_k = max(k_list)
    _, topk_indices = torch.topk(scores, k=max_k, dim=1)
    
    # 2. 정답이 Top-K 안에 있는지 확인 (Broadcasting)
    # labels.view(-1, 1): (N, 1)
    # topk_indices: (N, max_k)
    # hit_matrix: (N, max_k) -> 정답이면 True, 아니면 False
    hit_matrix = (topk_indices == labels.view(-1, 1))
    
    # 3. Recall@K 계산
    for k in k_list:
        # 각 쿼리별로 k번째까지 True가 하나라도 있는지 확인
        hits = hit_matrix[:, :k].any(dim=1)
        metrics[f"R@{k}"] = hits.float().mean().item()
        
    # 4. MRR 계산
    # hit_matrix에서 True가 있는 위치(rank)를 찾음
    # nonzero() -> [row_idx, col_idx] 반환. col_idx가 곧 rank(0-based)
    hits_indices = hit_matrix.nonzero()
    
    mrr_sum = 0.0
    if len(hits_indices) > 0:
        # col_idx는 0부터 시작하므로 +1 해줘야 실제 순위
        ranks = hits_indices[:, 1] + 1
        reciprocal_ranks = 1.0 / ranks.float()
        mrr_sum = reciprocal_ranks.sum().item()
        
    metrics['MRR'] = mrr_sum / scores.size(0)
    
    return metrics