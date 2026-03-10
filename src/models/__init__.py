import torch
from omegaconf import DictConfig


# 1. Base Models
from .bm25 import BM25Model
from src.models.our_retriever import OurRetriever
from src.models.our_retriever_max_pooling import OurRetrieverMaxPooling
from src.models.our_retriever_conv import OurRetrieverConv
from src.models.our_retriever_conv_max import OurRetrieverConvMax
from src.models.our_retriever_conv_meanmax import OurRetrieverConvMeanMax
from src.models.our_retriever_meanmax import OurRetrieverMeanMax
from src.models.our_reranker import OurCrossEncoder
from src.models.g2p_bm25_ngram import BM25NGramModel
from src.models.dpr import DPRModel

# 2. Pipeline & Hybrid Models
from src.models.conv_meanmax_with_reranker import ConvMeanMaxWithReranker
from src.models.bm25_dense_hybrid import BM25DenseHybridRetriever
from src.models.bm25_with_reranker import BM25WithReranker
from src.models.hybrid_with_reranker import HybridWithReranker
from src.models.bm25_dense_hybrid_rank import BM25DenseRankHybridRetriever
from src.models.g2p_bm25_dense_hybrid import G2PBM25DenseHybridRetriever
from src.models.g2p_bm25_dense_hybrid_rank import G2PBM25DenseRankHybridRetriever
from src.models.g2p_bm25_dense_hybrid_zscore import G2PBM25DenseZScoreHybridRetriever
from src.models.retriever_with_reranker import RetrieverWithReranker


def get_model(cfg_model):
    name = cfg_model.name

    # 안전한 params 추출 (BM25용)
    bm25_params = cfg_model.get("params", {})

    # Single Models
    if name == "bm25":
        return BM25Model(**bm25_params)

    elif name == "g2p_bm25_ngram":
        return BM25NGramModel(**bm25_params)

    elif name == "dpr":
        return DPRModel(**bm25_params)

    elif name == "our_retriever":
        model = OurRetriever(cfg_model)
        if torch.cuda.is_available(): model = model.cuda()
        return model

    elif name == "our_retriever_max_pooling":
        model = OurRetrieverMaxPooling(cfg_model)
        if torch.cuda.is_available(): model = model.cuda()
        return model

    elif name == "our_retriever_conv":
        model = OurRetrieverConv(cfg_model)
        if torch.cuda.is_available(): model = model.cuda()
        return model

    elif name == "our_retriever_conv_max":
        model = OurRetrieverConvMax(cfg_model)
        if torch.cuda.is_available(): model = model.cuda()
        return model

    elif name == "our_retriever_conv_meanmax":
        model = OurRetrieverConvMeanMax(cfg_model)
        if torch.cuda.is_available(): model = model.cuda()
        return model

    elif name == "our_retriever_meanmax":
        model = OurRetrieverMeanMax(cfg_model)
        if torch.cuda.is_available(): model = model.cuda()
        return model

    elif name == "our_reranker":
        model = OurCrossEncoder(cfg_model)
        if torch.cuda.is_available(): model = model.cuda()
        return model

    # Hybrid Models
    elif name == "bm25_dense_hybrid":
        bm25 = BM25Model(**bm25_params)
        dense = OurRetrieverConvMeanMax(cfg_model)

        alpha = cfg_model.get('alpha', 0.5)
        model = BM25DenseHybridRetriever(bm25, dense, alpha=alpha)
        if torch.cuda.is_available(): model = model.cuda()
        return model

    elif name == "bm25_dense_hybrid_rank":
        bm25 = BM25Model(**bm25_params)
        dense = OurRetrieverConvMeanMax(cfg_model)

        alpha = cfg_model.get('alpha', 0.5)
        rrf_k = cfg_model.get('rrf_k', 60)
        model = BM25DenseRankHybridRetriever(bm25, dense, alpha=alpha, rrf_k=rrf_k)
        if torch.cuda.is_available(): model = model.cuda()
        return model

    elif name == "conv_meanmax_with_reranker":
        retriever = OurRetrieverConvMeanMax(cfg_model)
        reranker = OurCrossEncoder(cfg_model)

        model = ConvMeanMaxWithReranker(retriever, reranker, max_length=cfg_model.get('max_length', 512))
        if torch.cuda.is_available(): model = model.cuda()
        return model

    elif name == "bm25_with_reranker":
        bm25 = BM25Model(**bm25_params)
        reranker = OurCrossEncoder(cfg_model)

        model = BM25WithReranker(bm25, reranker, max_length=cfg_model.get('max_length', 512))
        if torch.cuda.is_available(): model = model.cuda()
        return model

    elif name == "hybrid_with_reranker":
        bm25 = BM25Model(**bm25_params)
        dense = OurRetrieverConvMeanMax(cfg_model)
        alpha = cfg_model.get('alpha', 0.5)
        hybrid_retriever = BM25DenseHybridRetriever(bm25, dense, alpha=alpha)

        reranker = OurCrossEncoder(cfg_model)

        model = HybridWithReranker(hybrid_retriever, reranker, max_length=cfg_model.get('max_length', 512))
        if torch.cuda.is_available(): model = model.cuda()
        return model

    elif name == "g2p_bm25_dense_hybrid":
        bm25_params = cfg_model.get('bm25_params', {})
        alpha = cfg_model.get('alpha', 0.5)

        bm25 = BM25NGramModel(**bm25_params)
        dense = OurRetrieverConvMeanMax(cfg_model)

        model = G2PBM25DenseHybridRetriever(
            bm25_model=bm25,
            dense_retriever=dense,
            alpha=alpha
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        return model

    elif name == "g2p_bm25_dense_hybrid_rank":
        bm25_params = cfg_model.get('bm25_params', {})
        alpha = cfg_model.get('alpha', 0.5)
        rrf_k = cfg_model.get('rrf_k', 60)

        bm25 = BM25NGramModel(**bm25_params)
        dense = OurRetrieverConvMeanMax(cfg_model)

        model = G2PBM25DenseRankHybridRetriever(
            bm25_model=bm25,
            dense_retriever=dense,
            alpha=alpha,
            rrf_k=rrf_k
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        return model

    elif name == "g2p_bm25_dense_hybrid_zscore":
        bm25_params = cfg_model.get('bm25_params', {})
        alpha = cfg_model.get('alpha', 0.5)

        bm25 = BM25NGramModel(**bm25_params)
        dense = OurRetrieverConvMeanMax(cfg_model)

        model = G2PBM25DenseZScoreHybridRetriever(
            bm25_model=bm25,
            dense_retriever=dense,
            alpha=alpha
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        return model

    elif name == "reranker_pipeline":
        base_cfg = cfg_model.get("base_retriever_cfg")
        if not base_cfg:
            raise ValueError("base_retriever_cfg is missing from config.")

        base_retriever = get_model(base_cfg)

        reranker_model = OurCrossEncoder(cfg_model)
        if torch.cuda.is_available():
            reranker_model = reranker_model.cuda()

        rerank_max_len = cfg_model.get("reranker_max_length", 384)

        model = RetrieverWithReranker(
            retriever=base_retriever,
            reranker=reranker_model,
            max_length=rerank_max_len
        )

        return model

    elif name == "semantic_pipeline":
        from src.models.semantic_pipeline import SemanticPipelineRetriever
        model = SemanticPipelineRetriever(
            e5_model_name=cfg_model.e5_model_name,
            bge_reranker_name=cfg_model.bge_reranker_name
        )
        return model

    else:
        raise ValueError(f"Unknown Model Name: {name}")
