from sentence_transformers import CrossEncoder
from utils.logger import logger
from config import CROSS_ENCODER_MODEL, RERANK_TOP_K

_reranker = None

def load_reranker():
    global _reranker

    if _reranker is None:
        logger.info("Loading Cross Encoder...")

        _reranker = CrossEncoder(
            CROSS_ENCODER_MODEL
        )

    return _reranker

def rerank_documents(query, documents, top_k=RERANK_TOP_K):
    
    model = load_reranker()

    pairs = [(query, doc.page_content) for doc in documents]

    scores = model.predict(pairs)

    scored_documents = list(zip(documents, scores))

    scored_documents.sort(
        key=lambda x: x[1],
        reverse=True
    )

    logger.info(f"Reranked {len(documents)} documents.")

    for i, (doc, score) in enumerate(scored_documents[:top_k], 1):
        logger.info(
            f"[{i}] {doc.metadata['chunk_id']} | Score={score:.4f}"
        )

    return scored_documents[:top_k]

