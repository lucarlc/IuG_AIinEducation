# FAISS-Build, optionaler Hybrid-Retriever (BM25), einfacher Rank-Fusion
from typing import List, Optional
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document

try:
    from langchain_community.retrievers import BM25Retriever  # nutzt rank-bm25
except Exception:
    BM25Retriever = None


def build_faiss(docs: List[Document]) -> FAISS:
    emb = OpenAIEmbeddings()
    return FAISS.from_documents(docs, emb)


class HybridEnsemble:
    """
    Ensemble über FAISS (semantisch) + optional BM25 (lexikalisch).
    - MMR für Diversität (reduziert Dopplungen)
    - einfache Rank-Fusion (FAISS bevorzugt, BM25 ergänzt)
    """
    def __init__(self, faiss: FAISS, docs_for_bm25: Optional[List[Document]] = None):
        self.faiss = faiss
        self.docs = docs_for_bm25 or []
        self.bm25 = BM25Retriever.from_documents(self.docs) if (BM25Retriever and self.docs) else None

    def search(self, query: str, k: int = 6, mmr: bool = True) -> List[Document]:
        faiss_hits = (
            self.faiss.max_marginal_relevance_search(query, k=k)
            if mmr else
            self.faiss.similarity_search(query, k=k)
        )
        if not self.bm25:
            return faiss_hits

        bm25_hits = self.bm25.get_relevant_documents(query)

        # gewichtete Rank-Fusion: FAISS (0.7) > BM25 (0.3)
        pool, seen = [], set()
        weighted = []
        for i,d in enumerate(faiss_hits):
            weighted.append((0.7/(i+1), d))
        for j,d in enumerate(bm25_hits):
            weighted.append((0.3/(j+1), d))
        for _, d in sorted(weighted, key=lambda x: -x[0]):
            key = (d.metadata.get("file"), d.metadata.get("page_start"))
            if key in seen:
                continue
            seen.add(key)
            pool.append(d)
        return pool[:k]
