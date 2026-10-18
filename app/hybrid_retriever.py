from typing import List, Optional
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever


class HybridRetriever:
    def __init__(self, vectorstore, llm, k=20, fetch_k=40,
                 all_documents=None, search_type="mmr"):
        """
        vectorstore: the raw Chroma vectorstore (not a pre-built retriever)
        all_documents: full list of child Document objects, used to build
                       per-query, per-document-filtered BM25 retrievers
        """
        self.vectorstore = vectorstore
        self.llm = llm
        self.k = k
        self.fetch_k = fetch_k
        self.all_documents = all_documents or []
        self.search_type = search_type

    def _build_vector_retriever(self, selected_documents):
        search_kwargs = {"k": self.k, "fetch_k": self.fetch_k}
        if selected_documents:
            search_kwargs["filter"] = {"document_id": {"$in": selected_documents}}
        return self.vectorstore.as_retriever(
            search_type=self.search_type, search_kwargs=search_kwargs
        )

    def _build_bm25_retriever(self, selected_documents):
        docs = self.all_documents
        if selected_documents:
            docs = [
                d for d in docs
                if d.metadata.get("document_id") in selected_documents
            ]
        if not docs:
            return None
        bm25 = BM25Retriever.from_documents(docs)
        bm25.k = self.k
        return bm25

    def invoke(self, query: str, selected_documents: Optional[List[str]] = None) -> List[Document]:
        from langchain_classic.retrievers import MultiQueryRetriever

        vector_retriever = self._build_vector_retriever(selected_documents)
        multi_vector = MultiQueryRetriever.from_llm(retriever=vector_retriever, llm=self.llm)
        vector_docs = multi_vector.invoke(query)

        bm25 = self._build_bm25_retriever(selected_documents)
        bm25_docs = bm25.invoke(query) if bm25 else []

        merged, seen = [], set()
        for doc in vector_docs + bm25_docs:
            text = doc.page_content
            if text in seen:
                continue
            seen.add(text)
            merged.append(doc)
        return merged