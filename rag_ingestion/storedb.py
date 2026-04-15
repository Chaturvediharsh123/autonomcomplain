import os
from langchain_community.vectorstores import Chroma

class VectorStore:
    def __init__(self, persist_dir="./chroma_db"):
        self.persist_dir = persist_dir
        self.vectorstore = None

    def create(self, chunks, embedder):
        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embedder,
            persist_directory=self.persist_dir
        )

    def save(self):
        if self.vectorstore:
            self.vectorstore.persist()

    def load(self, embedder):
        self.vectorstore = Chroma(
            persist_directory=self.persist_dir,
            embedding_function=embedder
        )
        return self.vectorstore

    def similarity_search(self, query, k=4):
        if not self.vectorstore:
            raise ValueError("Vector store not initialized. Load or create first.")
        results = self.vectorstore.similarity_search(query, k=k)
        return results