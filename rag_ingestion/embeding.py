from langchain_community.embeddings import OllamaEmbeddings

class Embedder:
    def __init__(self, model_name="nomic-embed-text"):
        self.model_name = model_name
        self.embedder = OllamaEmbeddings(model=self.model_name)

    def embed(self, chunks):
        embeddings = self.embedder.embed_documents([chunk.page_content for chunk in chunks])
        return embeddings

    def embed_query(self, query):
        embedding = self.embedder.embed_query(query)
        return embedding