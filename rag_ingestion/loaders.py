import os

from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader

class Loader:
    def __init__(self, file_path):
        self.file_path = file_path

    def load(self):
        if self.file_path.lower().endswith('.pdf'):
            loader = PyPDFLoader(self.file_path)

        elif self.file_path.lower().endswith('.txt'):
            loader = TextLoader(self.file_path)

        elif self.file_path.lower().endswith('.csv'):
            loader = CSVLoader(self.file_path)

        else:
            raise ValueError("Unsupported file type. Supported: .pdf, .txt, .csv")

        documents = loader.load()
        return documents