import os
import csv
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

load_dotenv()

class CSVIngestToPinecone:
    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.index_name = os.getenv("INDEX_NAME")
        print("OPENAI Embedding")
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            model="text-embedding-3-small"
        )
        print("API of Pinecone")
        self.pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

    def ingest(self):
        print("CSVLoader")
        documents = []
        with open(self.csv_path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if not row.get("Plot") or row["Plot"].strip() == "":
                    continue
                
                plot_text = row["Plot"].strip() # Allowing only plot to be vectrise
                
                meta = {
                    "title": row.get("Title"),
                    "year": row.get("Release Year"),
                    "director": row.get("Director"),
                    "genre": row.get("Genre")
                }
                
                doc = Document(page_content=plot_text, metadata=meta)
                documents.append(doc)

        print("PineCone Storing")
        try:
            PineconeVectorStore.from_documents(
                documents, 
                embedding=self.embeddings, 
                index_name=self.index_name,
            )
        except Exception as e:
            print("ERROR:", e)

if __name__ == "__main__":
    csv_file = "wiki_movie_plots.csv" 
    CSVIngestToPinecone(csv_file).ingest()
