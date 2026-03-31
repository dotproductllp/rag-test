import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
load_dotenv()

class PDFIngestToPinecone:
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.index_name = os.getenv("INDEX_NAME")

        self.embeddings = OpenAIEmbeddings(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            model="text-embedding-3-small"
        )
        
        self.pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

    def ingest(self):

        loader = PyPDFLoader(self.pdf_path)
        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=150)
        texts = text_splitter.split_documents(documents)

        try:
            PineconeVectorStore.from_documents(
                texts, 
                embedding=self.embeddings, 
                index_name=self.index_name,
            )
        except Exception as e:
            print("ERROR:", e)

if __name__ == "__main__":
    pdf_file = "Genemate_A_Real-Time_Animation_of_a_3D_Human_Model.pdf" 
    
    ingestor = PDFIngestToPinecone(pdf_file)
    ingestor.ingest()