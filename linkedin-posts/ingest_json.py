import os
import json
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

load_dotenv()

class JSONIngestToPinecone:
    def __init__(self, json_path: str):
        self.json_path = json_path
        self.index_name = os.getenv("INDEX_NAME")
        print("OPENAI Embedding")
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            model="text-embedding-3-small"
        )
        print("API of Pinecone")
        self.pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

    def ingest(self):
        print("Loading JSON")
        documents = []

        with open(self.json_path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except:
                f.seek(0)
                data = [json.loads(line) for line in f]

        for row in data:
            body = row.get("article_body")

            # ✅ Only vectorize article_body
            if not body or body.strip() == "":
                continue

            text = body.strip()

            meta = clean_metadata(row)

            doc = Document(
                page_content=text,
                metadata=meta
            )

            documents.append(doc)

        print(f"Total documents: {len(documents)}")

        print("Pinecone Storing")

        batch_size = 500
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i+batch_size]
            print(f"Processing batch {i} ({i}-{i+len(batch)-1})")
            print(f"Batch {i}")
            try:
                PineconeVectorStore.from_documents(
                    batch,
                    embedding=self.embeddings,
                    index_name=self.index_name,
                )
            except Exception as e:
                print("ERROR:", e)

# Metadata should be flat. No nested json components in metadata.
# Metadata value must be a string, number, boolean or list of strings
def clean_metadata(row: dict) -> dict:
    exclude_fields = ["comments", "reshared_post"]  # already excluding
    clean_meta = {}

    for k, v in row.items():
        if k in exclude_fields:
            continue

        # None → empty string
        if v is None:
            clean_meta[k] = ""
        # List of strings → keep
        elif isinstance(v, list):
            # Convert list of objects to list of strings
            if all(isinstance(x, str) for x in v):
                clean_meta[k] = v
            else:
                # flatten objects into strings (e.g., full_name)
                clean_meta[k] = [json.dumps(x) if isinstance(x, dict) else str(x) for x in v]
        # Primitive types → keep
        elif isinstance(v, (str, int, float, bool)):
            clean_meta[k] = v
        else:
            # Anything else → convert to string
            clean_meta[k] = str(v)

    return clean_meta
    

if __name__ == "__main__":
    json_file = "part-00001-cdd8d172-4e8c-4f51-a941-475ee8444692-c000.json"
    JSONIngestToPinecone(json_file).ingest()
