import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

load_dotenv()

# Initialize embeddings and Pinecone
embeddings = OpenAIEmbeddings(
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    model="text-embedding-3-small"
)
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = os.getenv("INDEX_NAME")

# Flat metadata
flat_doc = Document(
    page_content="This is a test document with flat metadata.",
    metadata={
        "title": "Flat Example",
        "author": "Alice",
        "tags": ["test", "flat"],
        "views": 100
    }
)

# Nested metadata
nested_doc = Document(
    page_content="This is a test document with nested metadata.",
    metadata={
        "title": "Nested Example",
        "author": {"first_name": "Bob", "last_name": "Smith"},  # nested dict
        "tags": ["test", "nested"]
    }
)

# Try inserting flat metadata
print("Inserting flat metadata document...")
try:
    PineconeVectorStore.from_documents(
        [flat_doc],
        embedding=embeddings,
        index_name=index_name
    )
    print("Flat metadata succeeded ✅")
except Exception as e:
    print("Flat metadata failed ❌", e)

# Try inserting nested metadata
print("\nInserting nested metadata document...")
try:
    PineconeVectorStore.from_documents(
        [nested_doc],
        embedding=embeddings,
        index_name=index_name
    )
    print("Nested metadata succeeded ❌")  # We expect this to fail
except Exception as e:
    print("Nested metadata failed as expected ✅", e)