import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
load_dotenv()

class SemanticSearchEngine:
    def __init__(self):
        self.index_name = os.getenv("INDEX_NAME")
        self.top_k = 7

        self.embeddings = OpenAIEmbeddings(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            model="text-embedding-3-small"
        )

        self.vectorstore = PineconeVectorStore.from_existing_index(
            index_name=self.index_name,
            embedding=self.embeddings
        )
        
    def semantic_search_with_score(self, query):
        results = self.vectorstore.similarity_search_with_score(
            query,
            k=self.top_k
        )
        return results

if __name__ == "__main__":
    engine = SemanticSearchEngine()

    while True:
        user_query = input("Enter text: ")

        results = engine.semantic_search_with_score(user_query)

        print(f"\nTop {len(results)} results:\n")
        for i, (doc, score) in enumerate(results, start=1):
            post_id = doc.metadata.get("id")
            author_name = doc.metadata.get("author_name")
            snippet = doc.page_content[:900].replace('\n', ' ')
            print(f"{i}-> Score: {score:.4f} | Post ID: {post_id} | Author: {author_name} : BODY: {snippet}\n")
