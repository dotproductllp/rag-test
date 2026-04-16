import os, urllib3
from dotenv import load_dotenv
from azure.cosmos import CosmosClient
from openai import OpenAI

load_dotenv()
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

class HybridSearcher:
    def __init__(self):
        self.openai = OpenAI()
        self.client = CosmosClient(
            os.getenv("COSMOS_URL"),
            credential=os.getenv("COSMOS_KEY"),
            connection_verify=False
        )
        self.container = self.client.get_database_client("Coresignal_linkedin").get_container_client("PostEmbeddings")

    def perform_hybrid_search(self, semantic_query: str, required_keywords: list = None, top_k: int = 10):
        # 1. Get Query Embedding
        query_vector = self.openai.embeddings.create(
            input=semantic_query,
            model="text-embedding-3-small"
        ).data[0].embedding
        
        # 2. Base SQL Query
        query = "SELECT TOP @limit c.id, c.article_body, VectorDistance(c.embedding, @vector) AS score FROM c"
        parameters = [
            {"name": "@vector", "value": query_vector},
            {"name": "@limit", "value": top_k * 4}
        ]

        # 3. Dynamically add multiple Keyword Filters (if provided)
        if required_keywords:
            conditions = []
            for i, kw in enumerate(required_keywords):
                parameters.append({"name": f"@kw{i}", "value": kw})
                conditions.append(f"CONTAINS(LOWER(c.article_body), LOWER(@kw{i}))")
            query += f" WHERE {' AND '.join(conditions)}"
            print(f"🔍 Hybrid Search (Keywords: {required_keywords})...")
        else:
            print("🔍 Pure Vector Search...")

        query += " ORDER BY VectorDistance(c.embedding, @vector)"

        # 4. Execute Query & Deduplicate
        raw_results = list(self.container.query_items(
            query=query, parameters=parameters,
            enable_cross_partition_query=True
        ))
        unique_results, seen = [], set()

        for res in raw_results:
            fingerprint = res.get('article_body', '')[:150].lower()
            if fingerprint and fingerprint not in seen:
                seen.add(fingerprint)
                unique_results.append(res)
            if len(unique_results) == top_k:
                break
        
        # 5. Print Results
        print(f"\n{'='*50}\nRESULTS FOR: '{semantic_query}'\n{'='*50}")
        for i, res in enumerate(unique_results):
            print(f"{i+1}. [ID: {res['id']}] [Score: {res.get('score', 0):.4f}]\n{res['article_body'][:3000]}\n")

        return unique_results

if __name__ == "__main__":
    searcher = HybridSearcher()
    
    # Test
    searcher.perform_hybrid_search(
        semantic_query="Dubai real estate market in 2026, property market trends, housing and commercial real estate outlook, investment opportunities, prices, demand, supply, development activity, rental market, off-plan projects, market forecast, United Arab Emirates property sector",
        required_keywords=['Dubai', '2026'] # keep empty array for pure vector searching.
    )
