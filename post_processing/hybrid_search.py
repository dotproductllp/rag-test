import os, urllib3
from dotenv import load_dotenv
from azure.cosmos import CosmosClient
from openai import OpenAI

load_dotenv()
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

class HybridSearch:
    def __init__(self):
        self.openai = OpenAI()
        self.client = CosmosClient(
            os.getenv("COSMOS_URL"),
            credential=os.getenv("COSMOS_KEY"),
            connection_verify=False
        )
        self.container = self.client.get_database_client("Coresignal_linkedin").get_container_client("VectorEmbeddings")

    def perform_hybrid_search(self, semantic_query: str, required_keywords: list = None, top_k: int = 5):
        # 1. Get Query Embedding
        query_vector = self.openai.embeddings.create(
            input=semantic_query,
            model="text-embedding-3-small"
        ).data[0].embedding
        
        # 2. Base Parameters
        parameters =[
            {"name": "@vector", "value": query_vector},
            {"name": "@limit", "value": top_k}
        ]

        # 3. Construct Native Hybrid Query
        if required_keywords:
            # Map each keyword to a SQL parameter
            kw_vars =[]
            for i, kw in enumerate(required_keywords):
                param_name = f"@kw{i}"
                parameters.append({"name": param_name, "value": kw})
                kw_vars.append(param_name)
            
            # Format: @kw0, @kw1, @kw2
            kw_str = ", ".join(kw_vars)
            
            # Use ORDER BY RANK RRF to fuse Vector Search and BM25 Full Text Search
            query = f"""
                SELECT TOP @limit 
                    c.id, 
                    c.article_body, 
                    VectorDistance(c.embedding, @vector) AS score
                FROM c 
                ORDER BY RANK RRF(
                    VectorDistance(c.embedding, @vector), 
                    FullTextScore(c.article_body, {kw_str})
                )
            """
            print(f"🔍 Native Hybrid Search (Keywords: {required_keywords})...")
        else:
            # Pure Vector Search fallback
            query = """
                SELECT TOP @limit c.id, c.article_body, VectorDistance(c.embedding, @vector) AS score 
                FROM c 
                ORDER BY VectorDistance(c.embedding, @vector)
            """
            print("🔍 Pure Vector Search...")

        # 4. Execute Query
        results = list(self.container.query_items(
            query=query, 
            parameters=parameters,
            enable_cross_partition_query=True
        ))
        
        # 5. Print Results
        print(f"\n{'='*50}\nRESULTS FOR: '{semantic_query}'\n{'='*50}")
        for i, res in enumerate(results):
            # Note: RRF doesn't expose a raw 'score' property in the SELECT clause by default
            score_display = f" [Score: {res['score']:.4f}]" if 'score' in res else ""
            print(f"{i+1}. [ID: {res['id']}]{score_display}\n{res['article_body'][:3000]}\n")

        return results

if __name__ == "__main__":
    searcher = HybridSearch()
    
    # Test
    searcher.perform_hybrid_search(
        semantic_query="Dubai real estate market in 2026, property market trends, housing and commercial real estate outlook, investment opportunities, prices, demand, supply, development activity, rental market, off-plan projects, market forecast, United Arab Emirates property sector",
        required_keywords=["Dubai", "2026"] # keep empty array for pure vector searching.
    )
