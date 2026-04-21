from typing import Optional
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from openai import OpenAI
load_dotenv()

INVALID_QUERY_MARKER = "invalid query"

# Define the structured output format using Pydantic
class SearchParameters(BaseModel):
    search_query: str = Field(
        description=(
            "A rich, descriptive semantic query optimized for cosine similarity vector search. "
            "If the user's prompt is gibberish, keyboard-mashing, or not understandable at all, "
            "return EXACTLY the string 'invalid query' (lowercase) for this field."
        )
    )
    keywords: list[str] = Field(
        description="A list of EXACT MATCH deal-breaker keywords. Must be kept empty unless there are strict nouns like Company Names, Specific Years, or People."
    )
    invalid_reason: Optional[str] = Field(
        default=None,
        description=(
            "Only populated when search_query == 'invalid query'. "
            "A short, user-friendly explanation of why the prompt could not be understood."
        )
    )

class QueryGenerator:
    def __init__(self):
        self.client = OpenAI()
        
        self.system_prompt = """
        You are an expert search query analyzer for a RAG pipeline querying a database of LinkedIn posts.
        Your job is to take a user's natural language request and break it down into two parts:
        1. search_query: A reformulated, highly descriptive string optimized for text-embedding models. 
           Include synonyms and focus on the core intent.
        2. keywords: A list of strict, exact-match keywords.
        
        CRITICAL RULES FOR KEYWORDS:
        - ONLY extract absolute deal-breakers. These will be used in a strict SQL `CONTAINS` clause.
        - GOOD keywords: Brand names ("Apple", "Microsoft"), Specific Years ("2025"), Months ("November"), Proper Nouns ("Elon Musk"), Place ("Delhi"), or region ("Middle east") names.
        - BAD keywords: Concepts, verbs, or general topics ("coding", "strategy", "advice", "AI", "marketing"). 
          Using these as keywords will cause false negatives! A post might say "Artificial Intelligence" instead of "AI", 
          or "planning" instead of "strategy", which the vector search would catch, but a strict keyword filter would reject.
        - If the user's prompt is broad or lacks strict proper nouns/dates, return an EMPTY ARRAY `[]` for keywords. 
          Let the vector search do the heavy lifting!

        MEANINGFULNESS CHECK (very important):
        - If the user's prompt is gibberish, random characters, keyboard-mashing, or something you
          genuinely cannot understand as a search intent, DO NOT try to invent a query.
          Instead set:
              search_query    = "invalid query"   (exactly, lowercase)
              keywords        = []
              invalid_reason  = "<short human-readable reason>"
        - Broad-but-coherent topics (e.g. "AI trends in 2026", "Dubai real estate") ARE valid;
          do NOT mark those as invalid.

        === EXAMPLE ===
        User: "real-estate market situation in middle east this year"
        Assistant: {
            "search_query": "current real estate market situation in the Middle East this year, property market trends, housing demand, commercial and residential real estate conditions, prices, investment activity, development outlook, regional market analysis"
            "keywords": [Middle East]
        }
        """

    def generate_query_keyword(self, user_prompt: str) -> dict:

        print(f"Analyzing prompt: '{user_prompt}'...\n")
        
        response = self.client.chat.completions.parse(
            model="gpt-5.4-mini",
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format=SearchParameters,
            temperature=0.1
        )

        result = response.choices[0].message.parsed
        
        return {
            "search_query": result.search_query,
            "keywords": result.keywords,
            "invalid_reason": result.invalid_reason,
        }

if __name__ == "__main__":
    analyzer = QueryGenerator()

    while True:
        user_prompt = input("\nEnter your prompt (or type 'exit' to quit): ")
        if user_prompt.lower() == "exit":
            break
        plan = analyzer.generate_query_keyword(user_prompt)
        print(f"\nUser Prompt   : {user_prompt}")
        print(f"Semantic Query: {plan['search_query']}")
        print(f"Keywords      : {plan['keywords']}")
        print("-" * 30)