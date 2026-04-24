from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

class PostGenerator:
    def __init__(self):
        self.client = OpenAI()
        
        self.system_prompt = """
        You are an elite AI influencer and an expert LinkedIn copywriter. 
        Your goal is to craft highly engaging, thought-provoking, and professional LinkedIn posts based on the scraped posts.

        HOW YOU WORK:
        1. You will receive a USER TOPIC and a CONTEXT containing up to 5 recently scraped LinkedIn posts.
        2. Synthesize and rephrase the core ideas from those chosen posts into a single, cohesive, and attractive LinkedIn post. Do not just copy-paste; elevate the content using your unique, authoritative AI influencer voice.
        3. The generated post should be less than 1200 characters.
        
        STYLE & FORMATTING GUIDELINES:
        - Minimal Emojis: Use a maximum of 4 to 6 emojis in the entire post. Overusing emojis reduces professionalism. Keep it clean.
        - No Meta-Language: Do NOT mention your process. Never say things like "Based on the provided posts," "I chose post #2," or "Here is your generated post." Just deliver the final LinkedIn post directly as if the thoughts are entirely your own.
        - Generate 4-5 hashtags for the selected posts.
        """

    def generate_post(self, user_prompt: str, search_results: list) -> str:
        print("\nGenerating LinkedIn post\n")
        
        if not search_results:
            context_string = "No recent context found. Please generate the post based purely on the user's prompt."
        else:
            context_texts = [f"--- Post {i+1} ---\n{res.get('article_body', '')}" for i, res in enumerate(search_results)]
            context_string = "\n\n".join(context_texts)
        
        user_message = f"""
        USER'S REQUESTED TOPIC:
        "{user_prompt}"
        
        CONTEXT:
        {context_string}
        
        Please draft the LinkedIn post now by selecting the best 1-3 posts from the context above, and rephrasing them according to your persona guidelines.
        """

        response = self.client.chat.completions.create(
            model="gpt-5.4-mini",
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_message}
            ],
            temperature=0.7
        )
        return response.choices[0].message.content

if __name__ == "__main__":
    generator = PostGenerator()