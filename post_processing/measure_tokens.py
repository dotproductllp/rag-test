import tiktoken, json

# Read JSON from a file instead of embedding
with open("post.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Extract article_body and comments
article_body = data["article_body"]
comments = " ".join([c["body"] for c in data.get("comments", [])])

full_text = article_body + "\n\n" + comments

# Count tokens
enc = tiktoken.encoding_for_model("text-embedding-3-small")
tokens = enc.encode(full_text)
print("Number of tokens:", len(tokens))