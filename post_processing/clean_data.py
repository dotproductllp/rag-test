import json, re, os

# Matches HTTP/HTTPS links and www links
URL_REGEX = re.compile(
    r'(?:https?://|www\.)\S+|'
    r'\b(?:[a-zA-Z0-9-]+\.)+(?:com|net|org|io|in|co|us|uk)'
    r'(?:/[^\s]*)?\b'
)

# Matches hashtags (e.g., #AI, #Web3) 
HASHTAG_SYMBOL_REGEX = re.compile(r'#(?=\w+)')

# Matches multiple whitespaces, newlines (\n), carriage returns (\r), tabs (\t)
WHITESPACE_REGEX = re.compile(r'\s+')

EMAIL_REGEX = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,7}\b')

# Standard UUID matcher (safe to remove explicitly)
UUID_REGEX = re.compile(r'\b[a-fA-F0-9]{8}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{12}\b')


def remove_garbage_alphanumerics(text: str) -> str:
    """
    Ratio Rule: Only removes strings if they are longer than 15 characters 
    AND consist of more than 40% digits. 
    Protects: "Web3Infrastructure", "BRICS2025", "B2B"
    Removes: "dmfn7GAN1234987654", "1771585794934"
    """
    words = text.split()
    clean_words = []
    
    for word in words:
        # Check if the word is long AND dense with numbers
        if len(word) > 15:
            digit_count = sum(c.isdigit() for c in word)
            if (digit_count / len(word)) > 0.4:
                clean_words.append('[ID_HASH]')
                continue
                
        clean_words.append(word)
        
    return " ".join(clean_words)

def clean_text(text: str) -> str:
    if not text or not isinstance(text, str):
        return text
        
    # Step 1: Clean literal escape sequence noise
    # Replaces weird formats
    text = text.replace('\\/', '/').replace('’',"'").replace('–','-').replace("\u200F", "").replace('\u2063','')
    # Step 2: Replace URLs with tag
    text = URL_REGEX.sub('[EXTERNAL_LINK]', text)
    
    # Step 3: Remove Hashtags from the body (since they are in the JSON array)
    text = HASHTAG_SYMBOL_REGEX.sub('', text)
    
    # Step 4: Remove exact UUIDs
    text = UUID_REGEX.sub('[ID_HASH]', text)
    
    # Step 5: Apply the Ratio Rule for remaining garbage strings
    text = remove_garbage_alphanumerics(text)
    
    # Step 6: Normalize all whitespaces (turns multiple spaces/newlines into a single space)
    text = WHITESPACE_REGEX.sub(' ', text)

    text = EMAIL_REGEX.sub('[EMAIL]', text)
    
    # Return stripped text
    return text.strip()


def process_data(input_path: str, output_path: str):
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    processed_count = 0
    
    print(f"Starting processing of {input_path}...")
    
    with open(input_path, 'r', encoding='utf-8') as infile, \
         open(output_path, 'w', encoding='utf-8') as outfile:
        
        for line in infile:
            if not line.strip():
                continue
                
            try:
                record = json.loads(line)
                
                # 1. Clean the main article body
                if record.get('article_body'):
                    record['article_body'] = clean_text(record['article_body'])
                
                # 2. Clean the comments
                if record.get('comments') and isinstance(record['comments'], list):
                    for comment in record['comments']:
                        if comment.get('body'):
                            comment['body'] = clean_text(comment['body'])
                            
                # Write the cleaned JSON object back to the new file
                outfile.write(json.dumps(record, ensure_ascii=False) + '\n')
                processed_count += 1
                
                # Progress tracker
                if processed_count % 10000 == 0:
                    print(f"Processed {processed_count} records...")
                    
            except json.JSONDecodeError:
                print("Warning: Skipped a malformed JSON line.")
                continue

    print(f"\n✅ Success! Cleaned {processed_count} records.")
    print(f"Saved output to: {output_path}")


if __name__ == "__main__":
    # Setup paths relative to the script location
    INPUT_FILE = "real_data/part-00000-cdd8d172-4e8c-4f51-a941-475ee8444692-c000.json"
    OUTPUT_FILE = "./cleaned_output.json"
    
    process_data(INPUT_FILE, OUTPUT_FILE)