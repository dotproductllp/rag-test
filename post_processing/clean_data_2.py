import orjson, os, re
from tqdm import tqdm

_LATIN_ALPHA_RE = re.compile(r'[A-Za-z\xC0-\u024F]')  # latin basic + extended

# Matches HTTP/HTTPS links and www links
URL_REGEX = re.compile(
    r'(?:https?://|www\.)\S+|'
    r'\b(?:[a-zA-Z0-9-]+\.)+(?:com|net|org|io|in|co|us|uk|app|info|ai|dev)'
    r'(?:/[^\s]*)?\b'
)

# Matches hashtags (e.g., #AI, #Web3) 
HASHTAG_SYMBOL_REGEX = re.compile(r'#(?=\w+)')

# Matches multiple whitespaces, newlines (\n), carriage returns (\r), tabs (\t)
WHITESPACE_REGEX = re.compile(r'\s+')

EMAIL_REGEX = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,7}\b')

# Standard UUID matcher (safe to remove explicitly)
UUID_REGEX = re.compile(r'\b[a-fA-F0-9]{8}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{12}\b')

# tokens that are >15 chars
_GARBAGE_WORD_RE = re.compile(r'\b\w{16,}\b')

# removing some noted charachters that cause embediing issues.
BAD_CHARS_REGEX = re.compile(r'[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f-\x9f\ud800-\udfff]')

# Matches Zero-Width spaces and Byte Order Marks
ZERO_WIDTH_REGEX = re.compile(r'[\u200B-\u200D\uFEFF]')

def is_latin_post(text: str, threshold: float = 0.85) -> bool:
    total = len(text)
    latin = len(_LATIN_ALPHA_RE.findall(text))
    return (latin / total) >= threshold

# Translation table for single-char replacements (much faster than chained replace())
_TRANS = str.maketrans({
    '\u2019': "'",   # right single quote  →  apostrophe
    '\u2013': '-',   # en-dash 1           →  hyphen
    '\u2011' :'-',   # en-dash 2           →  hyphen
    '\u200F': '',    # right-to-left mark  →  remove
    '\u2063': '',    # invisible-separator →  remove
    '\u200E': '',    # invisible-separator →  remove
})

def _replace_garbage(m: re.Match) -> str:
    """
    Ratio Rule: Only removes strings if they are longer than 15 characters 
    AND consist of more than 40% digits. 
    Protects: "Web3Infrastructure", "BRICS2025", "B2B"
    Removes: "dmfn7GAN1234987654", "1771585794934"
    """
    word = m.group()
    digit_count = sum(c.isdigit() for c in word)
    return '[ID_HASH]' if (digit_count / len(word)) > 0.4 else word

def _build_fancy_unicode_map():
    ranges = [
        (0x1D400, 'A', 26), (0x1D41A, 'a', 26),  # Mathematical Bold
        (0x1D434, 'A', 26), (0x1D44E, 'a', 26),  # Mathematical Italic
        (0x1D468, 'A', 26), (0x1D482, 'a', 26),  # Mathematical Bold Italic
        (0x1D49C, 'A', 26), (0x1D4B6, 'a', 26),  # Mathematical Script (partial, skip gaps)
        (0x1D5D4, 'A', 26), (0x1D5EE, 'a', 26),  # Mathematical Sans-Serif Bold
        (0x1D608, 'A', 26), (0x1D622, 'a', 26),  # Mathematical Sans-Serif Italic
        (0x1D63C, 'A', 26), (0x1D656, 'a', 26),  # Mathematical Sans-Serif Bold Italic
        (0x1D670, 'A', 26), (0x1D68A, 'a', 26),  # Mathematical Monospace
        (0x1D7CE, '0', 10),                      # Mathematical Bold Digits
        (0x1D7CE, '0', 10),                      # Bold digits
        (0x1D7D8, '0', 10),                      # Double-struck digits
        (0x1D7E2, '0', 10),                      # Sans-serif digits
        (0x1D7EC, '0', 10),                      # Sans-serif bold digits
        (0x1D7F6, '0', 10),                      # Monospace digits
    ]
    table = {}
    for start, base_char, count in ranges:
        for i in range(count):
            table[start + i] = ord(base_char) + i
    return table

_FANCY_UNICODE_TABLE = _build_fancy_unicode_map()

def clean_text(text: str) -> str:

    # Step 0: Normalize Unicode bold/italic mathematical letters to ASCII
    text = text.translate(_FANCY_UNICODE_TABLE)

    # Step 1: Clean literal escape sequence noise
    # Replaces weird formats
    text = text.translate(_TRANS)
    # Step 2: Replace URLs with tag
    text = URL_REGEX.sub('[URL]', text)
    
    # Step 3: Remove Hashtags from the body (since they are in the JSON array)
    text = HASHTAG_SYMBOL_REGEX.sub('', text)
    
    # Step 4: Remove exact UUIDs
    text = UUID_REGEX.sub('[UUID]', text)
    
    # Step 5: Apply the Ratio Rule for remaining garbage strings
    text = _GARBAGE_WORD_RE.sub(_replace_garbage, text)
    
    # Step 6: Normalize all whitespaces (turns multiple spaces/newlines into a single space)
    text = WHITESPACE_REGEX.sub(' ', text)

    text = EMAIL_REGEX.sub('[EMAIL]', text)

    text = BAD_CHARS_REGEX.sub('', text)

    # Remove zero-width spaces (often causes OpenAI JSON parsing errors)
    text = ZERO_WIDTH_REGEX.sub('', text)
    
    # Return stripped text
    return text.strip()

def process_record(record: str):
    
    if not is_latin_post(record.get('article_body', '')):
        return None

    if record.get('article_body'):
        record['article_body'] = clean_text(record['article_body'])

    filtered_record = {
        "id": record.get("id"),
        "article_body": record.get("article_body"),
        "date_published" : record.get("date_published")
    }

    return filtered_record

def process_data(input_path: str, output_path: str):
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    print(f"Starting processing of {input_path}...")
    output = []
    with open(input_path, 'rb') as infile, open(output_path, 'wb') as outfile:
        records = orjson.loads(infile.read())
        for record in tqdm(records, desc="Cleaning"):
            data = process_record(record)
            if data:
                output.append(data)
        outfile.write(orjson.dumps(output))
    print(f"Saved output to: {output_path}")


if __name__ == "__main__":
    import cProfile, pstats, io

    INPUT_FILE = "../data/base.json"
    OUTPUT_FILE = "../data/cleaned_output.json"

    pr = cProfile.Profile()
    pr.enable()
    process_data(INPUT_FILE, OUTPUT_FILE)
    pr.disable()

    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats(pstats.SortKey.CUMULATIVE)
    ps.print_stats(20)
    print(s.getvalue())