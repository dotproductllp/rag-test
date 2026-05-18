import orjson

input_file = "../data/base.ndjson"
output_file = "../data/out.json"

with open(input_file, "rb") as infile, open(output_file, "wb") as outfile:
    records = [orjson.loads(line) for line in infile if line.strip()]
    outfile.write(orjson.dumps(records, option=orjson.OPT_INDENT_2))