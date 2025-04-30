from transformers import MT5Tokenizer
import json

# استفاده از همان tokenizer مورد نظر شما
model_name = "google/mt5-small"
tokenizer = MT5Tokenizer.from_pretrained(model_name)

input_file = "source/karbord-moshavere-tahsili-shoghli.jsonl"
output_file = "source/long_inputs.jsonl"

with open(input_file, "r", encoding="utf-8") as infile, open(output_file, "w", encoding="utf-8") as outfile:
    for line in infile:
        data = json.loads(line)
        input_text = "خلاصه کن: " + data["input"]
        tokenized = tokenizer.tokenize(input_text)
        token_count = len(tokenized)
        if token_count > 512:
            annotated_line = f"{token_count} tokens | " + json.dumps(data, ensure_ascii=False)
            outfile.write(annotated_line + "\n")
