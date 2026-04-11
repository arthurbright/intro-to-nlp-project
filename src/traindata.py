import pickle
from huggingface import *
from util import *

def preprocess_data(line):
        line = line.rstrip('\n').lower()
        return line
# open dev
lines = []
# with open('data/open-dev/input.txt', "r", encoding="utf-8") as f:
#     for line in f:
#         lines.append(cls.preprocess_data(line))


# first half of test
with open('data/test.txt', "r", encoding="utf-8") as f:
    for line in f:
        lines.append(preprocess_data(line))

# open dev contains approx 4 million chars
# 6 million from open dev + test

langs = {'en': 1, 'ru': 1, 
            'zh': 3, 'ja': 2, 'hi': 1, 'ar': 1, 'ko': 2, 'fr': 1, 'de': 1, 'it': 1}
wiki_lines = []
for lang in langs:
    _lines = download_dataset(("wikimedia/wikipedia", f"20231101.{lang}"), "text", langs[lang] * 10_000_000) #og: 1_000_000
    wiki_lines.extend([preprocess_data(l) for l in _lines])

total_chars = sum([len(l) for l in lines])

# prefix deduplication - increase variance
lines = remove_prefixes(lines)
lines = lines * 2

wiki_lines = remove_prefixes(wiki_lines)

print(f"load_training_data: loaded {total_chars} chars")
print(f"After prefix dedupe and upsample: {sum([len(l) for l in lines])} chars")
print(f"Added from wiki: {sum([len(w) for w in wiki_lines])} chars")

with open('train_data_10mil.pickle', 'wb') as f:
    pickle.dump(lines + wiki_lines, f)