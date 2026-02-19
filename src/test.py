from better_bpe import *
from ngram import *

# text = "こg"
# tb = to_bytes(text)
# print(tb)
# t2 = from_bytes(tb + tb)
# print(t2)
# exit()

# b = bytes([128 + 64 + 32 + 4 + 2, 128 + 32 + 1, 128 + 1])
# print(from_bytes(b))
# exit()

lines = [to_bytes("abcabd")]
print(train_ngram_model(lines, 3))

lines = []
with open('data/open-dev/input.txt') as f:
    for line in f:
        line = line.rstrip('\n')
        lines.append(to_bytes(line))

print(len(lines))
vocab, segmented = iterative_bpe(lines, vocab_limit=257, prune_th = 0.0, prune_freq = 100)
print(len(vocab))
print([v for v in vocab if len(v) > 1])


