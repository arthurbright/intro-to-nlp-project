from better_bpe import *

text = "こg"
tb = to_bytes(text)
t2 = from_bytes(tb + tb)
print(t2)


lines = []
with open('data/open-dev/input.txt') as f:
    for line in f:
        line = line.rstrip('\n')
        lines.append(to_bytes(line))

print(len(lines))
vocab, segmented = iterative_bpe(lines, vocab_limit=260, prune_th = 0.0, prune_freq = 100)
print(len(vocab))
print([v for v in vocab if len(v) > 1])


