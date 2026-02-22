from better_bpe import *
from ngram import *
from tokenizers.pre_tokenizers import ByteLevel
import pickle
# text = "こg"
# tb = to_bytes(text)
# print(tb)
# t2 = from_bytes(tb + tb)
# print(t2)
# exit()

# b = bytes([128 + 64 + 32 + 4 + 2, 128 + 32 + 1, 128 + 1])
# print(from_bytes(b))
# exit()

def comma(line):
    return f'"{line.replace('"', '""')}"'
with open('pred2.txt', 'w') as f2:
    f2.write('id,prediction\n')
    ii = 0
    with open('tmp/33900.pickle', 'rb') as f:
        lines = pickle.load(f)
        for line in lines:
            f2.write(f"{ii},{comma(line)}\n")
            ii += 1
    with open('tmp/52000.pickle', 'rb') as f:
        lines = pickle.load(f)
        for line in lines:
            f2.write(f"{ii},{comma(line)}\n")
            ii += 1
    with open('tmp/56000.pickle', 'rb') as f:
        lines = pickle.load(f)
        for line in lines:
            f2.write(f"{ii},{comma(line)}\n")
            ii += 1
    with open('tmp/FINAL.pickle', 'rb') as f:
        lines = pickle.load(f)
        for line in lines:
            f2.write(f"{ii},{comma(line)}\n")
            ii += 1
    print(ii)
        
exit()

arr = "".join([BYTE_TO_CHAR[i] for i in range(256)])
# print(len(set(arr))
print(TO_BYTES(' '))
exit()

lines = [to_bytes("abcabd")]

lines = []
with open('data/open-dev/input.txt') as f:
    for line in f:
        line = line.rstrip('\n')
        lines.append(to_bytes(line))

print(len(lines))
# vocab, segmented = iterative_bpe(lines, vocab_limit=257, prune_th = 0.0, prune_freq = 100)
from tokenizers import Tokenizer, models, trainers, pre_tokenizers

tokenizer = Tokenizer(models.BPE())
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()

trainer = trainers.BpeTrainer(vocab_size=5_000, special_tokens=["<s>"])
tokenizer.train(["data/open-dev/input.txt"], trainer)
# tokenizer.save("work/tokenizer.json")

vocab = tokenizer.get_vocab()

# Print first 50 tokens
for token, idx in list(vocab.items())[:50]:
    print(idx, repr(token))

print(tokenizer.encode("H H").tokens)

# print(len(vocab))
# print([v for v in vocab if len(v) > 1])

# print(train_ngram_model(segmented, 3))
exit()




