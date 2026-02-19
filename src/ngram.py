from collections import defaultdict, Counter
from util import *

def train_ngram_model(corpus, n=3):
    """
    corpus: iterable of sentences (bytes)
    n: n-gram size
    Returns: probs[gram size][context][next]
    NOTE: gram_size = 2 means trigram.
    """

    counts = [defaultdict(Counter) for _ in range(n)]
    for sentence in corpus:
        for gram_size in range(n):
            padded = sentence
            for i in range(gram_size):
                padded = START.to_bytes() + padded
            for i in range(len(sentence)):
                context = padded[i:i + gram_size]
                char = padded[i + gram_size]
                counts[gram_size][context][char] += 1
    
    # Convert counts to probabilities
    probs = [{} for _ in range(n)]
    for gram_size in range(n):
        for context, counter in counts[gram_size].items():
            total = sum(counter.values())
            probs[gram_size][context] = {c: v / total for c, v in counter.items()}
        
    return probs