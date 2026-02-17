from collections import defaultdict, Counter
from util import *

def train_ngram_model(corpus, n=3):
    """
    corpus: iterable of sentences (plaintext)
    n: n-gram size (3 = trigram)
    Returns: dict context -> dict char -> probability
    """
    counts = defaultdict(Counter)
    for sentence in corpus:
        # TODO: find a start of sentence token for n-gram.
        # convert to ints?
        padded = 0.to_bytes() * (n - 1) + sentence
        for i in range(len(sentence)):
            context = padded[i:i + n - 1]
            char = padded[i + n - 1]
            counts[context][char] += 1
    
    # Convert counts to probabilities
    probs = {}
    for context, counter in counts.items():
        total = sum(counter.values())
        probs[context] = {c: v / total for c, v in counter.items()}
    
    return probs