from collections import defaultdict, Counter
import time
from util import *
import pickle

# ---------------------------
# Greedy tokenizer for a single line
# ---------------------------
def tokenize_greedy_line(text, vocab):
    """
    Greedy longest-match tokenization for a single line. Text = seq of bytes
    """
    tokens = []
    i = 0
    n = len(text)

    # Group tokens by length
    by_len = defaultdict(set)
    for t in vocab:
        by_len[len(t)].add(t)
    lengths = sorted(by_len.keys(), reverse=True)

    while i < n:
        matched = False
        for L in lengths:
            if i + L <= n and text[i:i+L] in by_len[L]:
                tokens.append(text[i:i+L])
                i += L
                matched = True
                break
        if not matched:
            tokens.append(text[i])
            i += 1
    return tokens


# ---------------------------
# Segment corpus line by line
# ---------------------------
def segment_corpus_greedy_lines(byte_lines, vocab):
    """
    Segment the corpus line by line into list of tokens
    """
    segmented = []
    for line in byte_lines:
        if len(line) == 0: continue
        segmented.append(tokenize_greedy_line(line, vocab))
    return segmented


def count_pairs_lines(byte_lines, vocab):
    """
    Count adjacent token pairs within each line
    """
    pairs = Counter()
    token_counts = Counter()
    for line in byte_lines:
        if not line:
            continue
        tokens = tokenize_greedy_line(line, vocab)
        for i in range(len(tokens) - 1):
            pair = (tokens[i], tokens[i + 1])
            pairs[pair] += 1
        for token in tokens:
            token_counts[token] += 1
    return pairs, token_counts


def iterative_bpe(
    byte_lines, vocab_limit=1000, min_pair_freq=2, initial_vocab = set(), prune_th = 0.8, prune_freq = 10
):
    # start with all possible bytes
    vocab = set([i.to_bytes() for i in range(256)])
    print(f"Initial vocab size: {len(vocab)}")

    segmented = segment_corpus_greedy_lines(byte_lines, vocab)

    iteration = 0
    while len(vocab) < vocab_limit:
        iteration += 1

        # Count pairs within lines only
        pairs, token_counts = count_pairs_lines(byte_lines, vocab)

        # Prune rare pairs
        pairs = {pair: freq for pair, freq in pairs.items() if freq >= min_pair_freq}

        if not pairs:
            print("No more pairs to merge")
            break

        # Skip pairs containing newline
        pairs = {pair: freq for pair, freq in pairs.items()}

        if not pairs:
            print("No more newline-safe pairs to merge")
            break

        # Choose most frequent pair
        best_pair = max(pairs.items(), key=lambda x: x[1])[0]

        # Merge: create new token
        new_token = best_pair[0] + best_pair[1]

        best_pair_count = pairs[best_pair]
        a_count = token_counts[best_pair[0]]
        b_count = token_counts[best_pair[1]]

        TH = prune_th
        if a_count - best_pair_count <= TH * best_pair_count and len(best_pair[0]) > 1:
            vocab.remove(best_pair[0])
            print("Removed token:", best_pair[0])
        if b_count - best_pair_count <= TH * best_pair_count and len(best_pair[1]) > 1:
            vocab.remove(best_pair[1])
            print("Removed token:", best_pair[1])


        vocab.add(new_token)

        # combine tokens that are mostly seen together
        if iteration % prune_freq == 0:
            for pair in pairs:
                together = pairs[pair]
                a_count = token_counts[pair[0]]
                b_count = token_counts[pair[1]]
                if a_count - together <= TH * together and len(pair[0]) > 1 and pair[0] in vocab:
                    vocab.remove(pair[0])
                    vocab.add(pair[0] + pair[1])
                    print("Merged token:", pair[0])
                if b_count - together <= TH * together and len(pair[1]) > 1 and pair[1] in vocab:
                    vocab.remove(pair[1])
                    vocab.add(pair[0] + pair[1])
                    print("Merged token:", pair[1])

        # Update segmentation
        segmented = segment_corpus_greedy_lines(byte_lines, vocab)
        total_tokens = sum([len(s) for s in segmented])
        print(
            f"Iter {iteration:03d} | Added token: '{new_token}' | vocab size={len(vocab)} | tokens={total_tokens}"
        )

        if len(vocab) >= vocab_limit:
            break

    return vocab, segmented
