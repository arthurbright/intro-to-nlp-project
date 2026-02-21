from collections import defaultdict
import math

def score_token_sequence(ngram_probs, tokens: list[str]):
        """ Returns log probability
        Tokens: UNPADDED sequence
        """
        # translate to bytes
        # tokens = self.tokenizer.decode([self.tokenizer.token_to_id(t) for t in tokens])
        # tokens = [to_bytes(t) for t in tokens]
        tokens = MyModel.pad_start_token(tokens)
        n = len(tokens)
        acc = 0.0 # log probability
        for i in range(N - 1, n):
            cur_token = tokens[i]

            found = False
            for ngram_size in range(N - 1, -1, -1):
                gram = tuple(tokens[i - ngram_size: i])
                if (gram in self.ngram_probs[ngram_size]
                    and cur_token in self.ngram_probs[ngram_size][gram]):
                    acc += math.log(self.ngram_probs[ngram_size][gram][cur_token])
                    found = True
                    break
                acc += math.log(self.STUPID_BACKOFF)
            if not found:
                # if code gets here, then the unigram is never used
                print(f"WARNING: found token that is never used: {cur_token}")
                return float("-inf")
        return acc

def best_tokenization(s, vocab, n, score_fn):
    """
    s: input string
    vocab: set of valid tokens
    n: n-gram order
    score_fn: function(prev_tokens_tuple, token) -> log-probability
    
    Returns: best list of tokens
    """
    
    # dp[pos][history] = (score, previous_state)
    dp = [defaultdict(lambda: (-math.inf, None)) for _ in range(len(s) + 1)]
    
    # Start with empty history
    start_history = tuple(["<s>"] * (n - 1))
    dp[0][start_history] = (0.0, None)
    
    for i in range(len(s)):
        for history, (score, backpointer) in dp[i].items():
            
            if score == -math.inf:
                continue
            
            # Try all tokens that match starting at position i
            for token in vocab:
                if s.startswith(token, i):
                    j = i + len(token)
                    
                    new_score = score + score_fn(history, token)
                    
                    new_history = (history + (token,))[-(n - 1):]
                    
                    if new_score > dp[j][new_history][0]:
                        dp[j][new_history] = (new_score, (i, history, token))
    
    # Find best ending state
    end_pos = len(s)
    best_score = -math.inf
    best_state = None
    
    for history, (score, backpointer) in dp[end_pos].items():
        if score > best_score:
            best_score = score
            best_state = (end_pos, history)
    
    if best_state is None:
        return None  # no valid tokenization
    
    # Backtrack
    tokens = []
    pos, history = best_state
    
    while pos > 0:
        score, backpointer = dp[pos][history]
        prev_pos, prev_history, token = backpointer
        tokens.append(token)
        pos = prev_pos
        history = prev_history
    
    tokens.reverse()
    return tokens