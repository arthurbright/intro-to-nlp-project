#!/usr/bin/env python
import os
import string
import random
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from better_bpe import *
from ngram import *
import pickle
from tokenizers import Tokenizer, models, trainers
import math
from collections import defaultdict
import pygtrie

N = 3

class MyModel:
    """
    This is a starter model to get you started. Feel free to modify this file.
    """
    @classmethod
    def pad_start_token(cls, tokens):
        return ["<s>" for _ in range(N - 1)] + tokens

    def __init__(self, ngram_probs, tokenizer):
        self.ngram_probs = ngram_probs
        self.tokenizer = tokenizer

    @classmethod
    def preprocess_data(cls, line):
        line = line.rstrip('\n').lower()
        return TO_BYTES(line)

    @classmethod
    def load_training_data(cls):
        LIMIT = 100000
        # for now, just read open dev
        lines = []
        with open('data/open-dev/input.txt') as f:
            for line in f:
                lines.append(cls.preprocess_data(line))
        print(f"load_training_data: loaded {len(lines)} lines")
        return lines

    @classmethod
    def load_test_data(cls, fname):
        data = []
        with open(fname) as f:
            for line in f:
                data.append(cls.preprocess_data(line))
        return data

    @classmethod
    def write_pred(cls, preds, fname, csv=False):
        if csv:
            with open(fname, 'wt') as f:
                f.write("id,prediction\n")
                ii = 0
                for p in preds:
                    f.write(f'{ii},{escape_csv(p)}\n')
                    ii += 1
        else:
            with open(fname, 'wt') as f:
                for p in preds:
                    f.write('{}\n'.format(escape_csv(p)))

    def run_train(self, lines, work_dir):
        # run byte level BPE
        tokenizer = Tokenizer(models.BPE())
        # tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
        print("TODO: TUNE VOCAB SIZE")
        trainer = trainers.BpeTrainer(vocab_size=30_000)
        tokenizer.train_from_iterator(lines, trainer)
        self.tokenizer = tokenizer

        # get n-gram probabilities
        token_lines = [tokenizer.encode(line).tokens for line in lines]
        token_lines = [MyModel.pad_start_token(l) for l in token_lines]
        self.ngram_probs = train_ngram_model(token_lines, n=N)

    STUPID_BACKOFF = 0.4

    def score_gram(self, context, token):
        assert len(context) == N - 1
        acc = 0
        for ngram_size in range(N - 1, -1, -1):
            gram = tuple(context[N - 1 - ngram_size:])
            if (gram in self.ngram_probs[ngram_size]
                and token in self.ngram_probs[ngram_size][gram]):
                return acc + math.log(self.ngram_probs[ngram_size][gram][token])
            acc += math.log(self.STUPID_BACKOFF)
        
        # if code gets here, then the unigram is never used
        # print(f"WARNING: found token that is never used: {token}")
        return float("-inf")
    
    def score_last_tokens(self, tokens, tokens2):
        acc = 0.0
        tokens3 = tokens + tokens2
        k = len(tokens)
        for i in range(len(tokens2)):
            acc += self.score_gram(tokens3[(k + i) - (N - 1):k + i], tokens2[i])
        return acc
        
    def score_token_sequence(self, tokens: list[str]):
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
            acc += self.score_gram(tokens[i - (N - 1): i], tokens[i])
        return acc

    # viterbi DP, with help from chatgpt
    def best_tokenization(self, s, n, score_fn):
        """
        s: input string
        n: n-gram order
        score_fn: function(prev_tokens_tuple, token) -> log-probability
        
        Returns: best list of tokens
        """
        vocab = set(self.tokenizer.get_vocab().keys())
        
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
            return None, float("-inf")  # no valid tokenization
        
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
        return tokens, best_score
    
    def contains_utf8_char(self, s, ind):
        """ Return None if too short, False if invalid, char if valid"""
        bs = [CHAR_TO_BYTE[s[i]] for i in range(ind, min(ind + 4, len(s)))]
        b1 = bs[0]
        if b1 < 128 + 64:
            num_bytes = 1
        elif b1 < 128 + 64 + 32:
            num_bytes = 2
        elif b1 < 128 + 64 + 32 + 16:
            num_bytes = 3
        else:
            num_bytes = 4
        
        if len(bs) < num_bytes: return None
        try:
            res = bytes(bs[:num_bytes]).decode("utf-8")
            assert len(res) == 1
            return res
        except:
            return False
    
    dfs_count = 0

    def dfs(self, cur_token):
        self.dfs_count += 1
        # if self.dfs_count > 40_000: return
        self._token_stack.append(cur_token)
        # assumign utf-8 (4 bytes max)
        token_stack_joined = "".join(self._token_stack)
        char = self.contains_utf8_char(token_stack_joined, self._p)
        if char is not None: 
            if not (char == False):
                self._probs[char] += math.exp(self.score_last_tokens(self._prev_tokens, self._token_stack))
            # terminate
            self._token_stack.pop()
            return
        
        # selectively dfs based on start of char
        needed = num_trailing_middle_bytes_needed(token_stack_joined)
        assert needed > 0
        # only consider tokens that follow
        if (cur_token,) in self.ngram_probs[1]:
            for next_token in self.ngram_probs[1][(cur_token,)]:
                self.dfs(next_token)

        # old approach: only consider legal tokens according to utf-8
        # for i in range(1, min(needed, self.max_type) + 1):
        #     for next_token in self.vocab_classes[i + 10]:
        #         self.dfs(next_token)
        # for next_token in self.vocab_classes[-needed]:
        #     self.dfs(next_token)

        self._token_stack.pop()
        return

    
    # Find next char distribution given prefix
    def sum_over_next_chars(self, tokens: list[str], prefix: str):
        """ Tokens is UNPADDED"""
        probs = defaultdict(float)
        p = len(prefix)
        tokens = self.pad_start_token(tokens)

        if len(prefix) == 0:
            matching_tokens = self.char_starts
        else:
            try:
                matching_tokens = list(self.vocab_trie.keys(prefix=prefix))
            except:
                # no matching prefix
                return probs


        for first_token in matching_tokens:
            if len(first_token) <= p: continue
            ty = bytes_type(first_token[p])
            if ty <= 0 or ty >= 10: continue # must be a start of char byte
            self._token_stack = []
            self._prev_tokens = tokens
            self._probs = probs
            self._p = p
            self.dfs(first_token)
        return probs



    def run_pred_line(self, line: str):
        n = len(line)
        probs = defaultdict(float)

        for i in range(n + 1):
            prefix = line[:i]
            suffix = line[i:]

            # naive encoding. Optimal score encoding is commented out below.
            encoded_prefix = self.tokenizer.encode(prefix).tokens
            prefix_score = self.score_token_sequence(encoded_prefix)
            if math.isinf(prefix_score): continue
            prefix_score = math.exp(prefix_score)
            # encoded_prefix, prefix_score  = self.best_tokenization(prefix, N, self.score_gram)
            # if encoded_prefix is not None:

            # consider all tokens that STRICTLY contain suffix
            suffix_probs = self.sum_over_next_chars(encoded_prefix, suffix)
            for char, p in suffix_probs.items():
                probs[char] += prefix_score * p

        top3 = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:3]
        return [pair[0] for pair in top3]

    def run_pred(self, data):
        assert self.ngram_probs is not None
        assert self.tokenizer is not None
        preds = []
        data = data[56000:]
        print(len(data))
        ii = 56000
        for line in data:
            guesses = self.run_pred_line(line)
            if ii % 200 == 0:
                print(f"Prediction {ii}: {guesses}")
            # print(self.dfs_count)
            self.dfs_count = 0

            if len(guesses) < 3:
                print(f"EMPTY GUESSES: {FROM_BYTES(line)}")
                guesses = ['a', 'b', 'c']
            preds.append(''.join(list(guesses)))
            ii += 1

            if ii % 4000 == 0:
                with open(f"tmp/{ii}.pickle", 'wb') as f:
                    pickle.dump(preds, f)
        with open(f"tmp/FINAL.pickle", 'wb') as f:
                    pickle.dump(preds, f)
        return preds

    def save(self, work_dir):
        # pickle the probs dict
        with open(os.path.join(work_dir, 'model.checkpoint'), 'wb') as f:
            pickle.dump(self.ngram_probs, f)
        self.tokenizer.save(os.path.join(work_dir, 'tokenizer.json'))

    @classmethod
    def load(cls, work_dir):
        # read the pickle
        with open(os.path.join(work_dir, 'model.checkpoint'), 'rb') as f:
            probs = pickle.load(f)
        m =  MyModel(ngram_probs=probs, 
                    tokenizer=Tokenizer.from_file(os.path.join(work_dir, 'tokenizer.json')))
        
        # precompute token vocab
        m.vocab = set(m.tokenizer.get_vocab().keys())
        m.vocab_classes = defaultdict(set)
        m.max_type = 0
        m.char_starts = []
        for v in m.vocab:
            ty = bytes_type(v)
            m.vocab_classes[ty].add(v)
            m.max_type = max(m.max_type, ty)
            if 1 <= ty <= 4:
                m.char_starts.append(v)
        for k in m.vocab_classes:
            print(k, len(m.vocab_classes[k]))

        # trie for fast prefix lookup
        m.vocab_trie = pygtrie.CharTrie()
        for v in m.vocab:
            m.vocab_trie[v] = True

        return m

if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('mode', choices=('train', 'test'), help='what to run')
    parser.add_argument('--work_dir', help='where to save', default='work')
    parser.add_argument('--test_data', help='path to test data', default='example/input.txt')
    parser.add_argument('--test_output', help='path to write test predictions', default='pred.txt')
    args = parser.parse_args()

    random.seed(0)

    if args.mode == 'train':
        if not os.path.isdir(args.work_dir):
            print('Making working directory {}'.format(args.work_dir))
            os.makedirs(args.work_dir)
        print('Instatiating model')
        model = MyModel(ngram_probs=None, tokenizer=None)
        print('Loading training data')
        train_data = MyModel.load_training_data()
        print('Training')
        model.run_train(train_data, args.work_dir)
        print('Saving model')
        model.save(args.work_dir)
    elif args.mode == 'test':
        print('Loading model')
        model = MyModel.load(args.work_dir)
        print('Loading test data from {}'.format(args.test_data))
        test_data = MyModel.load_test_data(args.test_data)
        print('Making predictions')
        pred = model.run_pred(test_data)
        print('Writing predictions to {}'.format(args.test_output))
        assert len(pred) == len(test_data), 'Expected {} predictions but got {}'.format(len(test_data), len(pred))
        model.write_pred(pred, args.test_output, csv=True)
    else:
        raise NotImplementedError('Unknown mode {}'.format(args.mode))
