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
    def load_training_data(cls):
        LIMIT = 100000
        # for now, just read open dev
        lines = []
        with open('data/open-dev/input.txt') as f:
            for line in f:
                line = line.rstrip('\n')
                lines.append(TO_BYTES(line))
        # lines = lines[:LIMIT]
        print(f"load_training_data: loaded {len(lines)} lines")
        return lines

    @classmethod
    def load_test_data(cls, fname):
        data = []
        with open(fname) as f:
            for line in f:
                inp = line.rstrip('\n')  # the last character is a newline
                data.append(TO_BYTES(inp))
        return data

    @classmethod
    def write_pred(cls, preds, fname):
        with open(fname, 'wt') as f:
            for p in preds:
                f.write('{}\n'.format(p))

    def run_train(self, lines, work_dir):
        # run byte level BPE
        tokenizer = Tokenizer(models.BPE())
        # tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
        print("TODO: TUNE VOCAB SIZE")
        trainer = trainers.BpeTrainer(vocab_size=5_000)
        tokenizer.train_from_iterator(lines, trainer)
        self.tokenizer = tokenizer

        # get n-gram probabilities
        token_lines = [tokenizer.encode(line).tokens for line in lines]
        token_lines = [MyModel.pad_start_token(l) for l in token_lines]
        self.ngram_probs = train_ngram_model(token_lines, n=N)

    STUPID_BACKOFF = 0.4
    
    def score_token_sequence(self, tokens: list[str]):
        """ Returns log probability
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



    def run_pred_line(self, line: str):
        n = len(line)
        vocab = self.tokenizer.get_vocab()
        max_likelihood = float("-inf")
        for i in range(n + 1):
            prefix = line[:i]
            suffix = line[i:]

            encoded_prefix = self.tokenizer.encode(prefix).tokens
            prefix_score = self.score_token_sequence(encoded_prefix)
            # consider all token seqs that contain suffix
            # print(self.score_token_sequence(self.tokenizer.encode(line).tokens))
        print(max_likelihood)
        return ['a', 'b', 'c']

    def run_pred(self, data):
        assert self.ngram_probs is not None
        assert self.tokenizer is not None
        preds = []
        all_chars = string.ascii_letters
        for line in data:
            guesses = self.run_pred_line(line)
            preds.append(''.join(list(guesses)))
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
        model.write_pred(pred, args.test_output)
    else:
        raise NotImplementedError('Unknown mode {}'.format(args.mode))
