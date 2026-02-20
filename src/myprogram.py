#!/usr/bin/env python
import os
import string
import random
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from better_bpe import *
from ngram import *
import pickle


N = 3

class MyModel:
    """
    This is a starter model to get you started. Feel free to modify this file.
    """
    ngram_probs = {}

    @classmethod
    def load_training_data(cls):
        LIMIT = 100
        # for now, just read open dev
        lines = []
        with open('data/open-dev/input.txt') as f:
            for line in f:
                line = line.rstrip('\n')
                lines.append(to_bytes(line))
        lines = lines[:LIMIT]
        print(f"load_training_data: loaded {len(lines)} lines")
        return lines

    @classmethod
    def load_test_data(cls, fname):
        data = []
        with open(fname) as f:
            for line in f:
                inp = line[:-1]  # the last character is a newline
                data.append(to_bytes(inp))
        return data

    @classmethod
    def write_pred(cls, preds, fname):
        with open(fname, 'wt') as f:
            for p in preds:
                f.write('{}\n'.format(p))

    def run_train(self, lines, work_dir):
        # run byte level BPE
        vocab, tokens = iterative_bpe(lines, vocab_limit=258, prune_th=0.8, prune_freq=100)
        # good vocab size: 30-50k??
        # get n-gram probabilities
        self.ngram_probs = train_ngram_model(tokens, n=N)

    # stupid backoff TODO do argmax instead of sampling
    BACKOFF_WEIGHT = 0.4
    def sample_next_byte(self, byte_line):
        for gram_size in range(N - 1, -1, -1):
            suffix = byte_line[-gram_size:]
            if suffix in self.ngram_probs[gram_size]:
                choices = list(self.ngram_probs[gram_size][suffix].keys())
                weights = list(self.ngram_probs[gram_size][suffix].values())
                return random.choices(choices, weights=weights, k=1)[0]
        raise "Should have returned; unigram distribution should always exist"

    def sample_next_char(self, byte_line):
        # assumes byte_line is already padded
        char_bytes = []
        b1 = self.sample_next_byte(byte_line)
        byte_line.append(b1)
        char_bytes.append(b1)
        if(ENCODING == 'utf-8'):
            if b1 < 128 + 64:
                remaining_bytes = 1
            elif b1 < 128 + 64 + 32:
                remaining_bytes = 2
            elif b1 < 128 + 64 + 32 + 16:
                remaining_bytes = 3
            else:
                remaining_bytes = 4
        else:
            raise "UNSPECIFIED ENCODING"
        
        for i in range(remaining_bytes):
            b = self.sample_next_byte(byte_line)
            byte_line.append(b)
            char_bytes.append(b)

        try:
            char = from_bytes(char_bytes)
            return char
        except:
            raise f"Illegal byte sequence: {char_bytes}"

    def run_pred(self, data):
        preds = []
        all_chars = string.ascii_letters
        for inp in data:
            byte_line = pad_start(inp)
            guesses = set()
            num_tries = 100
            while num_tries > 0 and len(guesses) < 3:
                num_tries -= 1
                try:
                    c = self.sample_next_char(byte_line.copy())
                except Exception as e:
                    print(f"Error sampling: {e}")
                
                guesses.insert(c)
            
            while len(guesses) < 3:
                guesses.add(random.choice(all_chars))

            preds.append(''.join(list(guesses)))
        return preds

    def save(self, work_dir):
        # pickle the probs dict
        with open(os.path.join(work_dir, 'model.checkpoint'), 'wb') as f:
            pickle.dump(self.ngram_probs, f)

    @classmethod
    def load(cls, work_dir):
        # read the pickle
        with open(os.path.join(work_dir, 'model.checkpoint'), 'rb') as f:
            probs = pickle.load(f)
        return MyModel(ngram_probs=probs)


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
        model = MyModel()
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
