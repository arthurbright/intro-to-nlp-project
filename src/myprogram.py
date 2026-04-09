#!/usr/bin/env python
import os
import string
import random
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
# from better_bpe import *
# from ngram import *
import pickle
# from tokenizers import Tokenizer, models, trainers
import math
from collections import defaultdict
# import pygtrie
from huggingface import *
# from transformers import AutoTokenizer
import torch
from lstm import *
import time
import torch.multiprocessing as mp
from parallel import *
import locale

VOCAB_SIZE = 1649


class MyModel:
    """
    This is a starter model to get you started. Feel free to modify this file.
    """

    def __init__(self):
        # model = CharLSTM(45) # TODO CHANGE
        # print("USING VOCAB SIZE", 45)
        # model.load_state_dict(torch.load("lstm/model.pth", weights_only=True))
        # model.eval()
        # self.model = model
        self.model = None
        pass

    @classmethod
    def preprocess_data(cls, line):
        line = line.rstrip('\n').lower()
        return line


    @classmethod
    def load_training_data(cls):
        # TODO 
        # open dev
        lines = []
        # with open('data/open-dev/input.txt', "r", encoding="utf-8") as f:
        #     for line in f:
        #         lines.append(cls.preprocess_data(line))
        # first half of test
        with open('data/test.txt', "r", encoding="utf-8") as f:
            for line in f:
                lines.append(cls.preprocess_data(line))

        # open dev contains approx 4 million chars
        # 6 million from open dev + test

        langs = ['en', 'ru', 'zh', 'ja', 'hi', 'ar', 'ko', 'fr', 'de', 'it']
        wiki_lines = []
        for lang in langs:
            _lines = download_dataset(("wikimedia/wikipedia", f"20231101.{lang}"), "text", 1_000_000) #og: 1_000_000
            wiki_lines.extend([cls.preprocess_data(l) for l in _lines])

        total_chars = sum([len(l) for l in lines])

        # prefix deduplication - increase variance
        lines = remove_prefixes(lines)
        lines = lines * 2

        wiki_lines = remove_prefixes(wiki_lines)
        
        print(f"load_training_data: loaded {total_chars} chars")
        print(f"After prefix dedupe and upsample: {sum([len(l) for l in lines])} chars")
        print(f"Added from wiki: {sum([len(w) for w in wiki_lines])} chars")

        return lines + wiki_lines

    @classmethod
    def load_test_data(cls, fname):
        data = []
        with open(fname, "r", encoding="utf-8") as f:
            for line in f:
                data.append(cls.preprocess_data(line))
        return data

    @classmethod
    def write_pred(cls, preds, fname, csv=False):
        
            with open(fname, 'wt', encoding='utf-8') as f:
                f.write("id,prediction\n")
                ii = 0
                for p in preds:
                    _s = f'{ii},{escape_csv(p)}\n'
                    # f.write(_s.encode(encoding=locale.getpreferredencoding()))
                    f.write(_s)
                    ii += 1
        
            with open(f"{fname}.flat", 'wt', encoding='utf-8') as f:
                for p in preds:
                    _s = '{}\n'.format(p)
                    # f.write(_s.encode(encoding=locale.getpreferredencoding()))
                    f.write(_s)

    def run_train(self, lines, work_dir):
        dataset = SentenceCharDataset(lines)
        self.model = CharLSTM(dataset.vocab_size)
        my_train(self.model, dataset, epochs=30)

        print("VOCAB_SIZE:", dataset.vocab_size)
        params = {"vocab_size": dataset.vocab_size,
                   "char2idx": dataset.char2idx,
                   "idx2char": dataset.idx2char}
        with open(f"{work_dir}/params.pickle", 'wb') as f:
            pickle.dump(params, f)
    
    def run_pred(self, data, work_dir):
        # data = data[:2048]
        start_time = time.time()
        res = run_parallel_inference(data, work_dir)
        print("time taken:", time.time() - start_time)
        print("avg time:", (time.time() - start_time)/len(data))
        res2 = [None for _ in range(len(res))]
        for idx, pred in res:
            res2[idx] = "".join(pred)
        # print(res2)
        return res2
        
        # preds = []  # list of strings
        # progress = 0
        # start_time = time.time()
        # for line in data:
        #    pred = self.predict_top3(line)
        #    preds.append("".join(pred))
        #    progress += 1
        #    if(progress % 100 == 0):
        #        print("Progress: ", progress, "Average: ", (time.time() - start_time)/progress)
        # return preds

    def save(self, work_dir):
        torch.save(self.model.state_dict(), f"{work_dir}/model.pth")

    @classmethod
    def load(cls, work_dir):
        with open(f"{work_dir}/params.pickle", 'rb') as f:
            params = pickle.load(f)
        vocab_size = params["vocab_size"]
        model = CharLSTM(vocab_size)
        print("USING VOCAB SIZE", vocab_size)
        model.load_state_dict(torch.load(f"{work_dir}/model.pth", weights_only=True))
        ret = MyModel()
        ret.model = model
        ret.vocab_size = params["vocab_size"]
        ret.char2idx = params["char2idx"]
        ret.idx2char = params["idx2char"]
        return ret

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
        # print('Loading model')
        # model = MyModel.load(args.work_dir)
        model = MyModel()
        print('Loading test data from {}'.format(args.test_data))
        test_data = MyModel.load_test_data(args.test_data)
        print('Making predictions')
        pred = model.run_pred(test_data, args.work_dir)
        print('Writing predictions to {}'.format(args.test_output))
        assert len(pred) == len(test_data), 'Expected {} predictions but got {}'.format(len(test_data), len(pred))
        model.write_pred(pred, args.test_output, csv=True)
    else:
        raise NotImplementedError('Unknown mode {}'.format(args.mode))
