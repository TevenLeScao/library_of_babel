import argparse
import multiprocessing
import pickle
from functools import partial
from time import time


def extract_ngrams(line, n):
    ids = line.split()
    return [tuple(ids[i:i + n]) for i in range(0, len(ids) - n + 1)]


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='c4 tokenizer args')
    parser.add_argument("--input-file", type=str)
    parser.add_argument("--output-file", type=str)
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--log-interval", type=int, default=100000)
    parser.add_argument("--num-workers", type=int, default=59)
    args = parser.parse_args()

    pool = multiprocessing.Pool(processes=args.num_workers)
    total_ngram_set = set()
    ngram_sets = []
    start_time = time()

    with open(args.input_file) as f:
        for i, ngrams in enumerate(pool.imap(partial(extract_ngrams, n=args.n), f)):
            ngram_sets.append(ngrams)
            if i > 0 and i % args.log_interval == 0:
                print(i, time() - start_time)

    total_ngram_dict = {ngram: index for index, ngram in enumerate(total_ngram_set)}
    with open(args.output_file, "wb") as g:
        pickle.dump(total_ngram_dict, g)
