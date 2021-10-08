import argparse
import json
import os

import numpy as np
from tqdm import tqdm
import random
from collections import OrderedDict

from utils import count_occurences


def build_reverse(word, **kwargs):
    return {word[::-1]}


def shuffle_string(string):
    chars = list(string)
    random.shuffle(chars)
    return ''.join(chars)


# this may return less than n_outputs - we're using a set - for example if there's less possibilities than n_outputs
def build_anagram_hard(word, n_outputs=5, **kwargs):
    return {word[:1] + shuffle_string(word[1:-1]) + word[-1:] for _ in range(n_outputs)}


# this may return less than n_outputs - we're using a set - for example if there's less possibilities than n_outputs
def build_anagram_easy(word, n_outputs=5, **kwargs):
    return {word[:2] + shuffle_string(word[2:-2]) + word[-2:] for _ in range(n_outputs)}


def build_cycles(word, **kwargs):
    return {word[i:len(word)] + word[:i] for i in range(len(word))}


# this may return less than n_outputs - we're using a set - for example if there's less possibilities than n_outputs
def build_insertion(word, chars=None, n_outputs=5, **kwargs):
    if chars is None:
        chars = [",", ".", "!", "/", "\\", ":", ";", " "]
    return {"".join(i + chars[random.randint(0, len(chars)-1)] for i in word) for _ in range(n_outputs)}


if __name__ == "__main__":

    transfos = OrderedDict(
        [("reverse", build_reverse), ("anagram_hard", build_anagram_hard), ("anagram_easy", build_anagram_easy),
         ("insertion", build_insertion), ("cycles", build_cycles)])
    parser = argparse.ArgumentParser(description='c4 tokenizer args')
    parser.add_argument("--words_to_test", type=str)
    parser.add_argument("--output_folder", type=str)
    parser.add_argument('--suffix', type=str, required=True)
    parser.add_argument('--tokenize', action='store_true')
    parser.add_argument("--sanity", action="store_true")
    parser.add_argument("--n_outputs", type=int, default=5)
    for transfo_name in transfos:
        parser.add_argument(f"--{transfo_name}", action="store_true")
    args = parser.parse_args()

    transfos = {name: fn for name, fn in transfos.items() if vars(args).get(name)}

    with open(args.words_to_test) as f:
        test_words = json.load(f)
    if args.sanity:
        test_words = test_words[:10]

    os.makedirs(args.output_folder, exist_ok=True)

    for tr_name, tr_fn in transfos.items():
        print(f"querying for {tr_name} transform")
        tr_counts = {}
        for word in tqdm(test_words):
            counts = {query: count_occurences(query, args.suffix, args.tokenize) for query in
                      tr_fn(word, n_outputs=args.n_outputs) if query != word}
            tr_counts[word] = counts
        average_count = np.mean([count for word_count in tr_counts.values() for count in word_count.values()])
        json.dump(tr_counts, open(os.path.join(args.output_folder, f"{tr_name}_per_word_count.json"), "w"), ensure_ascii=False, indent=2)
        json.dump(average_count, open(os.path.join(args.output_folder, f"{tr_name}_avg_count.json"), "w"), ensure_ascii=False, indent=2)

