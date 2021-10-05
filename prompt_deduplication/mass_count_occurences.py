# Modified from https://github.com/google-research/deduplicate-text-datasets/blob/master/scripts/count_occurances.py
import json
import os
import numpy as np

import argparse

from tqdm import tqdm


def cut_up(query, size=5):
    whitespaced = query.split()
    subsets = [" ".join(whitespaced[i:i+size]) for i in range(0, len(whitespaced) - size - 1, size)]
    return subsets + [subset.lower() for subset in subsets]


def count_occurences(suffix):
    return int((os.popen("../../deduplicate-text-datasets/target/debug/dedup_dataset count_occurances %s /tmp/fin" % (
        suffix)).read().strip().split("Number of times present: ")[-1]))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Count occurances of sequence.')
    parser.add_argument('--suffix', type=str, required=True)
    parser.add_argument('--queries_folder', type=str)
    parser.add_argument('--tokenize', action='store_true')
    parser.add_argument('--starting-size', type=int, default=20)

    args = parser.parse_args()

    flagged = {}

    for queries_file in os.listdir(args.queries_folder):
        with open(os.path.join(args.queries_folder, queries_file)) as f:
            queries = json.load(f)

        if queries is None:
            continue

        for q in tqdm(queries):
            if args.tokenize:
                from transformers import GPT2Tokenizer
                tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
                arr = np.array(tokenizer.encode(q), dtype=np.uint16).view(np.uint8).tobytes()
            else:
                arr = q.encode('utf-8')
            open("/tmp/fin","wb").write(arr)
            try:
                count = count_occurences(args.suffix)
            except ValueError:
                continue
            if count > 0:
                if queries_file in flagged:
                    flagged[queries_file].append(q)
                else:
                    flagged[queries_file] = [q]


    json.dump(flagged, open("flagged_dictionary.json", "w"), ensure_ascii=False, indent=2)
