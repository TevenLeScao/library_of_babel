import argparse
import json
import os
from os.path import isdir
import warnings

import numpy as np
from tqdm import tqdm


def count_occurences(q, suffix, tokenize=False):
    if tokenize:
        from transformers import GPT2Tokenizer
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        arr = np.array(tokenizer.encode(q), dtype=np.uint16).view(np.uint8).tobytes()
    else:
        arr = q.encode('utf-8')
    open("/tmp/fin", "wb").write(arr)
    counted = False
    tries = 0
    while not counted:
        try:
            count = int(
                (os.popen("../../deduplicate-text-datasets/target/debug/dedup_dataset count_occurances %s /tmp/fin" % (
                    suffix)).read().strip().split("Number of times present: ")[-1]))
            counted = True
        except ValueError:
            tries += 1
            if tries == 5:
                count = 0
                warnings.warn(f"Failed to count query {q}")
                break
    return count


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Count occurances of sequence.')
    parser.add_argument('--suffix', type=str, required=True)
    parser.add_argument('--queries_folder', type=str)
    parser.add_argument('--tokenize', action='store_true')

    args = parser.parse_args()
    counted_queries = {queries_file: [] for queries_file in os.listdir(args.queries_folder)}

    for queries_file in os.listdir(args.queries_folder):
        if isdir(os.path.join(args.queries_folder, queries_file)):
            continue
        print(queries_file)
        with open(os.path.join(args.queries_folder, queries_file)) as f:
            if queries_file.split(".")[-1] == "json":
                queries = json.load(f)
            elif queries_file.split(".")[-1] == "txt":
                queries = f.readlines()
            else:
                raise NotImplementedError(f"File extension {queries_file.split('.')[-1]} not supported")

        if queries is None:
            continue

        for query in tqdm(queries):
            query_count = count_occurences(query, args.suffix)
            counted_queries[queries_file].append(query, query_count)

        json.dump(counted_queries, open(os.path.join(args.queries_folder, "counts", "counts.json"), "w"),
                  ensure_ascii=False, indent=2)
