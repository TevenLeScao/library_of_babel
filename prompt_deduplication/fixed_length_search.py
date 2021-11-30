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


def fixed_length_search(top_level_query, suffix, substring_length):
    whitespaced = top_level_query.split()
    max_len = len(whitespaced)
    queries = [" ".join(whitespaced[i:i + substring_length]) for i in range(0, max_len, substring_length)]
    queries[-1] = " ".join(whitespaced[max(len(whitespaced) - substring_length, 0):len(whitespaced)])
    counts = [count_occurences(query, suffix, tokenize=args.tokenize) for query in queries]
    return any(counts), [query for i, query in enumerate(queries) if counts[i]]


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Count occurances of sequence.')
    parser.add_argument('--suffix', type=str, required=True)
    parser.add_argument('--queries_folder', type=str)
    parser.add_argument('--tokenize', action='store_true')
    parser.add_argument('--length', type=int, default=16)

    args = parser.parse_args()

    flagged_per_task = {}
    flagged_queries = {queries_file: [] for queries_file in os.listdir(args.queries_folder)}

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

        flagged = 0

        for q in tqdm(queries):
            match_found, matched_queries = fixed_length_search(q, args.suffix, args.length)
            if match_found:
                flagged += 1
                flagged_queries[queries_file].append(matched_queries)

        flagged_per_task[queries_file] = flagged
        print(f"flagged {flagged} prompts")

        os.makedirs(os.path.join(args.queries_folder, "flagged"), exist_ok=True)
        json.dump(flagged_per_task, open(os.path.join(args.queries_folder, "flagged", "flagged_per_task.json"), "w"),
                  ensure_ascii=False, indent=2)
        json.dump(flagged_queries, open(os.path.join(args.queries_folder, "flagged", "flagged_queries.json"), "w"),
                  ensure_ascii=False, indent=2)
