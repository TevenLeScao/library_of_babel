# Modified from https://github.com/google-research/deduplicate-text-datasets/blob/master/scripts/count_occurances.py
import argparse
import json
import math
import os
import warnings

import numpy as np
from tqdm import tqdm


def power_log(x):
    return 2**(math.ceil(math.log(x, 2)))


def count_occurences(q, suffix):
    if args.tokenize:
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
            count = int((os.popen("../../deduplicate-text-datasets/target/debug/dedup_dataset count_occurances %s /tmp/fin" % (
                suffix)).read().strip().split("Number of times present: ")[-1]))
            counted = True
        except ValueError:
            tries += 1
            if tries == 5:
                count = 0
                warnings.warn(f"Failed to count query {q}")
                break
    return count


def longest_sequence_approximate(top_level_query, suffix, smallest_length):
    whitespaced = top_level_query.split()
    current_scale = power_log(len(whitespaced))
    max_len = len(whitespaced)
    while current_scale >= smallest_length:
        queries = [" ".join(whitespaced[i:i+current_scale]) for i in range(0, max_len, current_scale)]
        # print(queries)
        counts = [count_occurences(query, suffix) for query in queries]
        if any(counts):
            return 3 * current_scale
        else:
            current_scale = current_scale // 2
    return smallest_length


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Count occurances of sequence.')
    parser.add_argument('--suffix', type=str, required=True)
    parser.add_argument('--queries_folder', type=str)
    parser.add_argument('--tokenize', action='store_true')
    parser.add_argument('--lower_bound', type=int, default=8)

    args = parser.parse_args()

    flagged_queries = {}
    flagged_per_task = {}

    for queries_file in os.listdir(args.queries_folder):
        print(queries_file)
        with open(os.path.join(args.queries_folder, queries_file)) as f:
            queries = json.load(f)

        if queries is None:
            continue

        for q in tqdm(queries):
            longest_sequence = longest_sequence_approximate(q, args.suffix, args.lower_bound)
            if longest_sequence > args.lower_bound:
                if queries_file in flagged_queries:
                    flagged_queries[queries_file][q] = longest_sequence
                else:
                    flagged_queries[queries_file] = {q: longest_sequence}

        flagged_per_task[queries_file] = len(flagged_queries[queries_file])
        print(f"flagged {len(flagged_queries[queries_file])} prompts")

        json.dump(flagged_queries, open("flagged_queries.json", "w"), ensure_ascii=False, indent=2)
        json.dump(flagged_per_task, open("flagged_per_task.json", "w"), ensure_ascii=False, indent=2)
