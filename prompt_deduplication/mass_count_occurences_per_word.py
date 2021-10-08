# Modified from https://github.com/google-research/deduplicate-text-datasets/blob/master/scripts/count_occurances.py
import argparse
import json
import math
import os

from tqdm import tqdm

from utils import count_occurences


def power_log(x):
    return 2**(math.ceil(math.log(x, 2)))


def max_possible_length(index, queries, current_scale):
    if index == 0:
        # start of sequence can only grow one way
        return 2 * current_scale
    elif index == len(queries) - 1:
        # end of sequence might be shorter than current_scale
        return 2 * len(queries[-1].split())
    else:
        # middle of sequence can grow both ways
        return 3 * current_scale


def longest_sequence_approximate(top_level_query, suffix, smallest_length):
    whitespaced = top_level_query.split()
    current_scale = power_log(len(whitespaced))
    max_len = len(whitespaced)
    while current_scale >= smallest_length:
        queries = [" ".join(whitespaced[i:i+current_scale]) for i in range(0, max_len, current_scale)]
        # print(queries)
        counts = [count_occurences(query, suffix, tokenize=args.tokenize) for query in queries]
        if any(counts):
            possible_lengths = [max_possible_length(i, queries, current_scale) for i, count in enumerate(counts) if count]
            output = max(possible_lengths)
            if output > smallest_length:
                # then we've found a real chunk
                return output
            else:
                # then it means the rump end of the sequence triggered it, so we're removng it
                whitespaced = whitespaced[:power_log(len(whitespaced)) // 2]
                max_len = len(whitespaced)
        else:
            current_scale = current_scale // 2
    return smallest_length


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Count occurances of sequence.')
    parser.add_argument('--suffix', type=str, required=True)
    parser.add_argument('--queries_folder', type=str)
    parser.add_argument('--tokenize', action='store_true')
    parser.add_argument('--lower_bound', type=int, default=16)

    args = parser.parse_args()

    flagged_queries = {queries_file: {} for queries_file in os.listdir(args.queries_folder)}
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
                flagged_queries[queries_file][q] = (longest_sequence, len(q.split()))

        flagged_per_task[queries_file] = len(flagged_queries[queries_file])
        print(f"flagged {len(flagged_queries[queries_file])} prompts")

        json.dump(flagged_queries, open("flagged_queries.json", "w"), ensure_ascii=False, indent=2)
        json.dump(flagged_per_task, open("flagged_per_task.json", "w"), ensure_ascii=False, indent=2)
