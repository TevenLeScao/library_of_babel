import argparse
import json
import os

from tqdm import tqdm

from utils import count_occurences


def fixed_length_search(top_level_query, suffix, substring_length):
    whitespaced = top_level_query.split()
    max_len = len(whitespaced)
    queries = [" ".join(whitespaced[i:i+substring_length]) for i in range(0, max_len, substring_length)]
    queries[-1] = " ".join(whitespaced[len(whitespaced)-substring_length:len(whitespaced)])
    # print(queries)
    counts = [count_occurences(query, suffix, tokenize=args.tokenize) for query in queries]
    return any(counts)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Count occurances of sequence.')
    parser.add_argument('--suffix', type=str, required=True)
    parser.add_argument('--queries_folder', type=str)
    parser.add_argument('--tokenize', action='store_true')
    parser.add_argument('--length', type=int, default=16)

    args = parser.parse_args()

    flagged_per_task = {}

    for queries_file in os.listdir(args.queries_folder):
        print(queries_file)
        with open(os.path.join(args.queries_folder, queries_file)) as f:
            queries = json.load(f)

        if queries is None:
            continue

        flagged = 0

        for q in tqdm(queries):
            if fixed_length_search(q, args.suffix, args.length):
                flagged += 1

        flagged_per_task[queries_file] = flagged
        print(f"flagged {flagged} prompts")

        json.dump(flagged_per_task, open(os.path.join(args.queries_folder, "flagged_per_task.json"), "w"), ensure_ascii=False, indent=2)
