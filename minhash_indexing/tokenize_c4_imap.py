import argparse
import json
from time import time
from multiprocessing import Pool
from functools import partial
import logging

import transformers


def batch_through_file(args):
    with open(args.input_file, "r") as input_file:
        batch = []
        for i, line in enumerate(input_file):
            batch.append(json.loads(line)["text"])
            if i % args.batch_size == 0 and i > 0:
                yield batch
                batch = []
        yield batch


def get_token_ids(batch, tokenizer):
    tokenized_batch = tokenizer(batch)["input_ids"]
    return "\n".join([" ".join([str(token) for token in sequence]) for sequence in tokenized_batch]) + "\n"


if __name__ == "__main__":
    logging.disable(logging.WARNING)

    parser = argparse.ArgumentParser(description='c4 tokenizer args')
    parser.add_argument("--input-file", type=str)
    parser.add_argument("--output-file", type=str)
    parser.add_argument("--tokenizer", type=str, default="t5-small")
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--log-interval", type=int, default=10)
    args = parser.parse_args()

    tokenizer = transformers.AutoTokenizer.from_pretrained(args.tokenizer)
    start_time = time()
    pool = Pool()
    with open(args.output_file, "w") as output_file:
        for i, ids_string in enumerate(pool.imap(partial(get_token_ids, tokenizer=tokenizer), batch_through_file(args))):
            if i > 0 and i % args.log_interval == 0:
                print(i, time() - start_time)
            output_file.write(ids_string)
