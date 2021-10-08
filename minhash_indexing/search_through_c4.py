import argparse
import json
from functools import partial
from multiprocessing import Process, Queue
from os import PathLike
from time import time
import re

from utils import split_file_from_size_and_workers


def build_reverse(word):
    return (word[::-1],)


def find_nat_prompts_from_line(line, transfo_regexs):
    return {transfo_name: len(re.findall(transfo_regex, line)) for transfo_name, transfo_regex in transfo_regexs.items()}


def find_nat_prompts(pid: int, path: PathLike, f_offset_start: int, f_offset_end: int, queue: Queue, transfo_regexs: dict):
    print(f"Starting process {pid} for chunk: {{{f_offset_start} -> {f_offset_end}}}")

    with open(path, "r") as f:
        f.seek(f_offset_start)
        chunk_len = f_offset_end - f_offset_start
        current_len = 0
        lines = 0

        while current_len < chunk_len:
            line = f.readline()
            lines += 1
            document = find_nat_prompts_from_line(line, transfo_regexs)
            current_len += len(line)
            queue.put(document)

    # Signal we are done
    print(f"Read {lines} lines for pid {pid}")
    queue.put_nowait(None)


def save_results(path: PathLike, queue: Queue, num_workers: int):
    none_count = 0
    doc_count = 0
    while True:
        document = queue.get()
        if document is not None:
            doc_count += 1
        else:
            none_count += 1

        if none_count == num_workers:
            break

    print(f"found {doc_count} natural-prompting documents")


def unitary_regex(w1, w2):
    return rf"(^|\W+){w1}\W+{w2}($|\W+)|(^|\W+){w2}\W+{w1}($|\W+)"


def regex_from_test_words(corrupted_test_words):
    unitary_regexs = [unitary_regex(original, corrupted) for original, corrupted_list in corrupted_test_words for
                      corrupted in corrupted_list]
    print("|".join(unitary_regexs))
    return re.compile("|".join(unitary_regexs))


if __name__ == "__main__":

    transfos = {"reverse": build_reverse}
    parser = argparse.ArgumentParser(description='c4 tokenizer args')
    parser.add_argument("--input-file", type=str)
    parser.add_argument("--output-file", type=str)
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--max-queue-size", type=int, default=10)
    for transfo_name in transfos:
        parser.add_argument(f"--{transfo_name}", action="store_true")
    args = parser.parse_args()

    transfos = {transfo_name: transfo for transfo_name, transfo in transfos.items() if vars(args).get(transfo_name)}
    with open("../gpt3_tasks/test_words.json") as f:
        test_words = json.load(f)
    test_words = test_words[:10]
    transfo_test_words = {transfo_name: [(word, transfo(word)) for word in test_words] for transfo_name, transfo in
                          transfos.items()}
    transfo_regexs = {transfo_name: regex_from_test_words(corrupted_test_words) for transfo_name, corrupted_test_words
                      in transfo_test_words.items()}

    # Chunk the file for each worker
    f_chunks = split_file_from_size_and_workers(args.input_file, args.num_workers)

    processes = []
    queue = Queue()

    total_ngram_set = set()
    ngram_sets = []
    start_time = time()

    # Create the consumer
    consumer = Process(target=save_results,
                       args=(args.output_file, queue, args.num_workers))
    consumer.start()

    fn_map = partial(find_nat_prompts, path=args.input_file, queue=queue)
    for pid, (f_offset_start, f_offset_end) in enumerate(zip(f_chunks[:-1], f_chunks[1:])):
        process = Process(target=fn_map,
                          kwargs={"pid": pid, "f_offset_start": f_offset_start, "f_offset_end": f_offset_end,
                                  "queue": queue, "transfo_regexs": transfo_regexs})
        processes.append(process)

        process.start()

    for process in processes:
        process.join()

    consumer.join()

    print(f"Took: {time() - start_time}")
