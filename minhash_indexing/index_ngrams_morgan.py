import argparse
from functools import partial
from multiprocessing import Process, Queue
from os import PathLike
from pickle import dump
from time import time

from utils import split_file_from_size_and_workers


def extract_ngrams_from_line(text: str, n: int) -> set:
    ids = text.split()
    return set(tuple(ids[i:i + n]) for i in range(0, len(ids) - n + 1))


def extract_ngrams(pid: int, path: PathLike, n: int, f_offset_start: int, f_offset_end: int, queue: Queue,
                   max_queue_size: int):
    print(f"Starting process {pid} for chunk: {{{f_offset_start} -> {f_offset_end}}}")

    with open(path, "r") as f:
        f.seek(f_offset_start)
        chunk_len = f_offset_end - f_offset_start
        current_len = 0
        lines = 0
        stored_ngrams = set()

        while current_len < chunk_len:
            line = f.readline()
            lines += 1
            ngrams = extract_ngrams_from_line(line, n)
            current_len += len(line)

            # Send to the consumer
            if queue.qsize() > max_queue_size:
                stored_ngrams.update(ngrams)
            else:
                queue.put(stored_ngrams)
                stored_ngrams = set()

    # Signal we are done
    print(f"Read {lines} lines for pid {pid}")
    queue.put_nowait(None)


def save_ngrams_to_file(path: PathLike, queue: Queue, num_workers: int):
    ngrams = set()
    none_count = 0

    while True:
        grams = queue.get()
        if grams is not None:
            ngrams.update(grams)
        else:
            none_count += 1

        if none_count == num_workers:
            break

    with open(path, "wb") as f:
        print(f"Saving {len(ngrams)} NGrams to {path}")
        dump(ngrams, f)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='c4 tokenizer args')
    parser.add_argument("--input-file", type=str)
    parser.add_argument("--output-file", type=str)
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=59)
    parser.add_argument("--max-queue-size", type=int, default=10)
    args = parser.parse_args()

    # Chunk the file for each worker
    f_chunks = split_file_from_size_and_workers(args.input_file, args.num_workers)

    processes = []
    queue = Queue()

    total_ngram_set = set()
    ngram_sets = []
    start_time = time()

    # Create the consumer
    consumer = Process(target=save_ngrams_to_file,
                       args=(args.output_file, queue, args.num_workers, args.max_queue_size))
    consumer.start()

    fn_map = partial(extract_ngrams, path=args.input_file, n=args.n, queue=queue)
    for pid, (f_offset_start, f_offset_end) in enumerate(zip(f_chunks[:-1], f_chunks[1:])):
        process = Process(target=fn_map,
                          kwargs={"pid": pid, "f_offset_start": f_offset_start, "f_offset_end": f_offset_end,
                                  "queue": queue, "max_queue_size": args.max_queue_size})
        processes.append(process)

        process.start()

    for process in processes:
        process.join()

    consumer.join()

    print(f"Took: {time() - start_time}")

    # total_ngram_dict = {ngram: index for index, ngram in enumerate(total_ngram_set)}
    # with open(args.output_file, "wb") as g:
    #     pickle.dump(total_ngram_dict, g)
