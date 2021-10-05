import argparse

from datasketch import MinHashLSHEnsemble, MinHash


def index_ngrams(line):
    ngram_set = set()
    min_hash = MinHash(num_perm=128)
    for d in ngram_set:
        min_hash.update(d)
    return min_hash

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='c4 tokenizer args')
    parser.add_argument("--num-perm", type=int, default=128)
    parser.add_argument("--num-part", type=int, default=32)
    parser.add_argument("--threshold", type=int, default=0.8)
    parser.add_argument("--input-file", type=str)
    args = parser.parse_args()

    min_hashes = []
    for line in args.input_file:
        ngram_set = index_ngrams(line)
        # Create MinHash objects

    # Create an LSH Ensemble index with threshold and number of partition
    # settings.
    lshensemble = MinHashLSHEnsemble(threshold=args.threshold, num_perm=args.num_perm, num_part=args.num_part)

    # Index takes an iterable of (key, minhash, size)
    lshensemble.index([("m2", m2, len(set2)), ("m3", m3, len(set3))])

    # Check for membership using the key
    print("m2" in lshensemble)
    print("m3" in lshensemble)

    # Using m1 as the query, get an result iterator
    print("Sets with containment > 0.8:")
    for key in lshensemble.query(m1, len(set1)):
        print(key)