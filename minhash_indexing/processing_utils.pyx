def extract_ngrams(line, n):
    cdef list ids
    cdef set output
    ids = line.split()
    output = set(tuple(ids[i:i + n]) for i in range(0, len(ids) - n + 1))
    return output
