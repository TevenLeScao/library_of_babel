from os import PathLike, linesep
from os.path import getsize
from typing import Set


def split_file_from_size_and_workers(path: PathLike, num_workers: int):
    assert num_workers > 0, f"Cannot determine number of workers (got {num_workers})"

    # Retrieve the size of the file
    f_size = getsize(path)

    # Divide the size of the file to dispatch a slice to each worker
    f_chunk_size = f_size // num_workers

    # We need to round up the chunk so it maps an entire line
    chunks_offset, current_pos = [0], 0
    with open(path) as f:
        while current_pos <= f_size:
            # Size of the current chunk
            chunk_size = f_chunk_size

            # Move to the position of the next chunk according to current position in file
            f.seek(current_pos + f_chunk_size)

            # Look for line termination (i.e. read the remaining of the line)
            remaining = f.readline()

            # Add to the chunk
            chunk_size += len(remaining)
            current_pos += chunk_size

            # Append to the offsets list
            chunks_offset.append(min(current_pos, f_size))

    return chunks_offset


def count_line_buffered(path: PathLike) -> int:
    lines = 0

    with open(path) as f:
        buf_size = 4 * 1024  # 4kB
        read_f = f.read  # loop optimization

        while buf := read_f(buf_size):
            lines += buf.count(linesep)

    return lines
