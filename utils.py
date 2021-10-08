import os
import warnings

import numpy as np


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