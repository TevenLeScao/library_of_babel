import os
import wget
from tqdm import tqdm


def download_dataset(name, split, output_dir):
    url = f"https://storage.googleapis.com/bigscience/experiment_d/experiment_d_cached_tasks/v0.2/{name}/{split}.tfrecord-00000-of-00001"
    try:
        wget.download(url, os.path.join(output_dir, f"{name}.{split}.tfrecord"))
    except:
        print(url)



FILE_LIST = ["d4_eval.txt"]

if __name__ == "__main__":

    output_dir = "input_sequences"
    os.makedirs(output_dir, exist_ok=True)

    for file in FILE_LIST:
        with open(file) as f:
            for line in tqdm(f):
                download_dataset(name=line[:-1], split="test", output_dir=output_dir)
