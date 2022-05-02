import argparse
import json
from functools import partial

import datasets
import numpy as np
import torch
from multiprocess import set_start_method
from sentence_transformers import SentenceTransformer


def embed_text(examples, rank, model, text_key, batch_size):
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(rank % torch.cuda.device_count())
    device = f"cuda:{rank % torch.cuda.device_count()}"
    model = model.to(device)
    embeddings = np.stack(
        model.encode(examples[text_key], show_progress_bar=False, convert_to_numpy=True,
                     batch_size=batch_size, device=device))
    examples["embedding"] = embeddings
    return examples


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # model args
    parser.add_argument("--dataset_name", default=None, type=str)
    parser.add_argument("--dataset_path", default=None, type=str)
    parser.add_argument("--dataset_format", default="json", type=str)
    parser.add_argument("--dataset_config", type=str)
    parser.add_argument("--model_name_or_path", default="roberta-base", type=str)
    parser.add_argument(f"--batch_size", default=16, type=int)
    parser.add_argument(f"--text_key", default="text", type=str)
    parser.add_argument("--output_path", type=str)
    # dataset args
    parser.add_argument("--subset", default=None, type=int)
    args = parser.parse_args()

    model = SentenceTransformer(args.model_name_or_path)
    if args.dataset_name:
        assert args.dataset_path is None, "Can't pass both dataset_name and dataset_path"
        dataset = datasets.load_dataset(args.dataset_name, args.dataset_config)["train"]
    elif args.dataset_path:
        dataset = datasets.load_dataset(args.dataset_format, data_files=args.dataset_path)["train"]
    else:
        raise NotImplementedError("You have to pass either dataset_name or dataset_path")

    if args.subset:
        dataset = dataset.shuffle(seed=1066)
        dataset = dataset.select(range(args.subset))
    set_start_method("spawn")
    dataset = dataset.map(
        partial(embed_text, model=model, text_key=args.text_key, batch_size=args.batch_size),
        batched=True, batch_size=args.batch_size, with_rank=True, num_proc=torch.cuda.device_count())
    dataset.to_json(args.output_path)
