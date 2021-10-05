import tensorflow as tf
import json
import os
from tqdm import tqdm


def load_cached_task(cache_dir, split):
    with tf.io.gfile.GFile(os.path.join(cache_dir, f"info.{split}.json")) as f:
        split_info = json.load(f)
        features = split_info["features"]

    # Use `FixedLenSequenceFeature` for sequences with variable length.
    def _feature_config(shape, dtype):
        if dtype in ("int32", "bool"):
            # int32 and bool are stored as int64 in the tf.train.Example protobuf.
            # TODO(adarob): Support other conversions.
            dtype = "int64"
        if shape and shape[0] is None:
            return tf.io.FixedLenSequenceFeature(
                shape[1:], dtype, allow_missing=True)
        return tf.io.FixedLenFeature(shape, dtype)

    feature_description = {
        feat: _feature_config(**desc) for feat, desc in features.items()
    }

    tfrecords = os.path.join(
        cache_dir, f"{split}.tfrecord-*-of-*{split_info['num_shards']}"
    )
    ds = tf.data.TFRecordDataset(tf.io.gfile.glob(tfrecords))
    ds = ds.map(
        lambda pb: tf.io.parse_single_example(pb, feature_description),
        num_parallel_calls=tf.data.experimental.AUTOTUNE)
    # Cast features back to the types from the info JSON since some features
    # must be cast for storage (e.g., in32 is stored as int64).
    ds = ds.map(
        lambda x: {k: tf.cast(v, features[k]["dtype"]) for k, v in x.items()},
        num_parallel_calls=tf.data.experimental.AUTOTUNE)
    return ds


FILE_LIST = ["zero_shot_eval.txt"]

if __name__ == "__main__":

    output_dir = "zero_shot_sequences"
    os.makedirs(output_dir, exist_ok=True)

    for file in FILE_LIST:
        with open(file) as f:
            for line in tqdm(f):
                task_name = line[:-1]
                try:
                    split = "test"
                    output_file = os.path.join(output_dir, f"{task_name}.{split}.json")
                    if os.path.isfile(output_file):
                        continue
                    ds = load_cached_task(f"gs://bigscience/experiment_d/experiment_d_cached_tasks/v0.2/{task_name}/", split)
                    queries = [ex["inputs_pretokenized"].decode("utf8") for ex in ds.as_numpy_iterator()]
                except:
                    try:
                        split = "validation"
                        output_file = os.path.join(output_dir, f"{task_name}.{split}.json")
                        if os.path.isfile(output_file):
                            continue
                        ds = load_cached_task(f"gs://bigscience/experiment_d/experiment_d_cached_tasks/v0.2/{task_name}/", split)
                        queries = [ex["inputs_pretokenized"].decode("utf8") for ex in ds.as_numpy_iterator()]
                    except:
                        queries = None
                with open(output_file, "w") as g:
                    json.dump(queries, g, ensure_ascii=False, indent=2)
