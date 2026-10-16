"""
Download a Darija <-> English parallel corpus and write the json files the DataLoader reads.

    python -m translation_engine.data.prepare_data

Source: the DODa sentences (Darija Open Dataset, Darija in Latin script) as packaged in
https://huggingface.co/datasets/atlasia/darija_english (config "doda", ~45k pairs).
Output (keys "dr" and "en", the default s_suffix / t_suffix):

    translation_engine/src/dataset/{train,val,test}.json
    translation_engine/src/dataset/{tiny_train,tiny_val,tiny_test}.json   (small copies for quick checks)

Any other corpus works too: write a json list of {"dr": "...", "en": "..."} objects.
"""

import argparse
import csv
import io
import json
import os
import random
import urllib.request

DODA_URL = "https://huggingface.co/datasets/atlasia/darija_english/resolve/main/atlasia_doda.csv"


def download_pairs(url: str = DODA_URL) -> list:
    """Return the list of {"dr", "en"} pairs from the DODa csv (empty rows and duplicates removed)."""
    with urllib.request.urlopen(url, timeout=120) as response:
        text = response.read().decode("utf-8")
    pairs, seen = [], set()
    for row in csv.DictReader(io.StringIO(text)):
        dr = " ".join((row.get("darija") or "").split())
        en = " ".join((row.get("en") or "").split())
        if dr and en and (dr, en) not in seen:
            seen.add((dr, en))
            pairs.append({"dr": dr, "en": en})
    return pairs


def split_pairs(pairs: list, val_ratio: float, test_ratio: float, seed: int) -> dict:
    """Shuffle once with a fixed seed, then cut into train / val / test."""
    pairs = list(pairs)
    random.Random(seed).shuffle(pairs)
    n_val = int(len(pairs) * val_ratio)
    n_test = int(len(pairs) * test_ratio)
    return {
        "val": pairs[:n_val],
        "test": pairs[n_val : n_val + n_test],
        "train": pairs[n_val + n_test :],
    }


def write_splits(splits: dict, out_dir: str, tiny_size: int) -> None:
    os.makedirs(out_dir, exist_ok=True)
    for name, items in splits.items():
        for prefix, subset in (("", items), ("tiny_", items[:tiny_size])):
            path = os.path.join(out_dir, f"{prefix}{name}.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(subset, f, ensure_ascii=False, indent=0)
            print(f"{path}: {len(subset):,} pairs")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", default="./translation_engine/src/dataset")
    parser.add_argument("--url", default=DODA_URL)
    parser.add_argument("--val-ratio", type=float, default=0.03)
    parser.add_argument("--test-ratio", type=float, default=0.03)
    parser.add_argument("--tiny-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    pairs = download_pairs(args.url)
    print(f"downloaded {len(pairs):,} unique pairs")
    write_splits(split_pairs(pairs, args.val_ratio, args.test_ratio, args.seed), args.out_dir, args.tiny_size)


if __name__ == "__main__":
    main()
