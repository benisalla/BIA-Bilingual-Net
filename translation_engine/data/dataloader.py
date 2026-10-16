import json
from transformers import AutoTokenizer
import os
import torch
from random import shuffle
from itertools import groupby
from torch.nn.utils.rnn import pad_sequence


SPECIAL_TOKENS = {"pad_token": "<PAD>", "bos_token": "<SOS>", "eos_token": "<EOS>"}


def build_tokenizer(name: str = "gpt2"):
    """
    GPT-2 BPE tokenizer + <PAD>, <SOS>, <EOS>. Both languages share this vocabulary,
    which is what lets the model tie its embeddings. Training, fine-tuning and the
    app must all use this same tokenizer.
    """
    tokenizer = AutoTokenizer.from_pretrained(name)
    tokenizer.add_special_tokens(SPECIAL_TOKENS)
    tokenizer.model_max_length = int(1e9)  # we filter long sentences ourselves
    return tokenizer


class DataLoader(object):
    """
    A custom DataLoader for loading, processing, and batching data for sequence-to-sequence tasks using the GPT-2 tokenizer.

    This DataLoader reads data from a JSON file, tokenizes the input and target sequences, and creates batches
    of data suitable for training or evaluation of sequence models. Sentences of similar lengths are grouped
    so that each batch holds about `toks_in_batch` target tokens with little padding.

    Attributes:
        data_dir (str): The directory where the dataset is located.
        s_suffix (str): The key for the source sequences in the JSON data.
        t_suffix (str): The key for the target sequences in the JSON data.
        split (str): The dataset split to load (train, val, test, etc.).
        toks_in_batch (int): The maximum number of target tokens in each batch.
        tokenizer (AutoTokenizer): The tokenizer used for tokenizing sequences.
        max_seq_len (int, optional): Pairs whose source or target is longer than this are dropped
                                     (the model cannot embed positions beyond its max_seq_len).
        limit (int, optional): Only use the first `limit` pairs of the file.
        for_training (bool): Whether the DataLoader is in training mode (batches are shuffled every epoch).
        data (list): A list of tuples (source ids, target ids, source length, target length).
        batches (list): A list of batches, each containing a subset of the data.
        n_batches (int): The number of batches created.
        current_batch (int): The index of the current batch being processed.

    Methods:
        create_batches(): Creates batches of data by grouping sequences with similar lengths and shuffling them for training.
        __iter__(): Returns an iterator over the DataLoader (restarts from the first batch).
        __next__(): Returns the next batch of data, including padded source and target sequences and their lengths.
    """

    def __init__(
        self,
        data_dir,
        s_suffix,
        t_suffix,
        split,
        toks_in_batch,
        tokenizer=None,
        max_seq_len=None,
        limit=None,
    ):
        self.toks_in_batch = toks_in_batch
        assert split.lower() in {
            "train",
            "val",
            "test",
            "tiny_train",
            "tiny_val",
            "tiny_test",
        }, "'split' must be in ['train', 'val', 'test', 'tiny_train', 'tiny_val', 'tiny_test'] !"
        self.split = split.lower()
        self.s_suffix = s_suffix
        self.t_suffix = t_suffix
        self.for_training = self.split.endswith("train")  # training ?

        self.tokenizer = tokenizer if tokenizer is not None else build_tokenizer()
        bos, eos = self.tokenizer.bos_token_id, self.tokenizer.eos_token_id

        file_path = os.path.join(data_dir, f"{split}.json")
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if limit is not None:
            data = data[:limit]  # e.g. a few pairs for an overfitting sanity check

        # tokenize once here (not at every batch of every epoch)
        self.data = list()
        self.n_dropped = 0
        for item in data:
            s, t = item[s_suffix].strip(), item[t_suffix].strip()
            if not s or not t:
                self.n_dropped += 1
                continue
            if self.t_suffix == "ar":
                t = t[::-1]  # right-to-left target: decode with is_ltr=True
            s_ids = self.tokenizer.encode(s)
            t_ids = [bos] + self.tokenizer.encode(t) + [eos]
            if max_seq_len is not None and (len(s_ids) > max_seq_len or len(t_ids) > max_seq_len):
                self.n_dropped += 1
                continue
            # [0, 1, 2, 3] <==> [src_ids, tgt_ids, src_lengths, tgt_lengths]
            self.data.append((s_ids, t_ids, len(s_ids), len(t_ids)))

        # Pre-sort by tgt lengths - required for itertools.groupby() later
        self.data.sort(key=lambda x: x[3])

        self.create_batches()

    def create_batches(self):
        chunks = [
            list(t) for _, t in groupby(self.data, key=lambda x: x[3])
        ]  # chunk by tgt sequence lengths
        self.batches = list()
        for chunk in chunks:
            chunk.sort(
                key=lambda x: x[2]
            )  # (samples with same src sequence length) ==> (in one batch)
            seqs_per_batch = max(
                1, self.toks_in_batch // chunk[0][3]
            )  # toks_in_batch // tgt_length ==> how many sentences fit in toks_in_batch
            self.batches.extend(
                [
                    chunk[i : i + seqs_per_batch]
                    for i in range(0, len(chunk), seqs_per_batch)
                ]
            )  # chunk ==> batches

        if self.for_training:
            shuffle(self.batches)  # Randomly shuffle batchs
        self.n_batches = len(self.batches)
        self.current_batch = -1

    def __len__(self):
        return self.n_batches

    def __iter__(self):
        self.current_batch = -1
        return self

    def __next__(self):
        self.current_batch += 1
        try:
            s_data, t_data, s_lens, t_lens = zip(*self.batches[self.current_batch])
        except IndexError:
            raise StopIteration

        s_data = pad_sequence(
            sequences=[torch.LongTensor(s) for s in s_data],
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )
        t_data = pad_sequence(
            sequences=[torch.LongTensor(t) for t in t_data],
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )

        s_lens = torch.LongTensor(s_lens)
        t_lens = torch.LongTensor(t_lens)

        return s_data, t_data, s_lens, t_lens
