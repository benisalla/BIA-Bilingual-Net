import json
from transformers import AutoTokenizer
import os
import torch
from random import shuffle
from itertools import groupby
from torch.nn.utils.rnn import pad_sequence


class DataLoader(object):
    """
    A custom DataLoader for loading, processing, and batching data for sequence-to-sequence tasks using the GPT-2 tokenizer.

    This DataLoader reads data from a JSON file, tokenizes the input and target sequences, and creates batches
    of data suitable for training or evaluation of sequence models. It supports custom batch creation strategies,
    including sorting by sequence lengths for efficient batching and padding.

    Attributes:
        data_dir (str): The directory where the dataset is located.
        s_suffix (str): The key for the source sequences in the JSON data.
        t_suffix (str): The key for the target sequences in the JSON data.
        split (str): The dataset split to load (train, val, test, etc.).
        toks_in_batch (int): The maximum number of tokens in each batch.
        tokenizer (AutoTokenizer): The tokenizer used for tokenizing sequences.
        for_training (bool): Whether the DataLoader is in training mode, which affects how batches are created.
        data (list): A list of tuples containing source sequences, target sequences, and their lengths.
        batches (list): A list of batches, each containing a subset of the data.
        n_batches (int): The number of batches created.
        current_batch (int): The index of the current batch being processed.

    Methods:
        create_batches(): Creates batches of data by grouping sequences with similar lengths and shuffling them for training.
        __iter__(): Returns an iterator over the DataLoader.
        __next__(): Returns the next batch of data, including padded source and target sequences and their lengths.
    """

    def __init__(self, data_dir, s_suffix, t_suffix, split, toks_in_batch):
        self.toks_in_batch = toks_in_batch
        assert split.lower() in {
            "train",
            "val",
            "test",
            "tiny_train",
            "tiny_val",
            "tiny_test",
        }, "'split' must be in ['train', 'val', 'test'] !"
        self.split = split.lower()
        self.s_suffix = s_suffix
        self.t_suffix = t_suffix
        self.for_training = True if self.split == "train" else False  # training ?

        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.tokenizer.add_special_tokens(
            {"pad_token": "<PAD>", "bos_token": "<SOS>", "eos_token": "<EOS>"}
        )

        file_path = os.path.join(data_dir, f"{split}.json")
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        s_data, t_data = [], []
        for item in data:
            s_data.append(item[s_suffix])
            t_data.append(item[t_suffix])

        s_lens = [len(self.tokenizer.encode(s)) for s in s_data]
        t_lens = [len(self.tokenizer.encode("<SOS>" + t + "<EOS>")) for t in t_data]

        # [0, 1, 2, 3] <==> [src_data, tgt_data, src_lengths, tgt_lengths]
        self.data = list(zip(s_data, t_data, s_lens, t_lens))

        # Pre-sort by tgt lengths - required for itertools.groupby() later
        if self.for_training:
            self.data.sort(key=lambda x: x[3])

        self.create_batches()

    def create_batches(self):
        if self.for_training:
            chunks = [
                list(t) for _, t in groupby(self.data, key=lambda x: x[3])
            ]  # chunk by tgt sequence lengths
            self.batches = list()
            for chunk in chunks:
                chunk.sort(
                    key=lambda x: x[2]
                )  # (samples with same src sequence length) ==> (in one batch)
                seqs_per_batch = (
                    self.toks_in_batch // chunk[0][3]
                )  # toks_in_batch // max(src_lengths) ==> How many max_length(src_data) in toks_in_batch
                self.batches.extend(
                    [
                        chunk[i : i + seqs_per_batch]
                        for i in range(0, len(chunk), seqs_per_batch)
                    ]
                )  # chunk ==> batches

            shuffle(self.batches)  # Randomly shuffle batchs
            self.n_batches = len(self.batches)
            self.current_batch = -1
        else:
            self.batches = [[d] for d in self.data]
            self.n_batches = len(self.batches)
            self.current_batch = -1

    def __iter__(self):
        return self

    def __next__(self):
        self.current_batch += 1
        try:
            s_data, t_data, s_lens, t_lens = zip(*self.batches[self.current_batch])
        except IndexError:
            raise StopIteration

        s_data = [self.tokenizer.encode(s) for s in s_data]

        if self.t_suffix == "ar":
            t_data = [
                self.tokenizer.encode("<SOS>" + t[::-1] + "<EOS>") for t in t_data
            ]
        else:
            t_data = [self.tokenizer.encode("<SOS>" + t + "<EOS>") for t in t_data]

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
