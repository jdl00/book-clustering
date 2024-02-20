import os

import torch
import pandas as pd
from torch.utils.data import Dataset
from transformers import AutoTokenizer


DATASET_FOLDER_PATH = os.path.abspath("dataset")
TOKENISER_SRING = "deepset/bert-base-cased-squad2"


class Tokeniser:
    def __init__(self) -> None:
        self._tokenizer = AutoTokenizer.from_pretrained(TOKENISER_SRING)
        self._tokenizer.pad_token_id = 0

    def tokenise(self, sequence: str):
        inputs = self._tokenizer(
            sequence,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=512,
        )

        return inputs["input_ids"]


class BookDataset(Dataset):
    def __init__(self, dataset: str, device) -> None:
        self.data = []
        self._dataset_path = os.path.join(DATASET_FOLDER_PATH, dataset)
        self._tokeniser = Tokeniser()
        self._device = device

        desc_df = pd.read_csv(self._dataset_path, dtype=str)
        for row in desc_df["desc"]:
            tokenized_text = self._tokeniser.tokenise(row)
            tokenized_text = tokenized_text.to(self._device).squeeze(0)
            self.data.append(tokenized_text)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]
