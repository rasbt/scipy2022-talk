import random

import pytorch_lightning as pl
import torch
import torchtext
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from torchvision import datasets, transforms


class Cifar10DataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size,
        train_transform=None,
        test_transform=None,
        num_workers=4,
        data_path="./",
    ):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.custom_train_transform = train_transform
        self.custom_test_transform = test_transform

    def prepare_data(self):
        datasets.CIFAR10(root=self.data_path, download=True)
        return

    def setup(self, stage=None):

        if self.custom_train_transform is None:
            self.train_transform = transforms.Compose(
                [
                    transforms.Resize((70, 70)),
                    transforms.RandomCrop((64, 64)),
                    transforms.ToTensor(),
                ]
            )
        else:
            self.train_transform = self.custom_train_transform

        if self.custom_train_transform is None:
            self.test_transform = transforms.Compose(
                [
                    transforms.Resize((70, 70)),
                    transforms.CenterCrop((64, 64)),
                    transforms.ToTensor(),
                ]
            )
        else:
            self.test_transform = self.custom_test_transform

        train = datasets.CIFAR10(
            root=self.data_path,
            train=True,
            transform=self.train_transform,
            download=False,
        )

        self.test = datasets.CIFAR10(
            root=self.data_path,
            train=False,
            transform=self.test_transform,
            download=False,
        )

        self.train, self.valid = random_split(train, lengths=[45000, 5000])

    def train_dataloader(self):
        train_loader = DataLoader(
            dataset=self.train,
            batch_size=self.batch_size,
            drop_last=True,
            shuffle=True,
            persistent_workers=True,
            num_workers=self.num_workers,
        )
        return train_loader

    def val_dataloader(self):
        valid_loader = DataLoader(
            dataset=self.valid,
            batch_size=self.batch_size,
            drop_last=False,
            persistent_workers=True,
            shuffle=False,
            num_workers=self.num_workers,
        )
        return valid_loader

    def test_dataloader(self):
        test_loader = DataLoader(
            dataset=self.test,
            batch_size=self.batch_size,
            drop_last=False,
            persistent_workers=True,
            shuffle=False,
            num_workers=self.num_workers,
        )
        return test_loader


def get_trip_advisor_datasetloaders(
    csv_path, random_seed, vocab_size, batch_size, device
):

    TEXT = torchtext.legacy.data.Field(
        tokenize="spacy",  # default splits on whitespace
        tokenizer_language="en_core_web_sm",
        include_lengths=True,
    )

    LABEL = torchtext.legacy.data.LabelField(dtype=torch.long)

    fields = [("TEXT_COLUMN_NAME", TEXT), ("LABEL_COLUMN_NAME", LABEL)]

    dataset = torchtext.legacy.data.TabularDataset(
        path=csv_path, format="csv", skip_header=True, fields=fields
    )

    train_data, test_data = dataset.split(
        split_ratio=[0.8, 0.2], random_state=random.seed(random_seed)
    )

    train_data, valid_data = train_data.split(
        split_ratio=[0.85, 0.15], random_state=random.seed(random_seed)
    )

    TEXT.build_vocab(train_data, max_size=vocab_size)
    LABEL.build_vocab(train_data)

    (
        train_loader,
        valid_loader,
        test_loader,
    ) = torchtext.legacy.data.BucketIterator.splits(
        (train_data, valid_data, test_data),
        device=device,
        batch_size=batch_size,
        sort_within_batch=True,  # necessary for packed_padded_sequence
        sort_key=lambda x: len(x.TEXT_COLUMN_NAME),
    )

    return train_loader, valid_loader, test_loader
