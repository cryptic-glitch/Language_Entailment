import dataclasses
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from itertools import chain
import json
import numpy as np
from torch import nn
from typing import Any, List, Optional
from string import punctuation
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
from itertools import combinations

from textual_graphs import Build_Fragments
from helper_fn import Collate, Metrics

device: Any = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ET_Dataset(Dataset):
    maptovector = {"neutral": [0, 1, 0],
                   "contradiction": [1, 0, 0],
                   "entailment": [0, 0, 1]}

    def __init__(self, tokenizer, train_path: str, val_path: str, load_val=False, sample_transform=True):
        self.train_path = train_path
        self.val_path = val_path
        self.sample_transform = sample_transform

        if load_val:
            with open(self.val_path, 'r') as f:
                self.d = list(f)[:1000]
        else:
            with open(self.train_path, 'r') as f:
                self.d = list(f)[:3000]
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.d)

    def __getitem__(self, item):
        dic = json.loads(self.d[item])
        sentence1 = dic["sentence1"]
        sentence2 = dic["sentence2"]
        if self.sample_transform:
            b1 = Build_Fragments(sentence1)
            b2 = Build_Fragments(sentence2)
            fragments2 = b2.show_sentences()
            entity1 = b1.get_entities(sentence1)
            entity2 = b2.get_entities(sentence2)
            # find if there is anything common if not use full sentences
            set_1 = set(list(filter(None, list(chain(*[i.split(" ") for i in entity1])))))
            set_2 = set(list(filter(None, list(chain(*[i.split(" ") for i in entity2])))))
            if set_1.intersection(set_2):
                if list(chain(*fragments2)):
                    frag_set2 = [set(i[0].split(" ")) for i in b2.show_sentences()]
                    intersect = np.array([len(set_2.intersection(i)) for i in frag_set2])
                    choose_2 = np.where(intersect == len(set_2))[0]
                    if choose_2.size:
                        sentence2 = fragments2[choose_2[0]][0]

        inputs = self.tokenizer(f"[CLS] {sentence1} [SEP] {sentence2} [SEP]", return_tensors="pt")
        label = dic["annotator_labels"][0]
        return {"input_ids": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"],
                "target": self.maptovector[label]
                }


class Deberta(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained("../input/debertamodel/local-pt-checkpoint").to(
            device)
        for p in self.model.base_model.parameters():
            p.requires_grad_(requires_grad=False)

    def forward(self, **kwargs):
        return self.model(**kwargs)


def fit_one_epoch(model, loss_fn, train_loader, optimizer, metrics, epoch):
    metrics.reset_metrics()
    epoch_loss = []
    model.train()
    pbar = tqdm(train_loader)
    pbar.set_description(f"Epoch_{epoch}")
    # Iterate over the batches of the dataset
    for inputs, targets in pbar:
        optimizer.zero_grad()
        if torch.cuda.is_available():
            inputs = {key: val.cuda() for key, val in inputs.items()}
            targets = targets.cuda()

        preds = F.softmax(model(**inputs).logits)
        loss = loss_fn(preds, targets)
        metrics.compute_accuracy(gt=targets, preds=preds)
        epoch_loss.append(loss.tolist())
        loss.backward()
        optimizer.step()
        pbar.set_postfix(loss=loss, acc=metrics.get_acc())
    e_loss = sum(epoch_loss) / len(epoch_loss)


def infer_epoch(model, loss_fn, val_loader, metrics, epoch):
    metrics.reset_metrics()
    epch_val_loss = []
    model.eval()
    pbar = tqdm(val_loader)
    pbar.set_description(f"Val_{epoch}")
    for inputs, targets in pbar:
        if torch.cuda.is_available():
            inputs = {key: val.cuda() for key, val in inputs.items()}
            targets = targets.cuda()
        preds = model(**inputs).logits
        loss = loss_fn(preds, targets)
        epch_val_loss.append(loss.tolist())
        metrics.compute_accuracy(gt=targets, preds=preds)
        pbar.set_postfix(loss=loss, acc=metrics.get_acc())
    e_loss = sum(epch_val_loss) / len(epch_val_loss)


@dataclasses.dataclass
class Args():
    train_path: str = '../input/train-set/snli_1.0_train.jsonl'
    val_path: str = '../input/test-set/snli_1.0_test.jsonl'
    epochs: int = 20
    lr: float = 0.0001
    train_bs: int = 12
    val_bs: int = 64
    sample_transform: bool = True
    train_size: int = 3000
    val_size: int = 2000


def main():
    # hp
    args = Args()

    # define model
    model = Deberta()

    # metrics
    metrics = Metrics()
    tokenizer = AutoTokenizer.from_pretrained("../input/debertatokenizer/local-pt-tokenizer")

    # batching
    data_collator = Collate(tokenizer=tokenizer)
    train_dataset = ET_Dataset(tokenizer=tokenizer, train_path=args.train_path,
                               val_path=args.val_path, sample_transform=args.sample_transform)
    train_dataloader = DataLoader(train_dataset, batch_size=args.train_bs, collate_fn=data_collator)
    val_dataset = ET_Dataset(tokenizer=tokenizer, train_path=args.train_path,
                             val_path=args.val_path, load_val=True, sample_transform=False)
    val_dataloader = DataLoader(val_dataset, batch_size=args.val_bs, collate_fn=data_collator)

    # loss
    def criterion(outputs, labels):
        return nn.CrossEntropyLoss()(outputs, labels)

    # Optimizer
    optimizer = optim.SGD([p for p in model.parameters() if p.requires_grad],
                          lr=args.lr)

    for i in range(args.epochs):
        fit_one_epoch(model, criterion, train_dataloader, optimizer, metrics, i)
        infer_epoch(model, criterion, val_dataloader, metrics, i)


if __name__ == "__main__":
    main()
