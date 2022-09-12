import dataclasses

from torch.utils.data import DataLoader, Dataset
import json
import numpy as np
import os
from torch import nn
from typing import Any, List, Optional, Callable, Dict
import torch.nn.functional as F
from loguru import logger
import torch.optim as optim
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from helper_fn import Metrics, Collate
from itertools import chain
from textual_graphs import Build_Fragments

device: Any = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"using {device}")

class ET_Dataset(Dataset):

    maptovector = {
        "neutral": [0, 1, 0],
        "contradiction": [1, 0, 0],
        "entailment": [0, 0, 1],
    }

    def __init__(
        self,
        tokenizer,
        train_path: str,
        val_path: str,
        load_val=False,
        sample_transform=False, #default value
    ):
        self.train_path = train_path
        self.val_path = val_path
        self.sample_transform = sample_transform

        if load_val:
            with open(self.val_path, "r") as f:
                self.d = list(f)[:1000]
        else:
            with open(self.train_path, "r") as f:
                self.d = list(f)[:3000]
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.d)

    def __getitem__(self, item: int) -> Dict[str, Any]:
        dic = json.loads(self.d[item])
        sentence1 = dic["sentence1"]
        sentence2 = dic["sentence2"]
        if self.sample_transform:
            b1 = Build_Fragments(sentence1)
            b2 = Build_Fragments(sentence2)
            fragments1 = b1.show_sentences()
            fragments2 = b2.show_sentences()
            entity1 = b1.get_entities(sentence1)
            entity2 = b2.get_entities(sentence2)
            # find if there is anything common if not use full sentences
            set_1 = set(
                list(filter(None, list(chain(*[i.split(" ") for i in entity1]))))
            )
            set_2 = set(
                list(filter(None, list(chain(*[i.split(" ") for i in entity2]))))
            )
            if set_1.intersection(set_2):
                if list(chain(*fragments2)):
                    frag_set2 = [set(i[0].split(" ")) for i in b2.show_sentences()]
                    intersect = np.array(
                        [len(set_2.intersection(i)) for i in frag_set2]
                    )
                    choose_2 = np.where(intersect == len(set_2))[0]
                    if choose_2.size:
                        sentence2 = fragments2[choose_2[0]][0]

        inputs = self.tokenizer(
            f"[CLS] {sentence1} [SEP] {sentence2} [SEP]", return_tensors="pt"
        )
        label = dic["annotator_labels"][0]
        return {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "target": self.maptovector[label],
        }


class Deberta(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "/home/pop/Language_Entailment_Proj/Language_Entailment/local-pt-checkpoint"
        )

    def forward(self, **kwargs) -> torch.Tensor:
        return self.model(**kwargs)


def fit_one_epoch(model: torch.nn.Module, loss_fn: Callable, train_loader: DataLoader, optimizer: torch.optim.Optimizer, metrics: Metrics, epoch: int, run: Any=False) -> None:
    metrics.reset_metrics()
    epoch_loss = []
    model.train()
    pbar = tqdm(train_loader)
    pbar.set_description(f"Train_{epoch}")
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
        if run:
            run.log({"train/step_loss": loss, "train/acc": metrics.get_acc()})
    e_loss = sum(epoch_loss) / len(epoch_loss)
    if run:
        run.log({"train/epch_loss": e_loss, "train/epch_acc": metrics.get_acc()})


def val_epoch(model: torch.nn.Module, loss_fn: Callable, val_loader: DataLoader, metrics: Metrics, epoch: int, run: Any=False) -> None:
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
        if run:
            run.log({"val/step_loss": loss, "val/acc": metrics.get_acc()})

    e_loss = sum(epch_val_loss) / len(epch_val_loss)
    if run:
        run.log({"val/epch_loss": e_loss, "val/epch_acc": metrics.get_acc()})


def main(args) -> None:

    # logger
    wandb = False
    if args.wandb:
        import wandb

        os.environ["WANDB_API_KEY"] = args.api_key
        wandb.init(project="Entailment", entity="prashantbahuguna")
        logger.success("added external logger")
    else:
        logger.info("no external logger found.")

    # define model
    model = Deberta()

    # metrics
    metrics = Metrics()
    tokenizer = AutoTokenizer.from_pretrained("/home/pop/Language_Entailment_Proj/Language_Entailment/local-pt-tokenizer")

    # collate_fn
    data_collator = Collate(tokenizer=tokenizer)
    train_dataset = ET_Dataset(
        tokenizer=tokenizer,
        train_path=args.train_path,
        val_path=args.val_path,
        sample_transform=args.sample_transform,
    )
    train_dataloader = DataLoader(
        train_dataset, batch_size=args.train_bs, collate_fn=data_collator
    )
    val_dataset = ET_Dataset(
        tokenizer=tokenizer,
        train_path=args.train_path,
        val_path=args.val_path,
        load_val=True,
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=args.val_bs, collate_fn=data_collator
    )

    # loss
    def criterion(outputs, labels):
        return nn.CrossEntropyLoss()(outputs, labels)

    # Optimizer
    optimizer = optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    logger.info("commencing training")
    for i in range(args.epochs):
        fit_one_epoch(model, criterion, train_dataloader, optimizer, metrics, i, wandb)
        val_epoch(model, criterion, val_dataloader, metrics, i, wandb)
    logger.succes("training completed")


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(
        description="Training script for Textual entailement"
    )
    parser.add_argument("train_path", type=str)
    parser.add_argument("val_path", type=str)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--train_bs", type=int, default=128)
    parser.add_argument("--val_bs", type=int, default=12)
    parser.add_argument("--sample_transform", type=bool, default=False)
    parser.add_argument("--wandb", type=bool, default=False)
    parser.add_argument(
        "--wd",
        "--weight-decay",
        default=0,
        type=float,
        help="weight decay",
        dest="weight_decay",
    )
    parser.add_argument("--lr", type=float, default=0.0001)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args_ = parse_args()
    main(args_)
